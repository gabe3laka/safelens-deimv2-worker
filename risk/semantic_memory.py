"""
risk/semantic_memory.py -- Short-lived per-session semantic label cache.

Safety rules (hard):
  * Stores ONLY small text metadata (semantic_label, confidence, timestamps).
  * Never stores images, crops, base64 frames, or full Gemini responses.
  * Semantic labels are keyed by session_id + track_id/detection_id/entity_id.
  * Semantic labels NEVER change risk_level, risk_score, hazard_type, or should_alert.
  * Labels expire after SEMANTIC_LABEL_CACHE_TTL_MS (default 15 000 ms).
  * SEMANTIC_LABEL_APPLY_ENABLED=false (default) = shadow mode:
      labels are stored + logged but NOT written to entities.

Rollout env flags (all have safe defaults):
  SEMANTIC_LABEL_CACHE_ENABLED       true
  SEMANTIC_LABEL_APPLY_ENABLED       false      <- shadow mode by default
  SEMANTIC_LABEL_CACHE_TTL_MS        15000
  SEMANTIC_LABEL_MIN_CONFIDENCE      0.55
  SEMANTIC_LABEL_MAX_PER_SESSION     64
  SEMANTIC_LABEL_BBOX_FALLBACK_MAX_AGE_MS   3000
  SEMANTIC_LABEL_BBOX_FALLBACK_IOU_MIN      0.50
  SEMANTIC_LABEL_BBOX_FALLBACK_CENTER_MAX   0.08
"""

from __future__ import annotations

import logging
import os
import re
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger("safelens-vision-worker.semantic_memory")

# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _cache_enabled() -> bool:
    return os.getenv("SEMANTIC_LABEL_CACHE_ENABLED", "true").strip().lower() in (
        "1", "true", "yes", "on"
    )


def _apply_enabled() -> bool:
    return os.getenv("SEMANTIC_LABEL_APPLY_ENABLED", "false").strip().lower() in (
        "1", "true", "yes", "on"
    )


def _ttl_ms() -> int:
    try:
        return max(1000, int(os.getenv("SEMANTIC_LABEL_CACHE_TTL_MS", "15000")))
    except (TypeError, ValueError):
        return 15000


def _min_confidence() -> float:
    try:
        return float(os.getenv("SEMANTIC_LABEL_MIN_CONFIDENCE", "0.55"))
    except (TypeError, ValueError):
        return 0.55


def min_confidence() -> float:
    """Public accessor for SEMANTIC_LABEL_MIN_CONFIDENCE threshold."""
    return _min_confidence()


def _max_per_session() -> int:
    try:
        return max(1, int(os.getenv("SEMANTIC_LABEL_MAX_PER_SESSION", "64")))
    except (TypeError, ValueError):
        return 64


def _bbox_fallback_max_age_ms() -> int:
    try:
        return max(0, int(os.getenv("SEMANTIC_LABEL_BBOX_FALLBACK_MAX_AGE_MS", "3000")))
    except (TypeError, ValueError):
        return 3000


def _bbox_fallback_iou_min() -> float:
    try:
        return float(os.getenv("SEMANTIC_LABEL_BBOX_FALLBACK_IOU_MIN", "0.50"))
    except (TypeError, ValueError):
        return 0.50


def _bbox_fallback_center_max() -> float:
    try:
        return float(os.getenv("SEMANTIC_LABEL_BBOX_FALLBACK_CENTER_MAX", "0.08"))
    except (TypeError, ValueError):
        return 0.08


# ---------------------------------------------------------------------------
# Sanitizer
# ---------------------------------------------------------------------------

# Reject labels containing any of these risk words (word-boundary check).
# Single-word entries are matched with word boundaries; multi-word entries
# are matched as substrings (e.g. "near edge" covers the phrase).
_RISK_WORDS = frozenset({
    "danger", "dangerous", "hazard", "hazardous", "unsafe", "risk",
    "yellow", "orange", "red", "fall", "falling", "edge", "near edge",
    "warning", "alert",
})

# Allowed characters: letters, digits, spaces, hyphens, forward slashes.
_ALLOWED_RE = re.compile(r"^[a-z0-9 \-/]+$")

# Reject if label looks like a sentence (contains punctuation or many words).
_SENTENCE_RE = re.compile(r"[.,;:!?()\"']")
_MAX_WORDS = 6


def sanitize_label(label: Optional[str]) -> Optional[str]:
    """Sanitize a semantic label string.

    Returns the cleaned label, or None if it should be rejected.

    Rules:
      - trim + lowercase
      - max 48 characters
      - allow only letters/numbers/spaces/hyphens/slashes
      - reject empty labels
      - reject labels containing risk words
      - reject labels that look like full sentences
    """
    if not label:
        return None
    cleaned = label.strip().lower()[:48]
    if not cleaned:
        return None
    # Reject if contains sentence punctuation.
    if _SENTENCE_RE.search(cleaned):
        return None
    # Reject if too many words (likely a sentence).
    if len(cleaned.split()) > _MAX_WORDS:
        return None
    # Reject if contains risk words (whole-word match).
    for rw in _RISK_WORDS:
        pattern = r"(?<![a-z])" + re.escape(rw) + r"(?![a-z])"
        if re.search(pattern, cleaned):
            return None
    # Reject if contains disallowed characters.
    if not _ALLOWED_RE.match(cleaned):
        return None
    return cleaned


# ---------------------------------------------------------------------------
# Cache storage
# ---------------------------------------------------------------------------
# Structure:
#   _CACHE[session_id][entity_key] = {
#       "semantic_label": str,
#       "confidence": float,
#       "last_seen_ms": int,
#       "expires_at_ms": int,
#       "source": str,
#   }

_LOCK = threading.RLock()
_CACHE: Dict[str, Dict[str, Dict[str, Any]]] = {}


def _now_ms() -> int:
    return int(time.time() * 1000)


def _entity_key(entity: Dict[str, Any]) -> Optional[str]:
    """Derive entity key from track_id > detection_id > entity_id."""
    tid = entity.get("track_id")
    if tid is not None:
        return f"track_{tid}"
    did = entity.get("detection_id")
    if did is not None:
        return f"det_{did}"
    eid = entity.get("entity_id") or entity.get("id")
    if eid is not None:
        return f"ent_{eid}"
    return None


def _entity_key_from_ids(
    track_id: Any = None,
    detection_id: Any = None,
    entity_id: Any = None,
) -> Optional[str]:
    """Derive entity key from explicit ID arguments."""
    if track_id is not None:
        return f"track_{track_id}"
    if detection_id is not None:
        return f"det_{detection_id}"
    if entity_id is not None:
        return f"ent_{entity_id}"
    return None


def _expire_session(session_entries: Dict[str, Dict[str, Any]], now_ms: int) -> None:
    """Remove expired entries from a session dict in-place."""
    expired = [k for k, v in session_entries.items() if now_ms >= v["expires_at_ms"]]
    for k in expired:
        del session_entries[k]


def _evict_oldest_if_needed(session_entries: Dict[str, Dict[str, Any]]) -> None:
    """Evict oldest entries when session exceeds max size."""
    limit = _max_per_session()
    while len(session_entries) > limit:
        oldest_key = min(
            session_entries, key=lambda k: session_entries[k]["last_seen_ms"]
        )
        del session_entries[oldest_key]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def update(
    session_id: str,
    *,
    track_id: Any = None,
    detection_id: Any = None,
    entity_id: Any = None,
    semantic_label: str,
    confidence: float,
    source: str = "gemini",
) -> bool:
    """Store or refresh a semantic label for a tracked entity.

    Returns True if the label was stored, False if rejected or disabled.
    """
    if not _cache_enabled():
        return False
    if confidence < _min_confidence():
        return False

    cleaned = sanitize_label(semantic_label)
    if cleaned is None:
        log.debug(
            "semantic_memory: label rejected by sanitizer session=%s label=%r",
            session_id, semantic_label,
        )
        return False

    key = _entity_key_from_ids(track_id, detection_id, entity_id)
    if key is None:
        return False

    now = _now_ms()
    ttl = _ttl_ms()
    entry = {
        "semantic_label": cleaned,
        "confidence": confidence,
        "last_seen_ms": now,
        "expires_at_ms": now + ttl,
        "source": source,
    }

    with _LOCK:
        if session_id not in _CACHE:
            _CACHE[session_id] = {}
        session_entries = _CACHE[session_id]
        # Lazy cleanup of expired entries.
        _expire_session(session_entries, now)
        session_entries[key] = entry
        _evict_oldest_if_needed(session_entries)

    log.debug(
        "semantic_memory: stored session=%s key=%s label=%r conf=%.2f",
        session_id, key, cleaned, confidence,
    )
    return True


def lookup(
    session_id: str,
    *,
    track_id: Any = None,
    detection_id: Any = None,
    entity_id: Any = None,
) -> Optional[Dict[str, Any]]:
    """Return the cached entry for an entity, or None if not found / expired."""
    key = _entity_key_from_ids(track_id, detection_id, entity_id)
    if key is None:
        return None
    now = _now_ms()
    with _LOCK:
        session_entries = _CACHE.get(session_id)
        if not session_entries:
            return None
        entry = session_entries.get(key)
        if entry is None:
            return None
        if now >= entry["expires_at_ms"]:
            del session_entries[key]
            return None
        return dict(entry)


def _compute_iou(a: Dict[str, Any], b: Dict[str, Any]) -> float:
    """Compute IoU between two normalized bboxes {x, y, w, h}."""
    ax0, ay0 = a.get("x", 0.0), a.get("y", 0.0)
    ax1, ay1 = ax0 + a.get("w", 0.0), ay0 + a.get("h", 0.0)
    bx0, by0 = b.get("x", 0.0), b.get("y", 0.0)
    bx1, by1 = bx0 + b.get("w", 0.0), by0 + b.get("h", 0.0)

    ix0 = max(ax0, bx0)
    iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1)
    iy1 = min(ay1, by1)
    inter_w = max(0.0, ix1 - ix0)
    inter_h = max(0.0, iy1 - iy0)
    inter = inter_w * inter_h

    area_a = (ax1 - ax0) * (ay1 - ay0)
    area_b = (bx1 - bx0) * (by1 - by0)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


def _center_distance(a: Dict[str, Any], b: Dict[str, Any]) -> float:
    """Compute normalized center distance between two bboxes."""
    cx_a = a.get("x", 0.0) + a.get("w", 0.0) / 2.0
    cy_a = a.get("y", 0.0) + a.get("h", 0.0) / 2.0
    cx_b = b.get("x", 0.0) + b.get("w", 0.0) / 2.0
    cy_b = b.get("y", 0.0) + b.get("h", 0.0) / 2.0
    return ((cx_a - cx_b) ** 2 + (cy_a - cy_b) ** 2) ** 0.5


def _find_bbox_fallback(
    session_id: str,
    entity_bbox: Dict[str, Any],
    now: int,
) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    """Try bbox fallback matching.

    Returns (entity_key, entry) when exactly one unambiguous match is found,
    otherwise (None, None).

    Rules:
      - Only within BBOX_FALLBACK_MAX_AGE_MS
      - Requires IoU >= BBOX_FALLBACK_IOU_MIN or center dist <= BBOX_FALLBACK_CENTER_MAX
      - Ambiguous (multiple candidates or same-label candidates) -> skip
    """
    max_age = _bbox_fallback_max_age_ms()
    iou_min = _bbox_fallback_iou_min()
    center_max = _bbox_fallback_center_max()

    session_entries = _CACHE.get(session_id) or {}
    candidates = []
    for key, entry in session_entries.items():
        age = now - entry["last_seen_ms"]
        if age > max_age or now >= entry["expires_at_ms"]:
            continue
        stored_bbox = entry.get("bbox")
        if not isinstance(stored_bbox, dict):
            continue
        iou = _compute_iou(entity_bbox, stored_bbox)
        dist = _center_distance(entity_bbox, stored_bbox)
        if iou >= iou_min or dist <= center_max:
            candidates.append((key, entry))

    if len(candidates) != 1:
        return None, None

    # Check for ambiguity: if any other current entity bbox is equally close,
    # skip to avoid mis-labelling. (This is checked by the caller per-entity.)
    return candidates[0]


def apply_to_entities(
    resp_dict: Dict[str, Any],
    *,
    session_id: str,
    tracks: List[Dict[str, Any]],
    apply_enabled: bool,
) -> Dict[str, int]:
    """Apply cached semantic labels to entities in resp_dict.

    If apply_enabled is False (shadow mode):
      - Does NOT mutate resp_dict["entities"] or resp_dict["tracks"].
      - Returns stats dict with counts (applied_count will be 0).

    If apply_enabled is True:
      - Adds semantic_label, display_label, semantic_label_confidence,
        semantic_label_source, semantic_label_age_ms to matching entities.
      - Never overwrites the original YOLO label.
      - Never modifies risk_level, risk_score, hazard_type, or should_alert.

    Returns stats dict with log fields.
    """
    stats = {
        "semantic_label_cache_size": 0,
        "semantic_label_applied_count": 0,
        "semantic_label_track_match_count": 0,
        "semantic_label_detection_match_count": 0,
        "semantic_label_bbox_fallback_count": 0,
        "semantic_label_ambiguous_skip_count": 0,
    }

    if not _cache_enabled():
        return stats

    now = _now_ms()
    with _LOCK:
        session_entries = _CACHE.get(session_id) or {}
        _expire_session(session_entries, now)
        stats["semantic_label_cache_size"] = len(session_entries)
        if not session_entries:
            return stats

        entities = resp_dict.get("entities") or []
        if not entities:
            return stats

        # Build set of entity keys that have been matched to detect ambiguity.
        matched_cache_keys: set = set()
        # First pass: collect exact track/det/entity matches.
        exact_matches: List[Tuple[int, str, Dict[str, Any]]] = []
        # bbox-fallback candidates (index, entity_bbox, cache_key, entry).
        bbox_candidates: List[Tuple[int, Dict[str, Any]]] = []

        for idx, entity in enumerate(entities):
            if not isinstance(entity, dict):
                continue
            # Priority 1: track_id exact match.
            tid = entity.get("track_id")
            if tid is not None:
                key = f"track_{tid}"
                entry = session_entries.get(key)
                if entry and now < entry["expires_at_ms"]:
                    exact_matches.append((idx, "track", entry))
                    matched_cache_keys.add(key)
                    stats["semantic_label_track_match_count"] += 1
                    continue
            # Priority 2: detection_id / entity_id exact match.
            did = entity.get("detection_id")
            if did is not None:
                key = f"det_{did}"
                entry = session_entries.get(key)
                if entry and now < entry["expires_at_ms"]:
                    exact_matches.append((idx, "det", entry))
                    matched_cache_keys.add(key)
                    stats["semantic_label_detection_match_count"] += 1
                    continue
            eid = entity.get("entity_id") or entity.get("id")
            if eid is not None:
                key = f"ent_{eid}"
                entry = session_entries.get(key)
                if entry and now < entry["expires_at_ms"]:
                    exact_matches.append((idx, "ent", entry))
                    matched_cache_keys.add(key)
                    stats["semantic_label_detection_match_count"] += 1
                    continue
            # No exact match: try bbox fallback.
            bbox = entity.get("bbox")
            if isinstance(bbox, dict):
                bbox_candidates.append((idx, bbox))

        # Bbox fallback pass: each candidate must match exactly one cache entry.
        bbox_matches: List[Tuple[int, Dict[str, Any]]] = []
        # Track which cache entries are already used to detect same-label ambiguity.
        used_cache_keys: set = set(matched_cache_keys)
        for idx, bbox in bbox_candidates:
            matched_key, matched_entry = _find_bbox_fallback(session_id, bbox, now)
            if matched_key is None or matched_entry is None:
                continue
            # Skip if the same cache entry could apply to multiple entities.
            if matched_key in used_cache_keys:
                stats["semantic_label_ambiguous_skip_count"] += 1
                continue
            # Check for same-label ambiguity: are there other not-yet-matched
            # bbox_candidates that would also match the same cache entry?
            ambiguous = False
            for other_idx, other_bbox in bbox_candidates:
                if other_idx == idx:
                    continue
                other_key, _ = _find_bbox_fallback(session_id, other_bbox, now)
                if other_key == matched_key:
                    ambiguous = True
                    break
            if ambiguous:
                stats["semantic_label_ambiguous_skip_count"] += 1
                continue
            bbox_matches.append((idx, matched_entry))
            used_cache_keys.add(matched_key)
            stats["semantic_label_bbox_fallback_count"] += 1

        # Apply all matches.
        if apply_enabled:
            for idx, _match_type, entry in exact_matches:
                _apply_label_to_entity(entities[idx], entry, now)
                stats["semantic_label_applied_count"] += 1
            for idx, entry in bbox_matches:
                _apply_label_to_entity(entities[idx], entry, now)
                stats["semantic_label_applied_count"] += 1
        else:
            # Shadow mode: count but do not mutate.
            stats["semantic_label_applied_count"] = 0

    return stats


def _apply_label_to_entity(entity: Dict[str, Any], entry: Dict[str, Any], now: int) -> None:
    """Write semantic label fields to an entity dict in-place.

    Never overwrites: label, risk_level, risk_score, hazard_type, should_alert.
    """
    label = entry.get("semantic_label")
    if not label:
        return
    # Do not overwrite existing YOLO label.
    entity["semantic_label"] = label
    entity.setdefault("display_label", label)
    entity["semantic_label_confidence"] = entry.get("confidence", 0.0)
    entity["semantic_label_source"] = "gemini_cache"
    entity["semantic_label_age_ms"] = now - entry.get("last_seen_ms", now)


def cache_size(session_id: str) -> int:
    """Return the current number of live entries for a session."""
    now = _now_ms()
    with _LOCK:
        session_entries = _CACHE.get(session_id) or {}
        _expire_session(session_entries, now)
        return len(session_entries)


def expire_session(session_id: str) -> None:
    """Remove all cache entries for a session."""
    with _LOCK:
        _CACHE.pop(session_id, None)


def reset() -> None:
    """Clear all cache (tests / shutdown)."""
    with _LOCK:
        _CACHE.clear()
