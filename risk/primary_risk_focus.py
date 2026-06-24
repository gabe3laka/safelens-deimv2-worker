"""
risk/primary_risk_focus.py -- Short-lived per-session primary risk focus cache.

Safety rules (hard):
  * Stores ONLY small metadata (track_id, detection_id, hazard_type, risk_level,
    confidence, timestamps). Never stores images, crops, base64 frames, or full
    Gemini responses.
  * Primary risk focus is keyed by session_id + hazard_type + track_id/detection_id.
  * Never keyed by semantic label text.
  * Semantic label cache and primary risk focus remain SEPARATE:
      - semantic_memory: session_id + track_id -> semantic_label (descriptive only)
      - primary_risk_focus: session_id + hazard_type + track_id -> risk attribution
  * Primary risk focus NEVER creates risk_level; it only helps attribute existing risks.
  * Expires after PRIMARY_RISK_FOCUS_TTL_MS (default 3500 ms).
    This is intentionally shorter than semantic label cache because risk focus
    directly affects which boxes stay yellow.

Rollout env flags (all have safe defaults):
  PRIMARY_RISK_FOCUS_ENABLED                          true
  PRIMARY_RISK_FOCUS_APPLY_ENABLED                    false   <- shadow mode by default
  PRIMARY_RISK_FOCUS_TTL_MS                           3500
  PRIMARY_RISK_FOCUS_MIN_CONFIDENCE                   0.65
  PRIMARY_RISK_FOCUS_REQUIRE_CURRENT_DETERMINISTIC_SUPPORT   true
  PRIMARY_RISK_CONTEXT_SUPPRESS_ENABLED               true
  PRIMARY_RISK_CONTEXT_SUPPRESS_APPLY_ENABLED         false   <- shadow mode by default

Hazard types where primary focus filtering applies (conservative list):
  object_near_edge, falling_object, blocked_path, broken_object

Hazard types explicitly excluded (never suppress):
  ppe_missing, worker_near_vehicle, unsafe_interaction, fire, smoke
"""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Any, Dict, List, Optional

log = logging.getLogger("safelens-vision-worker.primary_risk_focus")

# ---------------------------------------------------------------------------
# Hazard types where primary focus filtering is enabled (conservative)
# ---------------------------------------------------------------------------

_FOCUS_HAZARD_TYPES = frozenset({
    "object_near_edge",
    "falling_object",
    "blocked_path",
    "broken_object",
})

# Hazard types explicitly excluded from primary focus suppression.
_EXCLUDED_HAZARD_TYPES = frozenset({
    "ppe_missing",
    "worker_near_vehicle",
    "unsafe_interaction",
    "fire",
    "smoke",
})

# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def _focus_enabled() -> bool:
    return os.getenv("PRIMARY_RISK_FOCUS_ENABLED", "true").strip().lower() in (
        "1", "true", "yes", "on"
    )


def _apply_enabled() -> bool:
    return os.getenv("PRIMARY_RISK_FOCUS_APPLY_ENABLED", "false").strip().lower() in (
        "1", "true", "yes", "on"
    )


def _ttl_ms() -> int:
    try:
        return max(500, int(os.getenv("PRIMARY_RISK_FOCUS_TTL_MS", "3500")))
    except (TypeError, ValueError):
        return 3500


def _min_confidence() -> float:
    try:
        return float(os.getenv("PRIMARY_RISK_FOCUS_MIN_CONFIDENCE", "0.65"))
    except (TypeError, ValueError):
        return 0.65


def _require_deterministic_support() -> bool:
    return os.getenv(
        "PRIMARY_RISK_FOCUS_REQUIRE_CURRENT_DETERMINISTIC_SUPPORT", "true"
    ).strip().lower() in ("1", "true", "yes", "on")


def _suppress_enabled() -> bool:
    return os.getenv("PRIMARY_RISK_CONTEXT_SUPPRESS_ENABLED", "true").strip().lower() in (
        "1", "true", "yes", "on"
    )


def _suppress_apply_enabled() -> bool:
    return os.getenv("PRIMARY_RISK_CONTEXT_SUPPRESS_APPLY_ENABLED", "false").strip().lower() in (
        "1", "true", "yes", "on"
    )


def shadow_mode() -> bool:
    """Return True when operating in shadow mode (no actual suppression)."""
    return not _apply_enabled() or not _suppress_apply_enabled()


# ---------------------------------------------------------------------------
# Cache storage
# ---------------------------------------------------------------------------
# Structure:
#   _CACHE[session_id][risk_key] = {
#       "primary_track_id": str | None,
#       "primary_detection_id": str | None,
#       "hazard_type": str,
#       "risk_level": str,
#       "confidence": float,
#       "last_seen_ms": int,
#       "expires_at_ms": int,
#       "source": str,
#   }

_LOCK = threading.RLock()
_CACHE: Dict[str, Dict[str, Dict[str, Any]]] = {}


def _now_ms() -> int:
    return int(time.time() * 1000)


def _risk_key(hazard_type: str, track_id: Any, detection_id: Any) -> Optional[str]:
    """Build a risk cache key from hazard_type + track_id or detection_id.

    Never keys by semantic label.
    """
    if not hazard_type:
        return None
    if track_id is not None:
        return f"{hazard_type}:track_{track_id}"
    if detection_id is not None:
        return f"{hazard_type}:det_{detection_id}"
    return None


def _expire_session(session_entries: Dict[str, Dict[str, Any]], now_ms: int) -> None:
    """Remove expired entries from a session dict in-place."""
    expired = [k for k, v in session_entries.items() if now_ms >= v["expires_at_ms"]]
    for k in expired:
        del session_entries[k]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def update(
    session_id: str,
    *,
    hazard_type: str,
    risk_level: str,
    confidence: float,
    track_id: Any = None,
    detection_id: Any = None,
    source: str = "gemini",
) -> bool:
    """Store or refresh a primary risk focus entry.

    Returns True if the entry was stored, False if rejected or disabled.

    Rules:
    - focus_enabled must be True
    - hazard_type must be in the conservative allowed set
    - confidence must meet threshold
    - risk_level must be YELLOW/ORANGE/RED
    - must have track_id or detection_id
    - never keys by semantic label
    """
    if not _focus_enabled():
        return False
    if hazard_type not in _FOCUS_HAZARD_TYPES:
        return False
    if risk_level not in ("YELLOW", "ORANGE", "RED"):
        return False
    if confidence < _min_confidence():
        return False

    key = _risk_key(hazard_type, track_id, detection_id)
    if key is None:
        return False

    now = _now_ms()
    ttl = _ttl_ms()
    entry: Dict[str, Any] = {
        "primary_track_id": str(track_id) if track_id is not None else None,
        "primary_detection_id": str(detection_id) if detection_id is not None else None,
        "hazard_type": hazard_type,
        "risk_level": risk_level,
        "confidence": confidence,
        "last_seen_ms": now,
        "expires_at_ms": now + ttl,
        "source": source,
    }

    with _LOCK:
        if session_id not in _CACHE:
            _CACHE[session_id] = {}
        session_entries = _CACHE[session_id]
        _expire_session(session_entries, now)
        session_entries[key] = entry

    log.debug(
        "primary_risk_focus: stored session=%s key=%s level=%s conf=%.2f",
        session_id, key, risk_level, confidence,
    )
    return True


def lookup(
    session_id: str,
    *,
    hazard_type: str,
    track_id: Any = None,
    detection_id: Any = None,
) -> Optional[Dict[str, Any]]:
    """Return the cached primary focus entry, or None if not found / expired."""
    key = _risk_key(hazard_type, track_id, detection_id)
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


def get_all_valid(session_id: str) -> Dict[str, Dict[str, Any]]:
    """Return all non-expired primary focus entries for a session."""
    now = _now_ms()
    with _LOCK:
        session_entries = _CACHE.get(session_id)
        if not session_entries:
            return {}
        _expire_session(session_entries, now)
        return {k: dict(v) for k, v in session_entries.items()}


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


# ---------------------------------------------------------------------------
# Current-frame sync gate (Part 5)
# ---------------------------------------------------------------------------


def validate_focus_entry(
    entry: Dict[str, Any],
    *,
    current_entities: List[Dict[str, Any]],
    current_det_risks: List[Dict[str, Any]],
    session_id: str,
) -> bool:
    """Return True only when a cached primary focus is still valid against the current frame.

    Checks:
    1. Entry is not expired (caller should have already checked, but we re-verify).
    2. primary track/detection is visible in current entities (has a current bbox).
    3. If PRIMARY_RISK_FOCUS_REQUIRE_CURRENT_DETERMINISTIC_SUPPORT=true:
       - current deterministic risks must include the same hazard_type linked to the
         same track/detection.

    If the current frame no longer supports the risk, this returns False.
    The caller should then expire or ignore the entry.
    """
    now = _now_ms()
    if now >= entry.get("expires_at_ms", 0):
        return False

    hazard_type = entry.get("hazard_type", "")
    primary_track_id = entry.get("primary_track_id")
    primary_detection_id = entry.get("primary_detection_id")

    # 1. Check that the primary entity is still visible in the current frame.
    entity_visible = False
    for e in (current_entities or []):
        if not isinstance(e, dict):
            continue
        if primary_track_id is not None and str(e.get("track_id", "")) == primary_track_id:
            if isinstance(e.get("bbox"), dict):
                entity_visible = True
                break
        if primary_detection_id is not None and str(e.get("detection_id", "")) == primary_detection_id:
            if isinstance(e.get("bbox"), dict):
                entity_visible = True
                break

    if not entity_visible:
        log.debug(
            "primary_risk_focus: validate: entity not visible session=%s track=%s det=%s",
            session_id, primary_track_id, primary_detection_id,
        )
        return False

    # 2. If REQUIRE_CURRENT_DETERMINISTIC_SUPPORT, check current det risks.
    if _require_deterministic_support():
        det_support = False
        for r in (current_det_risks or []):
            if not isinstance(r, dict):
                continue
            if r.get("hazard_type") != hazard_type:
                continue
            # Check if this risk is linked to the same track/detection.
            if primary_track_id is not None:
                tids = [str(t) for t in (r.get("involved_track_ids") or [])]
                if primary_track_id in tids:
                    det_support = True
                    break
                for fld in ("linked_entity_id", "entity_id", "track_id"):
                    if r.get(fld) and str(r[fld]) == primary_track_id:
                        det_support = True
                        break
                if det_support:
                    break
            if primary_detection_id is not None:
                dids = [str(d) for d in (r.get("involved_detection_ids") or [])]
                if primary_detection_id in dids:
                    det_support = True
                    break
        if not det_support:
            log.debug(
                "primary_risk_focus: validate: no current det support session=%s hz=%s",
                session_id, hazard_type,
            )
            return False

    return True


# ---------------------------------------------------------------------------
# Primary/context filtering (Part 6)
# ---------------------------------------------------------------------------


def apply_focus_filter(
    resp_dict: Dict[str, Any],
    *,
    session_id: str,
    current_det_risks: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, int]:
    """Apply primary risk focus filtering to scene_risks.

    Must be called BEFORE _stamp_entity_risks().

    Shadow mode (PRIMARY_RISK_FOCUS_APPLY_ENABLED=false or
    PRIMARY_RISK_CONTEXT_SUPPRESS_APPLY_ENABLED=false):
    - Does NOT change scene_risks or entity stamping.
    - Logs what would be suppressed.

    Apply mode:
    - If a valid primary risk focus exists for a hazard_type:
      - keep scene_risks linked to the primary track/detection.
      - mark nearby same-hazard context/support risks as candidate_only=True +
        suppressed_by_primary_focus=True.
      - does NOT suppress any context object with its own independent RED/ORANGE risk.
      - does NOT suppress PPE/person/vehicle/fire/smoke risks.

    Returns stats dict with log fields.
    """
    stats = {
        "primary_risk_focus_cache_size": 0,
        "primary_risk_focus_update_count": 0,
        "primary_risk_focus_valid_count": 0,
        "primary_risk_focus_expired_count": 0,
        "primary_risk_focus_current_support_miss_count": 0,
        "primary_risk_context_suppressed_count": 0,
        "primary_risk_context_would_suppress_count": 0,
        "primary_risk_focus_shadow_mode": shadow_mode(),
        "reasoner_linked_risk_ttl_ms": _get_reasoner_linked_risk_ttl_ms(),
    }

    if not _focus_enabled() or not _suppress_enabled():
        return stats

    current_entities = resp_dict.get("entities") or []
    scene_risks = resp_dict.get("scene_risks") or []
    det_risks = current_det_risks or resp_dict.get("risks") or []

    now = _now_ms()
    with _LOCK:
        session_entries = _CACHE.get(session_id) or {}
        _expire_session(session_entries, now)
        stats["primary_risk_focus_cache_size"] = len(session_entries)
        if not session_entries:
            return stats

        # Validate each cached focus entry against the current frame.
        valid_focus: Dict[str, Dict[str, Any]] = {}
        for key, entry in list(session_entries.items()):
            is_valid = validate_focus_entry(
                entry,
                current_entities=current_entities,
                current_det_risks=det_risks,
                session_id=session_id,
            )
            if is_valid:
                valid_focus[key] = dict(entry)
                stats["primary_risk_focus_valid_count"] += 1
            else:
                stats["primary_risk_focus_current_support_miss_count"] += 1

    if not valid_focus:
        return stats

    # Build a set of primary track IDs per hazard_type from valid focus entries.
    # Structure: { hazard_type: set of primary_track_ids }
    primary_by_hazard: Dict[str, set] = {}
    for entry in valid_focus.values():
        hz = entry.get("hazard_type", "")
        if hz not in _FOCUS_HAZARD_TYPES:
            continue
        primary_tid = entry.get("primary_track_id")
        primary_did = entry.get("primary_detection_id")
        if hz not in primary_by_hazard:
            primary_by_hazard[hz] = set()
        if primary_tid is not None:
            primary_by_hazard[hz].add(str(primary_tid))
        if primary_did is not None:
            primary_by_hazard[hz].add(f"det_{primary_did}")

    if not primary_by_hazard:
        return stats

    apply = _apply_enabled() and _suppress_apply_enabled()

    # Examine scene_risks for context/support risks to suppress.
    for risk in scene_risks:
        if not isinstance(risk, dict):
            continue
        hz = risk.get("hazard_type", "")
        if hz not in _FOCUS_HAZARD_TYPES:
            continue
        if hz in _EXCLUDED_HAZARD_TYPES:
            continue
        if hz not in primary_by_hazard:
            continue

        rl = risk.get("risk_level", "GREEN")
        # Never suppress RED or ORANGE independent risks.
        if rl in ("RED", "ORANGE"):
            continue

        # Check if this risk is linked to a primary focus entity.
        primary_ids = primary_by_hazard[hz]
        risk_tids = set(str(t) for t in (risk.get("involved_track_ids") or []))
        for fld in ("linked_entity_id", "entity_id", "track_id"):
            if risk.get(fld):
                risk_tids.add(str(risk[fld]))

        if risk_tids & primary_ids:
            # This risk IS the primary object — do not suppress.
            continue

        # This risk is a context/support object for the same hazard.
        # It is a candidate for suppression.
        if apply:
            risk["candidate_only"] = True
            risk["suppressed_by_primary_focus"] = True
            # Record which primary focus track suppressed it.
            primary_tid_list = sorted(primary_ids)
            risk["primary_focus_track_id"] = primary_tid_list[0] if primary_tid_list else None
            stats["primary_risk_context_suppressed_count"] += 1
        else:
            # Shadow mode: log only.
            stats["primary_risk_context_would_suppress_count"] += 1
            log.debug(
                "primary_risk_focus: shadow: would suppress risk_id=%s hz=%s level=%s "
                "primary_ids=%s",
                risk.get("risk_id"), hz, rl, sorted(primary_ids),
            )

    return stats


def _get_reasoner_linked_risk_ttl_ms() -> int:
    """Read REASONER_LINKED_RISK_TTL_MS from env (recommended to lower stale VLM carryover)."""
    try:
        return int(os.getenv("REASONER_LINKED_RISK_TTL_MS", "8000"))
    except (TypeError, ValueError):
        return 8000
