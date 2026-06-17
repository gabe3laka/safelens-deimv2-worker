"""
risk/tracking.py -- per-session IoU/centroid multi-object tracker.

Correctness rule (B1): tracker memory is keyed by `session_id`, NEVER global --
two camera streams must not cross-contaminate track_ids. State lives in an
in-memory dict with TTL eviction + a bounded active-session count, reusing the
exact pattern Build Mode already uses (`_cleanup_expired` / `_evict_oldest`
sweep-on-access), so we do not invent a new state mechanism.

Designed as a simple, deterministic IoU+centroid greedy tracker that a stronger
ByteTrack/BoT-SORT backend can replace later behind the same `update()` API.
Pure Python (no torch/numpy) so it runs on the CPU live path and in tests.
"""

from __future__ import annotations

import os
import threading
import time
from typing import Any, Dict, List, Optional

from . import scene_graph

_LOCK = threading.RLock()
# session_id -> {tracks: {tid: track}, next_id, created_at_ms, updated_at_ms}
_SESSIONS: Dict[str, Dict[str, Any]] = {}


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _float_env(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def session_ttl_ms() -> int:
    return _int_env("SESSION_TTL_MS", 30000)


def session_max_active() -> int:
    return _int_env("SESSION_MAX_ACTIVE", 64)


def _iou_match() -> float:
    return _float_env("TRACK_IOU_MATCH", 0.3)


def _max_age_frames() -> int:
    return _int_env("TRACK_MAX_AGE_FRAMES", 30)


def _max_tracks_per_session() -> int:
    return _int_env("TRACK_MAX_PER_SESSION", 300)


def _now_ms() -> int:
    return int(time.time() * 1000)


def _cleanup_expired(now_ms: int) -> None:
    """Drop sessions idle longer than SESSION_TTL_MS (Build Mode sweep pattern)."""
    ttl = session_ttl_ms()
    for sid in [s for s, v in list(_SESSIONS.items())
                if now_ms - v.get("updated_at_ms", now_ms) > ttl]:
        _SESSIONS.pop(sid, None)


def _evict_oldest() -> None:
    if _SESSIONS:
        oldest = min(_SESSIONS.items(), key=lambda kv: kv[1].get("updated_at_ms", 0))[0]
        _SESSIONS.pop(oldest, None)


def _get_or_create(session_id: str, now_ms: int) -> Dict[str, Any]:
    sess = _SESSIONS.get(session_id)
    if sess is None:
        if len(_SESSIONS) >= session_max_active():
            _evict_oldest()
        sess = {"tracks": {}, "next_id": 1,
                "created_at_ms": now_ms, "updated_at_ms": now_ms}
        _SESSIONS[session_id] = sess
    return sess


def _match(entities: List[Dict[str, Any]], tracks: Dict[str, Dict[str, Any]],
           iou_thresh: float) -> Dict[int, str]:
    """Greedy IoU match (same class preferred). Returns {entity_index: track_id}."""
    candidates = []  # (iou, ent_idx, track_id)
    for tid, trk in tracks.items():
        for ei, e in enumerate(entities):
            same_class = int(e.get("class_id", -1)) == int(trk.get("class_id", -2))
            ov = scene_graph.iou(e.get("bbox") or {}, trk.get("bbox") or {})
            if ov >= iou_thresh:
                # bias toward same-class matches without excluding cross-class
                candidates.append((ov + (0.001 if same_class else 0.0), ei, tid))
    candidates.sort(reverse=True)
    assigned_e: Dict[int, str] = {}
    used_t: set = set()
    for _score, ei, tid in candidates:
        if ei in assigned_e or tid in used_t:
            continue
        assigned_e[ei] = tid
        used_t.add(tid)
    return assigned_e


def update(session_id: Optional[str], entities: List[Dict[str, Any]],
           ts_ms: Optional[int] = None) -> List[Dict[str, Any]]:
    """Update the tracker for one frame; return the active tracks as dicts.

    Never raises. A missing session_id is bucketed under '__default__' (the
    /detect HSE loop is stateless across calls unless the app sends a session).
    """
    sid = session_id or "__default__"
    now = ts_ms if ts_ms is not None else _now_ms()
    iou_thresh = _iou_match()
    max_age = _max_age_frames()
    cap = _max_tracks_per_session()

    with _LOCK:
        _cleanup_expired(now)
        sess = _get_or_create(sid, now)
        tracks: Dict[str, Dict[str, Any]] = sess["tracks"]

        assigned = _match(entities or [], tracks, iou_thresh)

        matched_tids = set()
        for ei, e in enumerate(entities or []):
            bb = e.get("bbox") or {}
            cen = scene_graph.centroid(bb)
            tid = assigned.get(ei)
            if tid is not None:
                trk = tracks[tid]
                prev_cen = trk.get("centroid", cen)
                dt = max(1e-3, (now - trk.get("last_seen_ms", now)) / 1000.0)
                trk["velocity"] = {"vx": round((cen["x"] - prev_cen["x"]) / dt, 4),
                                   "vy": round((cen["y"] - prev_cen["y"]) / dt, 4)}
                trk.update(label=str(e.get("label", trk.get("label", ""))),
                           class_id=int(e.get("class_id", trk.get("class_id", -1))),
                           confidence=float(e.get("confidence", 0.0)),
                           bbox=dict(bb), centroid=cen, last_seen_ms=now,
                           missed=0)
                trk["hits"] += 1
                trk["age_frames"] += 1
                matched_tids.add(tid)
            else:
                if len(tracks) >= cap:
                    continue  # bounded: drop new tracks past the per-session cap
                tid = f"trk_{sess['next_id']}"
                sess["next_id"] += 1
                tracks[tid] = {
                    "track_id": tid, "label": str(e.get("label", "")),
                    "class_id": int(e.get("class_id", -1)),
                    "confidence": float(e.get("confidence", 0.0)),
                    "bbox": dict(bb), "centroid": cen,
                    "velocity": {"vx": 0.0, "vy": 0.0},
                    "age_frames": 1, "hits": 1, "missed": 0,
                    "first_seen_ms": now, "last_seen_ms": now,
                }
                matched_tids.add(tid)

        # Age unmatched tracks; drop the stale ones.
        for tid in list(tracks.keys()):
            if tid not in matched_tids:
                tracks[tid]["missed"] += 1
                tracks[tid]["age_frames"] += 1
                if tracks[tid]["missed"] > max_age:
                    tracks.pop(tid, None)

        sess["updated_at_ms"] = now
        # Return active (currently-visible this frame) tracks, stable-ordered.
        return [dict(tracks[tid]) for tid in sorted(matched_tids) if tid in tracks]


# -- introspection / test helpers ---------------------------------------------

def active_session_count() -> int:
    with _LOCK:
        return len(_SESSIONS)


def session_track_count(session_id: str) -> int:
    with _LOCK:
        sess = _SESSIONS.get(session_id)
        return len(sess["tracks"]) if sess else 0


def get_track_ids(session_id: str) -> List[str]:
    with _LOCK:
        sess = _SESSIONS.get(session_id)
        return sorted(sess["tracks"].keys()) if sess else []


def reset(session_id: Optional[str] = None) -> None:
    """Clear one session (or all). Used by tests and on shutdown."""
    with _LOCK:
        if session_id is None:
            _SESSIONS.clear()
        else:
            _SESSIONS.pop(session_id, None)


def sweep(now_ms: Optional[int] = None) -> int:
    """Force a TTL sweep; return the number of sessions remaining."""
    with _LOCK:
        _cleanup_expired(now_ms if now_ms is not None else _now_ms())
        return len(_SESSIONS)
