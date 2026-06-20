"""
temporal_reasoning/session_memory.py -- bounded per-session temporal sub-record.

This is a SUB-RECORD keyed by the SAME session_id the risk tracker uses (it does
not invent a competing session lifecycle): same TTL/eviction pattern
(SESSION_TTL_MS / TEMPORAL_MAX_ACTIVE_SESSIONS, Build Mode sweep-on-access). It
holds short temporal context per session:

    last_seen_ms, frame_count, last_frame_id, recent frame metadata,
    per-track history / label history / confidence history,
    latest scene_context, latest semantic_corrections, latest VLM result,
    pending reasoner status, last_reasoner_trigger_ms.

Raw frames are NEVER stored (TEMPORAL_STORE_KEYFRAMES=false by default). History
is pure metadata (bbox/centroid/label/confidence + timestamps).
"""

from __future__ import annotations

import os
import threading
import time
from collections import deque
from typing import Any, Dict, List, Optional


# -- env helpers (shared across the temporal package) -------------------------

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


def _bool_env(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def enabled() -> bool:
    return _bool_env("TEMPORAL_REASONING_ENABLED", False)


def memory_window_frames() -> int:
    return max(2, _int_env("TEMPORAL_MEMORY_WINDOW_FRAMES", 45))


def memory_ttl_ms() -> int:
    # Aligns with the risk tracker's SESSION_TTL_MS when TEMPORAL_MEMORY_TTL_MS unset.
    return _int_env("TEMPORAL_MEMORY_TTL_MS", _int_env("SESSION_TTL_MS", 30000))


def max_active_sessions() -> int:
    return max(1, _int_env("TEMPORAL_MAX_ACTIVE_SESSIONS", 64))


def store_keyframes() -> bool:
    return _bool_env("TEMPORAL_STORE_KEYFRAMES", False)


def label_flip_window_frames() -> int:
    return max(2, _int_env("TEMPORAL_LABEL_FLIP_WINDOW_FRAMES", 8))


def _now_ms() -> int:
    return int(time.time() * 1000)


# -- store --------------------------------------------------------------------

_LOCK = threading.RLock()
_SESSIONS: Dict[str, Dict[str, Any]] = {}
_DEFAULT_SID = "__default__"


def _blank(sid: str) -> Dict[str, Any]:
    return {
        "session_id": sid,
        "camera_id": None,
        "created_at_ms": _now_ms(),
        "last_seen_ms": _now_ms(),
        "frame_count": 0,
        "last_frame_id": None,
        "recent_frames": deque(maxlen=memory_window_frames()),
        "track_history": {},        # tid -> deque[{bbox,centroid,ts}]
        "label_history": {},        # tid -> deque[label]
        "confidence_history": {},   # tid -> deque[float]
        "latest_scene_context": {},
        "latest_semantic_corrections": [],
        "latest_vlm_result": {},
        "pending_reasoner_state": "idle",
        "last_reasoner_trigger_ms": 0,
        "last_reasoner_result_ms": 0,
        "last_reasoner_state_ms": 0,
    }


def _cleanup_expired(now_ms: int) -> None:
    ttl = memory_ttl_ms()
    for sid in [s for s, v in list(_SESSIONS.items())
                if now_ms - v.get("last_seen_ms", now_ms) > ttl]:
        _SESSIONS.pop(sid, None)


def _evict_oldest() -> None:
    if _SESSIONS:
        oldest = min(_SESSIONS.items(), key=lambda kv: kv[1].get("last_seen_ms", 0))[0]
        _SESSIONS.pop(oldest, None)


def _get(sid: Optional[str]) -> Dict[str, Any]:
    key = sid or _DEFAULT_SID
    now = _now_ms()
    _cleanup_expired(now)
    rec = _SESSIONS.get(key)
    if rec is None:
        while len(_SESSIONS) >= max_active_sessions():
            _evict_oldest()
        rec = _blank(key)
        _SESSIONS[key] = rec
    return rec


def _centroid(bbox: Dict[str, float]) -> Dict[str, float]:
    try:
        return {"x": float(bbox.get("x", 0.0)) + float(bbox.get("w", 0.0)) / 2.0,
                "y": float(bbox.get("y", 0.0)) + float(bbox.get("h", 0.0)) / 2.0}
    except Exception:  # noqa: BLE001
        return {"x": 0.0, "y": 0.0}


def update(session_id: Optional[str], *, frame_id: Optional[str],
           entities: List[Dict[str, Any]], tracks: List[Dict[str, Any]],
           camera_id: Optional[str] = None) -> Dict[str, Any]:
    """Fold one frame's tracks/entities into the session sub-record.

    Returns the (locked-copy-safe) record reference. Never raises.
    """
    win = memory_window_frames()
    now = _now_ms()
    with _LOCK:
        rec = _get(session_id)
        if camera_id:
            rec["camera_id"] = camera_id
        rec["last_seen_ms"] = now
        rec["frame_count"] += 1
        rec["last_frame_id"] = frame_id
        rec["recent_frames"].append({"frame_id": frame_id, "ts": now,
                                     "entities": len(entities or [])})
        # Prefer tracks (stable ids); fall back to entity index ids.
        rows = tracks or []
        if not rows and entities:
            rows = [{"track_id": f"e{i}", "label": e.get("label"),
                     "confidence": e.get("confidence", 0.0), "bbox": e.get("bbox", {})}
                    for i, e in enumerate(entities)]
        seen = set()
        for t in rows:
            tid = str(t.get("track_id") or t.get("id") or "")
            if not tid:
                continue
            seen.add(tid)
            bbox = t.get("bbox", {}) or {}
            th = rec["track_history"].setdefault(tid, deque(maxlen=win))
            th.append({"bbox": dict(bbox), "centroid": _centroid(bbox), "ts": now})
            lh = rec["label_history"].setdefault(tid, deque(maxlen=label_flip_window_frames()))
            lh.append(str(t.get("label")))
            ch = rec["confidence_history"].setdefault(tid, deque(maxlen=win))
            ch.append(float(t.get("confidence", 0.0) or 0.0))
        # Drop history for tracks not seen for a while (bounded memory).
        for tid in list(rec["track_history"].keys()):
            if tid not in seen and len(rec["track_history"][tid]) == 0:
                rec["track_history"].pop(tid, None)
                rec["label_history"].pop(tid, None)
                rec["confidence_history"].pop(tid, None)
        return rec


def active_track_count(session_id: Optional[str]) -> int:
    with _LOCK:
        rec = _SESSIONS.get(session_id or _DEFAULT_SID)
        return len(rec["track_history"]) if rec else 0


def memory_frames(session_id: Optional[str]) -> int:
    with _LOCK:
        rec = _SESSIONS.get(session_id or _DEFAULT_SID)
        return len(rec["recent_frames"]) if rec else 0


def set_reasoner_state(session_id: Optional[str], state: str,
                       *, trigger: Optional[str] = None) -> None:
    with _LOCK:
        rec = _get(session_id)
        rec["pending_reasoner_state"] = state
        rec["last_reasoner_state_ms"] = _now_ms()
        if trigger is not None:
            rec["last_reasoner_trigger_ms"] = _now_ms()
            rec["last_trigger"] = trigger


def store_vlm_result(session_id: Optional[str], *, scene_context: Dict[str, Any],
                     semantic_corrections: List[Dict[str, Any]],
                     vlm_result: Dict[str, Any]) -> None:
    with _LOCK:
        rec = _get(session_id)
        if scene_context:
            rec["latest_scene_context"] = scene_context
        rec["latest_semantic_corrections"] = semantic_corrections or []
        rec["latest_vlm_result"] = vlm_result or {}
        rec["last_reasoner_result_ms"] = _now_ms()
        rec["pending_reasoner_state"] = "ready"
        rec["last_reasoner_state_ms"] = _now_ms()


def snapshot(session_id: Optional[str]) -> Dict[str, Any]:
    """Read-only view used to build the /detect temporal blocks. Never raises."""
    with _LOCK:
        rec = _SESSIONS.get(session_id or _DEFAULT_SID)
        if rec is None:
            return {}
        now = _now_ms()
        result_ms = rec.get("last_reasoner_result_ms", 0)
        state_ms = rec.get("last_reasoner_state_ms", 0)
        return {
            "session_id": rec["session_id"],
            "camera_id": rec.get("camera_id"),
            "frame_count": rec["frame_count"],
            "memory_frames": len(rec["recent_frames"]),
            "active_tracks": len(rec["track_history"]),
            "latest_scene_context": dict(rec.get("latest_scene_context") or {}),
            "latest_semantic_corrections": list(rec.get("latest_semantic_corrections") or []),
            "pending_reasoner_state": rec.get("pending_reasoner_state", "idle"),
            "pending_reasoner_state_age_ms": (now - state_ms) if state_ms else None,
            "last_trigger": rec.get("last_trigger"),
            "last_reasoner_result_ms": result_ms,
            "result_age_ms": (now - result_ms) if result_ms else None,
        }


def label_history(session_id: Optional[str], track_id: str) -> List[str]:
    with _LOCK:
        rec = _SESSIONS.get(session_id or _DEFAULT_SID)
        if not rec:
            return []
        return list(rec["label_history"].get(track_id, []))


def track_history(session_id: Optional[str], track_id: str) -> List[Dict[str, Any]]:
    with _LOCK:
        rec = _SESSIONS.get(session_id or _DEFAULT_SID)
        if not rec:
            return []
        return list(rec["track_history"].get(track_id, []))


def confidence_history(session_id: Optional[str], track_id: str) -> List[float]:
    with _LOCK:
        rec = _SESSIONS.get(session_id or _DEFAULT_SID)
        if not rec:
            return []
        return list(rec["confidence_history"].get(track_id, []))


def all_track_ids(session_id: Optional[str]) -> List[str]:
    with _LOCK:
        rec = _SESSIONS.get(session_id or _DEFAULT_SID)
        return list(rec["track_history"].keys()) if rec else []


def active_session_count() -> int:
    with _LOCK:
        return len(_SESSIONS)


def reset() -> None:
    with _LOCK:
        _SESSIONS.clear()
