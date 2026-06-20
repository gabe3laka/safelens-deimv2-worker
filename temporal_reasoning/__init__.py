"""
temporal_reasoning/ -- event-triggered temporal VLM perception layer (GPU side).

Wraps the per-frame detector output with short temporal memory + deterministic
object-near-edge risk, decides when (rarely) to run the VLM, and runs it
NON-BLOCKING on a bounded background executor under a bounded GPU slot. /detect
attaches the most recent cached scene_context / semantic_corrections and returns
immediately -- it never waits on the VLM.

Public API (import-light; no torch at import time):

    from temporal_reasoning import attach_temporal, config, status_snapshot, enabled

`attach_temporal(resp_dict, session_id=..., frame_id=..., frame_b64=..., payload=...)`
is the one-liner /detect calls. It is ADDITIVE and never raises: when
TEMPORAL_REASONING_ENABLED is false the response is byte-for-byte unchanged.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from . import async_reasoning, edge_risk, scene_context, semantic_corrections
from . import session_memory as mem
from . import triggers
from .session_memory import _int_env, enabled

log = logging.getLogger("safelens-vision-worker.temporal")

_LEVEL = {"GREEN": 0, "YELLOW": 1, "ORANGE": 2, "RED": 3}
_LEVEL_NAME = {v: k for k, v in _LEVEL.items()}


def _result_stale_ms() -> int:
    return _int_env("REASONER_RESULT_STALE_MS", 8000)


def _camera_id(payload: Dict[str, Any], session_id: Optional[str]) -> Optional[str]:
    cam = (payload or {}).get("camera_context") or {}
    return cam.get("camera_name") or cam.get("camera_id") or session_id


def _raise_level(resp: Dict[str, Any], level: str) -> None:
    cur = _LEVEL.get(str(resp.get("highest_risk_level", "GREEN")).upper(), 0)
    new = _LEVEL.get(str(level).upper(), 0)
    if new > cur:
        resp["highest_risk_level"] = _LEVEL_NAME[new]


def attach_temporal(resp_dict: Dict[str, Any], *, session_id: Optional[str] = None,
                    frame_id: Optional[str] = None, frame_b64: Optional[str] = None,
                    payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Merge the additive temporal-reasoning blocks into a /detect response.

    No-op (legacy shape) unless TEMPORAL_REASONING_ENABLED. Never raises.
    """
    if not enabled():
        return resp_dict
    payload = payload or {}
    try:
        entities: List[Dict[str, Any]] = resp_dict.get("entities", []) or []
        tracks: List[Dict[str, Any]] = resp_dict.get("tracks", []) or []
        deterministic_risks: List[Dict[str, Any]] = resp_dict.get("risks", []) or []
        highest_level = resp_dict.get("highest_risk_level", "GREEN")

        # 1) fold this frame into the per-session temporal sub-record
        mem.update(session_id, frame_id=frame_id, entities=entities, tracks=tracks,
                   camera_id=_camera_id(payload, session_id))

        # 2) deterministic object-near-edge risk (CPU-only; additive)
        edge_risks = edge_risk.evaluate(session_id, entities=entities, tracks=tracks)
        if edge_risks:
            resp_dict.setdefault("risks", [])
            resp_dict["risks"].extend(edge_risks)
            for er in edge_risks:
                _raise_level(resp_dict, er.get("risk_level", "YELLOW"))
            highest_level = resp_dict.get("highest_risk_level", highest_level)

        # 3) evaluate triggers (the VLM runs only on these)
        reasons = triggers.evaluate(
            session_id, entities=entities, tracks=tracks, highest_level=highest_level,
            deterministic_risks=deterministic_risks, edge_risks=edge_risks, payload=payload)

        # 4) NON-BLOCKING: maybe submit an async reasoning job; never wait.
        # Poll-only/cached-only live frames must not replace _PENDING[sid].
        prefs = payload.get("reasoning_preferences") or {}
        do_not_start_new_reasoning_job = prefs.get("do_not_start_new_reasoning_job") is True
        force_reason = prefs.get("force_reason") is True
        if do_not_start_new_reasoning_job and not force_reason:
            trigger_status = "not_triggered"
        else:
            trigger_status = async_reasoning.maybe_trigger(
                session_id, reasons=reasons, entities=entities, tracks=tracks,
                frame_b64=frame_b64, payload=payload)

        # 5) attach blocks from the most recent cached result (never blocks)
        snap = mem.snapshot(session_id)
        resp_dict["temporal_reasoning"] = {
            "enabled": True,
            "session_id": session_id,
            "memory_frames": snap.get("memory_frames", 0),
            "active_tracks": snap.get("active_tracks", 0),
            "triggered": bool(reasons),
            "trigger_reasons": reasons,
        }
        cached_ctx = snap.get("latest_scene_context") or {}
        if cached_ctx:
            resp_dict["scene_context"] = cached_ctx
        cached_corr = snap.get("latest_semantic_corrections") or []
        if cached_corr:
            # additive; raw detector entities are preserved untouched
            resp_dict["semantic_corrections"] = cached_corr

        result_age = snap.get("result_age_ms")
        state = snap.get("pending_reasoner_state", "idle")
        if trigger_status == "disabled":
            state = "disabled"
        elif trigger_status in ("triggered", "queued", "queued_latest") and state in ("idle", "ready"):
            state = "queued"
        elif trigger_status == "queued_latest":
            state = "queued_latest"
        elif trigger_status == "throttled" and state in ("idle",):
            state = "throttled"
        elif trigger_status == "running":
            state = "running"
        elif trigger_status in ("schema_error", "json_parse_error", "error", "timeout", "unavailable"):
            state = "schema_error" if trigger_status == "json_parse_error" else trigger_status
        resp_dict["reasoner_status"] = {
            "enabled": _vlm_enabled(),
            "mode": _vlm_mode(),
            "state": state,
            "last_trigger": snap.get("last_trigger"),
            "result_age_ms": result_age,
            "stale": bool(result_age is not None and result_age > _result_stale_ms()),
            "run_status": trigger_status,
        }
        return resp_dict
    except Exception as exc:  # noqa: BLE001 -- temporal must never break /detect
        log.warning("temporal: attach failed (returning detection): %s", exc)
        return resp_dict


def _vlm_enabled() -> bool:
    try:
        import risk.vlm_reasoner as vlm
        return vlm.enabled()
    except Exception:  # noqa: BLE001
        return False


def _vlm_mode() -> str:
    try:
        import risk.vlm_reasoner as vlm
        return vlm.mode()
    except Exception:  # noqa: BLE001
        return "unknown"


def config() -> Dict[str, Any]:
    """Non-sensitive snapshot for GET /debug/state."""
    return {
        "enabled": enabled(),
        "vlm_enabled": _vlm_enabled(),
        "mode": _vlm_mode(),
        "scene_context_enabled": scene_context.enabled(),
        "semantic_correction_enabled": semantic_corrections.enabled(),
        "contextual_suppression_enabled": semantic_corrections.contextual_suppression_enabled(),
        "object_edge_risk_enabled": edge_risk.enabled(),
        "memory_window_frames": mem.memory_window_frames(),
        "memory_ttl_ms": mem.memory_ttl_ms(),
        "max_active_sessions": mem.max_active_sessions(),
        "store_keyframes": mem.store_keyframes(),
        "active_sessions": mem.active_session_count(),
        "max_async_jobs": async_reasoning._max_async_jobs(),
        "trigger_min_interval_ms": async_reasoning._min_interval_ms(),
        "result_stale_ms": _result_stale_ms(),
        "human_review_score": _int_env("REASONER_HUMAN_REVIEW_SCORE", 10),
        "note": ("VLM is event-triggered + non-blocking; /detect never waits. "
                 "Perception corrections are advisory; safety drafts require human review."),
    }


def status_snapshot() -> Dict[str, Any]:
    """Alias used by /debug/state and /metrics."""
    return config()


def pending_reasoner_jobs() -> int:
    with async_reasoning._LOCK:
        return len(async_reasoning._INFLIGHT)


def reset_all() -> None:
    """Test helper: clear memory + inflight state."""
    mem.reset()
    async_reasoning.reset()


__all__ = ["attach_temporal", "config", "status_snapshot", "enabled",
           "pending_reasoner_jobs", "reset_all"]
