"""
temporal_reasoning/async_reasoning.py -- non-blocking VLM orchestration.

The detector runs every frame; this runs the VLM ONLY on a trigger, on a bounded
background executor, under a bounded GPU slot (gpu_vision). It NEVER blocks
/detect: maybe_trigger() submits work and returns immediately, and /detect reads
the most recent cached scene_context / semantic_corrections from session_memory.

Bounds (no unbounded queue):
  * one in-flight reasoner job per session
  * optional latest-frame-wins single pending replacement queue per session
  * global cap TEMPORAL_REASONING_MAX_ASYNC_JOBS
  * rate-limit max(TEMPORAL_REASONING_TRIGGER_MIN_INTERVAL_MS, REASONER_MIN_INTERVAL_MS)
  * a GPU slot must be free (gpu_vision.gpu_reasoner_slot) or the job is dropped
A newer trigger while a session job is in flight can replace one pending job (latest-wins).
"""

from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import scene_context as scenectx
from . import semantic_corrections as corrections
from . import session_memory as mem
from .session_memory import _int_env

log = logging.getLogger("safelens-vision-worker.temporal")

_LOCK = threading.RLock()
_INFLIGHT: set = set()
_LAST_TRIGGER_MS: Dict[str, int] = {}
_PENDING: Dict[str, Dict[str, Any]] = {}
_EXECUTOR: Optional[ThreadPoolExecutor] = None

_PROMPT_PATH = Path(__file__).resolve().parent.parent / "shared" / "prompts" / "senior_qhse_perception.md"
_PROMPT_CACHE: Dict[str, str] = {}


def _max_async_jobs() -> int:
    return max(1, _int_env("TEMPORAL_REASONING_MAX_ASYNC_JOBS", 1))


def _min_interval_ms() -> int:
    return max(_int_env("TEMPORAL_REASONING_TRIGGER_MIN_INTERVAL_MS", 1500),
               _int_env("REASONER_MIN_INTERVAL_MS", 1500))


def _latest_wins_enabled() -> bool:
    return _int_env("REASONER_LATEST_WINS", 1) != 0


def _pending_max_age_ms() -> int:
    return max(1, _int_env("REASONER_PENDING_FRAME_MAX_AGE_MS", 2500))


def _now_ms() -> int:
    return int(time.time() * 1000)


def _executor() -> ThreadPoolExecutor:
    global _EXECUTOR
    if _EXECUTOR is None:
        _EXECUTOR = ThreadPoolExecutor(max_workers=_max_async_jobs(),
                                       thread_name_prefix="temporal-reasoner")
    return _EXECUTOR


def _vlm():
    """Lazy import so importing the temporal package never pulls risk.vlm_reasoner."""
    import risk.vlm_reasoner as vlm
    return vlm


def _load_prompt() -> str:
    if "p" not in _PROMPT_CACHE:
        try:
            _PROMPT_CACHE["p"] = _PROMPT_PATH.read_text()
        except Exception:  # noqa: BLE001
            _PROMPT_CACHE["p"] = ("You are a senior QHSE manager. Correct detector "
                                  "mislabels using scene context and note real hazards. "
                                  "Perception corrections need no approval; safety drafts do.")
    return _PROMPT_CACHE["p"]


def _build_prompt(ctx: Dict[str, Any]) -> str:
    import json
    schema_hint = (
        '{"scene_context": {"scene_type": str, "environment_type": str, '
        '"confidence": float, "reason": str}, "semantic_corrections": '
        '[{"track_id": str, "raw_label": str, "corrected_label": str, '
        '"correction_type": "false_positive|relabel|suppress", "action": str, '
        '"confidence": float, "reason": str}]}')
    context = {
        "scene_hint": ctx.get("payload", {}).get("scene_hint"),
        "site_context": ctx.get("payload", {}).get("site_context"),
        "entities": [{"label": e.get("label"), "confidence": e.get("confidence")}
                     for e in (ctx.get("entities") or [])][:30],
        "trigger_reasons": ctx.get("reasons", []),
    }
    return (_load_prompt() + "\n\nReturn STRICT compact JSON only (no prose/code fences),"
            " max 3 semantic_corrections; keep evidence/actions"
            " to one short sentence each. Return empty arrays when unsupported."
            "\nReturn STRICT JSON ONLY matching:\n" + schema_hint
            + "\nContext:\n" + json.dumps(context, default=str)[:4000])


def _submit_job(sid: str, ctx: Dict[str, Any]) -> bool:
    try:
        _executor().submit(_run_job, sid, ctx)
        return True
    except Exception as exc:  # noqa: BLE001 -- executor refusal must never break /detect
        with _LOCK:
            _INFLIGHT.discard(sid)
        log.warning("temporal: could not submit reasoning job: %s", exc)
        return False


def _run_job(sid: str, ctx: Dict[str, Any]) -> None:
    """Background reasoning job. Never raises into the worker."""
    try:
        from gpu_vision import gpu_reasoner_slot
    except Exception:  # noqa: BLE001
        gpu_reasoner_slot = None  # type: ignore

    slot_cm = gpu_reasoner_slot() if gpu_reasoner_slot else None
    acquired = True
    try:
        if slot_cm is not None:
            acquired = slot_cm.__enter__()
        if not acquired:
            # GPU saturated -> drop this job (counted in gpu_vision metrics).
            mem.set_reasoner_state(sid, "throttled")
            return
        mem.set_reasoner_state(sid, "running")
        vlm = _vlm()
        m = vlm.mode()
        entities = ctx.get("entities") or []
        payload = ctx.get("payload") or {}
        if m == "mock":
            sc = scenectx.mock_scene_context(entities, payload)
            corr = corrections.mock_corrections(entities, sc, ctx.get("tracks"))
            mem.store_vlm_result(sid, scene_context=sc, semantic_corrections=corr,
                                 vlm_result={"mode": "mock"})
            return
        # real model path (lazy, blurred frame) -- weight-free if unavailable
        data = vlm.generate_json(_build_prompt(ctx),
                                 frame_b64=ctx.get("frame_b64"), entities=entities)
        if not data:
            mem.set_reasoner_state(sid, "unavailable")
            return
        sc = scenectx.from_vlm_json(data) or scenectx.mock_scene_context(entities, payload)
        corr = corrections.from_vlm_json(data, entities)
        mem.store_vlm_result(sid, scene_context=sc, semantic_corrections=corr,
                             vlm_result={"mode": m})
    except Exception as exc:  # noqa: BLE001
        log.warning("temporal: reasoning job failed: %s", exc)
        try:
            mem.set_reasoner_state(sid, "error")
        except Exception:  # noqa: BLE001
            pass
    finally:
        if slot_cm is not None and acquired:
            try:
                slot_cm.__exit__(None, None, None)
            except Exception:  # noqa: BLE001
                pass
        with _LOCK:
            _INFLIGHT.discard(sid)
            pending = _PENDING.pop(sid, None)
            should_submit_pending = False
            pending_ctx: Dict[str, Any] = {}
            if pending:
                age = _now_ms() - int(pending.get("queued_ms", 0))
                if age <= _pending_max_age_ms() and len(_INFLIGHT) < _max_async_jobs():
                    _INFLIGHT.add(sid)
                    _LAST_TRIGGER_MS[sid] = _now_ms()
                    should_submit_pending = True
                    pending_ctx = pending.get("ctx") or {}
                    mem.set_reasoner_state(sid, "queued_latest",
                                           trigger=((pending_ctx.get("reasons") or [None])[0]))
        if should_submit_pending:
            _submit_job(sid, pending_ctx)


def maybe_trigger(session_id: Optional[str], *, reasons: List[str],
                  entities: List[Dict[str, Any]], tracks: List[Dict[str, Any]],
                  frame_b64: Optional[str], payload: Dict[str, Any]) -> str:
    """Maybe submit an async reasoning job. Returns a status string; NEVER blocks.

    Status: disabled | not_triggered | throttled | triggered | running | queued_latest
    """
    vlm = _vlm()
    if not vlm.enabled():
        return "disabled"
    if not reasons:
        return "not_triggered"
    sid = session_id or "__default__"
    now = _now_ms()
    ctx = {"entities": entities, "tracks": tracks, "frame_b64": frame_b64,
           "payload": payload or {}, "reasons": reasons}
    with _LOCK:
        if sid in _INFLIGHT:
            if _latest_wins_enabled():
                _PENDING[sid] = {"ctx": ctx, "queued_ms": now}
                mem.set_reasoner_state(sid, "queued_latest",
                                       trigger=(reasons[0] if reasons else None))
                return "queued_latest"
            return "running"
        last = _LAST_TRIGGER_MS.get(sid, 0)
        if (now - last) < _min_interval_ms():
            return "throttled"
        if len(_INFLIGHT) >= _max_async_jobs():
            return "throttled"
        _INFLIGHT.add(sid)
        _LAST_TRIGGER_MS[sid] = now
    mem.set_reasoner_state(sid, "queued", trigger=(reasons[0] if reasons else None))
    if not _submit_job(sid, ctx):
        return "throttled"
    try:
        import worker_runtime as runtime
        runtime.inc("temporal_triggers_total", {"reason": reasons[0] if reasons else "unknown"})
    except Exception:  # noqa: BLE001
        pass
    return "triggered"


def reset() -> None:
    with _LOCK:
        _INFLIGHT.clear()
        _LAST_TRIGGER_MS.clear()
        _PENDING.clear()
