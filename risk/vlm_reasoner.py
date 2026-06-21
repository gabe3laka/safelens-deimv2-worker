"""
risk/vlm_reasoner.py -- event-driven vision reasoning adapter for POST /reason
and the non-blocking /detect trigger.

Design / safety rules (hard):
  * The deterministic engine is the safety signal. The VLM only explains /
    verifies / drafts AFTER a deterministic candidate exists. Its output is an
    AI DRAFT: produced_by="vlm_reasoner", requires_human_review=True,
    should_alert=False (enforced by reason_schema, not trusted from the model).
  * NEVER per-frame. /detect uses maybe_trigger(): rate-limited
    (REASONER_MIN_INTERVAL_MS), triggered only at/above REASONER_TRIGGER_LEVEL,
    run on a bounded background executor, and it NEVER blocks the live loop --
    /detect attaches the most recent cached draft (if any) + a reasoner_status
    and returns immediately.
  * Privacy: when PRIVACY_BLUR_ENABLED, the frame is blurred (persons) before it
    is ever passed to the model. No un-blurred frame reaches the VLM.

Modes (REASONER_MODE): gemini (default) | mock | disabled.
`mock` lets the app integrate the full contract on CPU with no weights.
`qwen_vl` and `deepseek_vl2` are no longer supported; they degrade to
reasoner_status="unavailable" with a clear error message -- no transformers or
weights are ever loaded for these modes.
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from typing import Any, Dict, List, Optional, Tuple

from . import controls, gemini_reasoner, privacy
from .reason_schema import ReasonRequest, ReasonResponse, VlmRisk

log = logging.getLogger("safelens-vision-worker.vlm")

_LEVEL = {"GREEN": 0, "YELLOW": 1, "ORANGE": 2, "RED": 3}


# -- flags / config -----------------------------------------------------------

def enabled() -> bool:
    return os.getenv("VLM_REASONER_ENABLED", "false").strip().lower() in ("1", "true", "yes", "on")


def mode() -> str:
    return os.getenv("REASONER_MODE", "gemini").strip().lower()


def _model_id() -> str:
    m = mode()
    if m == "gemini":
        return gemini_reasoner.model_id()
    if m == "mock":
        return "mock"
    if m == "disabled":
        return "disabled"
    return "unknown"


def trigger_level() -> str:
    return os.getenv("REASONER_TRIGGER_LEVEL", "YELLOW").strip().upper()


def _min_interval_ms() -> int:
    try:
        return int(os.getenv("REASONER_MIN_INTERVAL_MS", "1500"))
    except (TypeError, ValueError):
        return 1500


def _timeout_ms() -> int:
    try:
        return int(os.getenv("REASONER_TIMEOUT_MS", "2500"))
    except (TypeError, ValueError):
        return 2500


def _cache_ttl_ms() -> int:
    try:
        return int(os.getenv("REASONER_CACHE_TTL_MS", "10000"))
    except (TypeError, ValueError):
        return 10000


def _max_sessions() -> int:
    try:
        return int(os.getenv("REASONER_MAX_SESSIONS", "64"))
    except (TypeError, ValueError):
        return 64


def _max_image_side() -> int:
    try:
        return int(os.getenv("REASONER_MAX_IMAGE_SIDE", "512"))
    except (TypeError, ValueError):
        return 512


def _now_ms() -> int:
    return int(time.time() * 1000)


# -- per-session cache + non-blocking executor --------------------------------

_LOCK = threading.RLock()
_CACHE: Dict[str, Dict[str, Any]] = {}      # session -> {"response": dict, "ts": ms}
_LAST_RUN_MS: Dict[str, int] = {}
_INFLIGHT: set = set()
_LAST_STATUS: Dict[str, Any] = {"status": "idle", "ts": 0}
_EXECUTOR: Optional[ThreadPoolExecutor] = None
_ADAPTER_STATE: Dict[str, Any] = {}         # mode -> built adapter dict


def _executor() -> ThreadPoolExecutor:
    global _EXECUTOR
    if _EXECUTOR is None:
        workers = 1
        try:
            workers = max(1, int(os.getenv("REASONER_MAX_WORKERS", "1")))
        except (TypeError, ValueError):
            workers = 1
        _EXECUTOR = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="vlm-reasoner")
    return _EXECUTOR


def _sweep(now_ms: int) -> None:
    ttl = max(_cache_ttl_ms(), _min_interval_ms()) * 4
    for sid in [s for s, v in list(_CACHE.items()) if now_ms - v.get("ts", now_ms) > ttl]:
        _CACHE.pop(sid, None)
        _LAST_RUN_MS.pop(sid, None)
    # bound active sessions
    while len(_CACHE) > _max_sessions():
        oldest = min(_CACHE.items(), key=lambda kv: kv[1].get("ts", 0))[0]
        _CACHE.pop(oldest, None)
        _LAST_RUN_MS.pop(oldest, None)


def get_cached_draft(session_id: Optional[str], max_age_ms: Optional[int] = None) -> Optional[Dict[str, Any]]:
    """Return the freshest cached ReasonResponse for a session, or None."""
    sid = session_id or "__default__"
    horizon = max_age_ms if max_age_ms is not None else _cache_ttl_ms()
    with _LOCK:
        entry = _CACHE.get(sid)
        if entry and (_now_ms() - entry["ts"]) <= horizon:
            return dict(entry["response"])
    return None


def reset() -> None:
    """Clear caches + state (tests / shutdown)."""
    with _LOCK:
        _CACHE.clear()
        _LAST_RUN_MS.clear()
        _INFLIGHT.clear()
        _LAST_STATUS.update(status="idle", ts=0)
    global _ADAPTER_STATE
    _ADAPTER_STATE = {}


# -- public: non-blocking trigger used by /detect -----------------------------

def maybe_trigger(session_id: Optional[str], *, frame_b64: Optional[str],
                  highest_level: str, deterministic_risks: List[Dict[str, Any]],
                  entities: Optional[List[Dict[str, Any]]] = None,
                  scene_graph: Optional[Dict[str, Any]] = None,
                  tracks: Optional[List[Dict[str, Any]]] = None,
                  frame_id: Optional[str] = None,
                  force_reason: bool = False) -> Tuple[Optional[Dict[str, Any]], str]:
    """Maybe kick an async VLM reason; return (cached_draft_or_None, status).

    NEVER blocks: it submits work to a bounded executor and returns the most
    recent cached draft immediately. status is one of:
      disabled | not_triggered | throttled | triggered | cached | cached_and_triggered
      | timeout | error | schema_error | json_parse_error
    """
    if not enabled():
        return None, "disabled"
    sid = session_id or "__default__"
    now = _now_ms()
    should = _LEVEL.get((highest_level or "GREEN").upper(), 0) >= _LEVEL.get(trigger_level(), 2)
    with _LOCK:
        _sweep(now)
        draft = None
        entry = _CACHE.get(sid)
        if entry and (now - entry["ts"]) <= _cache_ttl_ms():
            draft = dict(entry["response"])
            draft["_cached_at_ms"] = entry["ts"]
            draft["_cache_age_ms"] = now - entry["ts"]
        cached_status = _cached_reasoner_status(draft)
        if not should:
            return draft, (cached_status if draft else "not_triggered")
        if draft and _is_terminal_failure_status(cached_status) and not force_reason:
            return draft, cached_status
        last = _LAST_RUN_MS.get(sid, 0)
        if sid in _INFLIGHT or (now - last) < _min_interval_ms():
            return draft, (cached_status if draft else "throttled")
        # trigger
        _LAST_RUN_MS[sid] = now
        _INFLIGHT.add(sid)
    req = {
        "session_id": sid, "frame_id": frame_id, "frame_b64": frame_b64,
        "entities": entities or [], "tracks": tracks or [],
        "scene_graph": scene_graph or {}, "deterministic_risks": deterministic_risks or [],
    }
    try:
        _executor().submit(_run_and_cache, sid, req)
    except Exception as exc:  # noqa: BLE001 -- executor refusal must never break /detect
        with _LOCK:
            _INFLIGHT.discard(sid)
        log.warning("vlm: could not submit reason job: %s", exc)
        return draft, (cached_status if draft else "error")
    return draft, ("cached_and_triggered" if draft else "triggered")


def _is_terminal_failure_status(status: Optional[str]) -> bool:
    return status in {"schema_error", "json_parse_error", "error", "timeout"}


def _cached_reasoner_status(draft: Optional[Dict[str, Any]]) -> str:
    if not draft:
        return "cached"
    status = draft.get("reasoner_status")
    if status in {"ok", "ready", "completed"}:
        return "cached"
    if isinstance(status, str) and status:
        return status
    return "cached"


def _cache_terminal_response(sid: str, resp: Dict[str, Any]) -> None:
    with _LOCK:
        _CACHE[sid] = {"response": resp, "ts": _now_ms()}
        _LAST_STATUS.update(status=resp.get("reasoner_status", "ok"), ts=_now_ms())


def _run_and_cache(sid: str, req: Dict[str, Any]) -> None:
    log.info("vlm_job_started", extra={"session_id": sid, "frame_id": req.get("frame_id")})
    timeout_s = max(0.05, _timeout_ms() / 1000.0)
    try:
        # Use a one-shot worker so the background /detect orchestration can impose
        # the same hard deadline as /reason without blocking the bounded trigger pool.
        ex = ThreadPoolExecutor(max_workers=1, thread_name_prefix="vlm-reason-timeout")
        fut = ex.submit(reason_sync, req)
        try:
            resp = fut.result(timeout=timeout_s)
        except FutureTimeout:
            log.warning("vlm_timeout", extra={"session_id": sid, "frame_id": req.get("frame_id")})
            resp = ReasonResponse(
                reasoner_status="timeout", reasoner_model=_model_id(),
                session_id=sid, frame_id=req.get("frame_id"),
                error=f"timeout after {timeout_s}s",
            ).enforce_draft_contract().model_dump()
            fut.cancel()
        finally:
            ex.shutdown(wait=False, cancel_futures=True)
        _cache_terminal_response(sid, resp)
        log.info("vlm_result_stored", extra={
            "session_id": sid, "frame_id": req.get("frame_id"),
            "reasoner_status": resp.get("reasoner_status"),
        })
    except Exception as exc:  # noqa: BLE001 -- background must never crash the worker
        log.warning("vlm_error", extra={"session_id": sid, "frame_id": req.get("frame_id")}, exc_info=True)
        resp = ReasonResponse(
            reasoner_status="error", reasoner_model=_model_id(),
            session_id=sid, frame_id=req.get("frame_id"),
            error=f"{type(exc).__name__}: {exc}",
        ).enforce_draft_contract().model_dump()
        _cache_terminal_response(sid, resp)
    finally:
        with _LOCK:
            _INFLIGHT.discard(sid)


# -- public: the /reason endpoint paths ---------------------------------------

def reason_sync(payload: Any) -> Dict[str, Any]:
    """Run the reasoner for one request and ALWAYS return a strict response dict.

    Never raises. Blocking (the caller imposes the timeout). Used directly by
    the background trigger and (via to_thread) by reason_async / POST /reason.
    """
    t0 = time.perf_counter()
    req = payload if isinstance(payload, ReasonRequest) else ReasonRequest(**(payload or {}))
    base = dict(reasoner_model=_model_id(), request_id=req.request_id,
                session_id=req.session_id, frame_id=req.frame_id)

    if not enabled():
        return _finish(ReasonResponse(reasoner_status="disabled", **base), t0)

    m = mode()
    try:
        if m == "mock":
            resp = _mock_reason(req)
        elif m == "gemini":
            resp = _gemini_reason(req)
        elif m == "disabled":
            resp = ReasonResponse(reasoner_status="disabled", **base)
        else:
            resp = ReasonResponse(
                reasoner_status="unavailable",
                error=f"unknown or removed REASONER_MODE={m}",
            )
    except Exception as exc:  # noqa: BLE001
        log.warning("vlm: reason failed: %s", exc)
        resp = ReasonResponse(reasoner_status="error", error=f"{type(exc).__name__}: {exc}")

    for k, v in base.items():
        setattr(resp, k, getattr(resp, k, None) or v)
    return _finish(resp, t0)


def _finish(resp: ReasonResponse, t0: float) -> Dict[str, Any]:
    resp.latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
    return resp.enforce_draft_contract().model_dump()


async def reason_async(payload: Any) -> Dict[str, Any]:
    """Async wrapper for POST /reason with a hard timeout -> reasoner_status=timeout."""
    timeout_s = max(0.05, _timeout_ms() / 1000.0)
    req = payload if isinstance(payload, dict) else {}
    try:
        return await asyncio.wait_for(asyncio.to_thread(reason_sync, payload), timeout_s)
    except asyncio.TimeoutError:
        resp = ReasonResponse(
            reasoner_status="timeout", reasoner_model=_model_id(),
            request_id=req.get("request_id"), session_id=req.get("session_id"),
            frame_id=req.get("frame_id"), error=f"timeout after {timeout_s}s")
        return resp.enforce_draft_contract().model_dump()
    except Exception as exc:  # noqa: BLE001
        resp = ReasonResponse(reasoner_status="error", reasoner_model=_model_id(),
                              error=f"{type(exc).__name__}: {exc}")
        return resp.enforce_draft_contract().model_dump()


# -- mock reasoner (no weights; deterministic; for tests + CPU integration) ----

def _mock_reason(req: ReasonRequest) -> ReasonResponse:
    """Synthesize a plausible AI draft from the deterministic risks.

    This is NOT a fake-success of the real model -- it is an explicit, labelled
    mock path (reasoner_model='mock') so the app can wire the full /reason
    contract before a GPU/Gemini deployment exists.
    """
    risks: List[VlmRisk] = []
    for i, dr in enumerate(req.deterministic_risks):
        hz = str(dr.get("hazard_type", "unknown"))
        ctrls = dr.get("recommended_controls") or controls.controls_for(hz)
        from .risk_schema import Control as _C
        risks.append(VlmRisk(
            risk_id=f"vlm_{dr.get('risk_id', 'risk_%d' % i)}",
            involved_track_ids=list(dr.get("involved_track_ids", []) or []),
            hazard_type=hz, risk_state=str(dr.get("risk_state", "latent")),
            trigger_condition=("May become active under a trigger (movement/contact)."
                               if dr.get("risk_state") == "latent" else "Active now."),
            risk_level=str(dr.get("risk_level", "GREEN")),
            severity=int(dr.get("severity", 1)), likelihood=int(dr.get("likelihood", 1)),
            risk_score=int(dr.get("risk_score", 1)),
            reason=f"VLM draft (mock) explaining deterministic hazard '{hz}'.",
            visual_evidence=[hz],
            recommended_controls=[_C(**c) if isinstance(c, dict) else c for c in ctrls],
            recommended_action=(dr.get("recommended_action")
                                or (ctrls[0]["action"] if ctrls and isinstance(ctrls[0], dict) else None)),
            confidence=0.6,
        ))
    labels = [str(e.get("label")) for e in req.entities][:8]
    summary = (f"Mock VLM scene summary: {len(req.entities)} object(s)"
               + (f" ({', '.join(labels)})" if labels else "")
               + f"; {len(risks)} deterministic candidate(s) explained.")
    return ReasonResponse(reasoner_status="ok", reasoner_model="mock",
                          scene_summary=summary, risks=risks)


# -- Gemini vision reasoner ---------------------------------------------------


def _gemini_reason(req: ReasonRequest) -> ReasonResponse:
    adapter = _get_adapter("gemini")
    if not adapter["available"]:
        return ReasonResponse(reasoner_status="unavailable",
                              error=adapter.get("error", "Gemini adapter unavailable"))
    image = _decode_blurred(req)
    prompt = _build_gemini_prompt(req)
    try:
        log.info("gemini_generate_started", extra={"session_id": req.session_id, "frame_id": req.frame_id})
        raw = adapter["generate"](prompt, image)
        log.info("gemini_generate_completed", extra={"session_id": req.session_id, "frame_id": req.frame_id})
    except Exception as exc:  # noqa: BLE001
        return ReasonResponse(reasoner_status="error", error=f"gemini generate: {exc}")
    data = _extract_json(raw)
    if data is None:
        log.warning(
            "gemini_json_parse_failed session_id=%s frame_id=%s raw_excerpt=%r",
            req.session_id, req.frame_id, (raw or "")[:400],
        )
        return ReasonResponse(reasoner_status="json_parse_error",
                              error="Gemini did not return valid JSON",
                              scene_summary="", risks=[], uncertain_items=[])
    try:
        resp = ReasonResponse(**{k: v for k, v in data.items()
                                 if k in ReasonResponse.model_fields})
        resp.reasoner_status = "ok"
        return resp
    except Exception as exc:  # noqa: BLE001
        log.warning("gemini_schema_failed", extra={"session_id": req.session_id, "frame_id": req.frame_id})
        return ReasonResponse(reasoner_status="schema_error", error=f"schema: {exc}")


def _build_gemini_prompt(req: ReasonRequest) -> str:
    max_labels = gemini_reasoner._max_detected_labels()
    context = {
        "deterministic_risks": req.deterministic_risks[:10],
        "entities": req.entities[:max_labels],
        "tracks": req.tracks[:20],
        "scene_graph": {"relations": (req.scene_graph or {}).get("relations", [])[:20]},
    }
    schema_hint = {
        "scene_summary": "Short visible scene description.",
        "risks": [
            {
                "risk_id": "gemini_1",
                "hazard_type": "object_near_edge",
                "risk_level": "YELLOW",
                "risk_state": "active",
                "linked_entity_id": "track_1",
                "involved_track_ids": ["track_1"],
                "involved_detection_ids": [],
                "bbox": None,
                "reason": "The object appears close to an edge.",
                "visual_evidence": ["object near edge"],
                "recommended_action": "Move the object away from the edge.",
                "confidence": 0.7,
            }
        ],
        "uncertain_items": [],
    }
    return (
        "Return JSON only. No markdown. No prose. No code fences. "
        "If uncertain, return an empty risks array. At most 3 risks. "
        "Every risk MUST include at least one linkability field: "
        "linked_entity_id, involved_track_ids, involved_detection_ids, bbox, or approximate_region. "
        "STRICT RULES: Do not invent risks; risks: [] is a successful answer. "
        "Do not set should_alert or requires_human_review. "
        "Do not output bbox coordinates for detections -- YOLO is the coordinate truth. "
        "Use this small schema shape exactly; omit fields you cannot support.\n"
        f"Target JSON example: {json.dumps(schema_hint, separators=(',', ':'))}\n"
        "Detector/tracker context:\n"
        + json.dumps(context, default=str, separators=(",", ":"))[:4000]
    )


def _decode_blurred(req: ReasonRequest):
    """Decode frame_b64 and blur persons before the model sees it (privacy)."""
    if not req.frame_b64:
        return None
    try:
        from PIL import Image
        raw = base64.b64decode(req.frame_b64)
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        side = _max_image_side()
        if max(img.size) > side:
            img.thumbnail((side, side))
        # Privacy egress guard (B8): no un-blurred frame may reach the VLM.
        if privacy.blur_enabled():
            img, _blurred = privacy.sanitize_for_egress(img, req.entities)
        return img
    except Exception as exc:  # noqa: BLE001
        log.warning("vlm: frame decode/blur failed: %s", exc)
        return None


def _get_adapter(m: str) -> Dict[str, Any]:
    """Lazily build/cache a generate() callable for the mode. Never raises."""
    with _LOCK:
        st = _ADAPTER_STATE.get(m)
        if st is not None:
            return st
    st = _build_adapter(m)
    with _LOCK:
        _ADAPTER_STATE[m] = st
    return st


def _build_adapter(m: str) -> Dict[str, Any]:
    """Build the adapter for the given mode. Gemini only; other modes are unavailable."""
    if m == "gemini":
        return gemini_reasoner.build_adapter()
    return {
        "available": False,
        "error": f"REASONER_MODE={m} is not available in live vision reasoner",
        "generate": None,
        "model_id": _model_id(),
        "diagnostics": {},
    }


# -- reusable raw-JSON generation (temporal perception layer) -----------------

def generate_json(prompt: str, *, frame_b64: Optional[str] = None,
                  entities: Optional[List[Dict[str, Any]]] = None) -> Optional[Dict[str, Any]]:
    """Run the configured VLM on (prompt, optional frame) and return parsed JSON.

    Reuses the same lazy adapter + the privacy blur in _decode_blurred, so no
    un-blurred frame ever reaches the model. Returns None in mock mode, when the
    model/deps are unavailable, or when the model does not emit valid JSON. Never
    raises -- callers degrade on None. This is the shared bridge the temporal
    perception layer (scene_context / semantic_corrections) uses so it does not
    duplicate model loading.
    """
    m = mode()
    if m == "mock" or not enabled():
        return None
    adapter = _get_adapter(m)
    if not adapter.get("available"):
        return None
    req = ReasonRequest(frame_b64=frame_b64, entities=entities or [])
    image = _decode_blurred(req)
    try:
        raw = adapter["generate"](prompt, image)
    except Exception as exc:  # noqa: BLE001
        log.warning("vlm: generate_json failed: %s", exc)
        return None
    return _extract_json(raw)


def adapter_available() -> bool:
    """True when the Gemini adapter is loaded and available.

    Returns False in mock/disabled modes and when GEMINI_API_KEY is missing.
    Never raises.
    """
    if not enabled() or mode() != "gemini":
        return False
    try:
        return bool(_get_adapter("gemini").get("available"))
    except Exception:  # noqa: BLE001
        return False


# -- status (for /debug/state) ------------------------------------------------

def status_snapshot() -> Dict[str, Any]:
    with _LOCK:
        active = len(_CACHE)
        last = dict(_LAST_STATUS)
    m = mode()
    snap: Dict[str, Any] = {
        "enabled": enabled(),
        "mode": m,
        "model_id": _model_id(),
        "serve_backend": "google_genai" if m == "gemini" else m,
        "trigger_level": trigger_level(),
        "min_interval_ms": _min_interval_ms(),
        "timeout_ms": _timeout_ms(),
        "cache_ttl_ms": _cache_ttl_ms(),
        "max_image_side": _max_image_side(),
        "privacy_blur_enabled": privacy.blur_enabled(),
        "active_sessions": active,
        "last_status": last,
    }
    if m == "gemini":
        snap.update({
            "gemini_max_output_tokens": gemini_reasoner._max_output_tokens(),
            "gemini_temperature": gemini_reasoner._temperature(),
            "gemini_max_detected_labels": gemini_reasoner._max_detected_labels(),
        })
    return snap


# -- JSON extraction helpers --------------------------------------------------

def _json_candidates(s: str) -> List[str]:
    """Return balanced ``{...}`` substrings in order, respecting strings/escapes.

    Unlike a naive ``find('{')`` / ``rfind('}')`` span (which breaks when the
    model emits more than one object or trailing prose), this walks the text with
    a brace-depth counter and skips braces inside string literals.
    """
    out: List[str] = []
    depth = 0
    start = -1
    in_str = False
    esc = False
    for i, ch in enumerate(s):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start != -1:
                    out.append(s[start:i + 1])
                    start = -1
    return out


def _strip_code_fences(s: str) -> str:
    """Return the body of the first ``` fenced block, else the input unchanged."""
    if "```" not in s:
        return s
    for body in s.split("```")[1:]:
        body = body.strip()
        if not body:
            continue
        # drop an optional leading language tag (e.g. ```json\n{...})
        nl = body.find("\n")
        if nl != -1 and "{" not in body[:nl] and " " not in body[:nl].strip():
            body = body[nl + 1:]
        if "{" in body:
            return body
    return s


def _extract_json(raw: Any) -> Optional[Dict[str, Any]]:
    """Best-effort parse of model output into a JSON object.

    Order: direct parse -> first markdown-fenced block -> balanced-brace
    candidate scan. Tolerates code fences, prose before/after the JSON, and
    multiple or nested objects. Prefers a candidate with the expected top-level
    keys. Returns None only when no balanced object parses.
    """
    if not raw:
        return None
    s = (raw if isinstance(raw, str) else str(raw)).strip()
    if not s:
        return None

    def _as_obj(text: str) -> Optional[Dict[str, Any]]:
        try:
            val = json.loads(text)
        except (ValueError, TypeError):
            return None
        return val if isinstance(val, dict) else None

    # 1) plain JSON object
    obj = _as_obj(s)
    if obj is not None:
        return obj
    # 2) first markdown-fenced block (```json ... ``` or ``` ... ```)
    fenced = _strip_code_fences(s)
    if fenced is not s:
        obj = _as_obj(fenced.strip())
        if obj is not None:
            return obj
    # 3) balanced-brace candidates (handles prose / multiple / nested objects)
    candidates = _json_candidates(fenced) or _json_candidates(s)
    parsed = [o for o in (_as_obj(c) for c in candidates) if o is not None]
    if not parsed:
        return None
    for o in parsed:
        if "scene_summary" in o or "risks" in o or "scene_context" in o:
            return o
    return parsed[0]
