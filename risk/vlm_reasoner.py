"""
risk/vlm_reasoner.py -- REAL event-driven Qwen-VL (and optional DeepSeek-VL2)
reasoning adapter for POST /reason and the non-blocking /detect trigger.

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
  * Real but lazy: torch/transformers are imported only on first model use, and
    weights resolve at runtime into REASONER_CACHE_DIR / the HF cache (NEVER
    baked at Docker build). If the model/deps are unavailable the worker
    degrades to reasoner_status="unavailable"/"timeout"/"disabled" with empty
    risks -- it never raises into the request path.
  * Privacy: when PRIVACY_BLUR_ENABLED, the frame is blurred (persons) before it
    is ever passed to the model. No un-blurred frame reaches the VLM.

Modes (REASONER_MODE): qwen_vl (default) | deepseek_vl2 | mock | disabled.
`mock` lets the app integrate the full contract on CPU with no weights.
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
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

from . import controls, privacy
from .reason_schema import ReasonRequest, ReasonResponse, VlmRisk

log = logging.getLogger("safelens-vision-worker.vlm")

_LEVEL = {"GREEN": 0, "YELLOW": 1, "ORANGE": 2, "RED": 3}


# -- flags / config -----------------------------------------------------------

def enabled() -> bool:
    return os.getenv("VLM_REASONER_ENABLED", "false").strip().lower() in ("1", "true", "yes", "on")


def mode() -> str:
    return os.getenv("REASONER_MODE", "qwen_vl").strip().lower()


def _model_id() -> str:
    m = mode()
    if m == "deepseek_vl2":
        return os.getenv("DEEPSEEK_VL_MODEL_ID", "deepseek-ai/deepseek-vl2-small")
    if m == "mock":
        return "mock"
    return os.getenv("QWEN_VL_MODEL_ID", "Qwen/Qwen2.5-VL-7B-Instruct")


def trigger_level() -> str:
    return os.getenv("REASONER_TRIGGER_LEVEL", "ORANGE").strip().upper()


def _min_interval_ms() -> int:
    try:
        return int(os.getenv("REASONER_MIN_INTERVAL_MS", "5000"))
    except (TypeError, ValueError):
        return 5000


def _timeout_ms() -> int:
    try:
        return int(os.getenv("REASONER_TIMEOUT_MS", "8000"))
    except (TypeError, ValueError):
        return 8000


def _cache_ttl_ms() -> int:
    try:
        return int(os.getenv("REASONER_CACHE_TTL_MS", "15000"))
    except (TypeError, ValueError):
        return 15000


def _max_sessions() -> int:
    try:
        return int(os.getenv("REASONER_MAX_SESSIONS", "64"))
    except (TypeError, ValueError):
        return 64


def _max_image_side() -> int:
    try:
        return int(os.getenv("REASONER_MAX_IMAGE_SIDE", "1024"))
    except (TypeError, ValueError):
        return 1024


def _now_ms() -> int:
    return int(time.time() * 1000)


# -- per-session cache + non-blocking executor --------------------------------

_LOCK = threading.RLock()
_CACHE: Dict[str, Dict[str, Any]] = {}      # session -> {"response": dict, "ts": ms}
_LAST_RUN_MS: Dict[str, int] = {}
_INFLIGHT: set = set()
_LAST_STATUS: Dict[str, Any] = {"status": "idle", "ts": 0}
_EXECUTOR: Optional[ThreadPoolExecutor] = None


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


# -- public: non-blocking trigger used by /detect -----------------------------

def maybe_trigger(session_id: Optional[str], *, frame_b64: Optional[str],
                  highest_level: str, deterministic_risks: List[Dict[str, Any]],
                  entities: Optional[List[Dict[str, Any]]] = None,
                  scene_graph: Optional[Dict[str, Any]] = None,
                  tracks: Optional[List[Dict[str, Any]]] = None,
                  frame_id: Optional[str] = None) -> Tuple[Optional[Dict[str, Any]], str]:
    """Maybe kick an async VLM reason; return (cached_draft_or_None, status).

    NEVER blocks: it submits work to a bounded executor and returns the most
    recent cached draft immediately. status is one of:
      disabled | not_triggered | throttled | triggered | cached | cached_and_triggered
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
        if not should:
            return draft, ("cached" if draft else "not_triggered")
        last = _LAST_RUN_MS.get(sid, 0)
        if sid in _INFLIGHT or (now - last) < _min_interval_ms():
            return draft, ("cached" if draft else "throttled")
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
        return draft, ("cached" if draft else "error")
    return draft, ("cached_and_triggered" if draft else "triggered")


def _run_and_cache(sid: str, req: Dict[str, Any]) -> None:
    try:
        resp = reason_sync(req)
        with _LOCK:
            _CACHE[sid] = {"response": resp, "ts": _now_ms()}
            _LAST_STATUS.update(status=resp.get("reasoner_status", "ok"), ts=_now_ms())
    except Exception as exc:  # noqa: BLE001 -- background must never crash the worker
        log.warning("vlm: background reason failed: %s", exc)
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
        elif m in ("qwen_vl", "deepseek_vl2"):
            resp = _model_reason(req, m)
        else:
            resp = ReasonResponse(reasoner_status="unavailable",
                                  error=f"unknown REASONER_MODE={m}")
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
    contract before a GPU/Qwen deployment exists.
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


# -- real model reasoner (lazy; Qwen-VL / DeepSeek-VL2) ------------------------

_ADAPTER_STATE: Dict[str, Any] = {}  # mode -> {"loaded": bool, "error": str, model/proc}


def _model_reason(req: ReasonRequest, m: str) -> ReasonResponse:
    adapter = _get_adapter(m)
    if not adapter["available"]:
        return ReasonResponse(reasoner_status="unavailable",
                              error=adapter.get("error", "model/deps unavailable"))
    image = _decode_blurred(req)
    prompt = _build_prompt(req)
    try:
        raw = adapter["generate"](prompt, image)
    except Exception as exc:  # noqa: BLE001
        return ReasonResponse(reasoner_status="error", error=f"generate: {exc}")
    data = _extract_json(raw)
    if data is None:
        return ReasonResponse(reasoner_status="error", error="model did not return valid JSON",
                              scene_summary="")
    try:
        resp = ReasonResponse(**{k: v for k, v in data.items()
                                 if k in ReasonResponse.model_fields})
        resp.reasoner_status = "ok"
        return resp
    except Exception as exc:  # noqa: BLE001
        return ReasonResponse(reasoner_status="error", error=f"schema: {exc}")


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


def _build_prompt(req: ReasonRequest) -> str:
    context = {
        "deterministic_risks": req.deterministic_risks,
        "entities": req.entities[:30],
        "scene_graph": {"relations": (req.scene_graph or {}).get("relations", [])[:30]},
        "known_hse_rules": req.known_hse_rules[:20],
        "company_profile": req.company_profile,
    }
    schema_hint = (
        '{"scene_summary": str, "risks": [{'
        '"risk_id": str, '
        '"linked_entity_id": str_or_null, '
        '"involved_track_ids": [str], '
        '"involved_detection_ids": [int], '
        '"bbox": {"x": float, "y": float, "w": float, "h": float}_or_null, '
        '"approximate_region": str_or_null, '
        '"hazard_type": str, '
        '"risk_state": "latent|active", '
        '"trigger_condition": str_or_null, '
        '"risk_level": "GREEN|YELLOW|ORANGE|RED", '
        '"severity": int, "likelihood": int, "risk_score": int, '
        '"risk_reason": str, "reason": str, '
        '"evidence": [str], "visual_evidence": [str], '
        '"recommended_controls": [{"level": str, "action": str}], '
        '"recommended_action": str, "confidence": float'
        '}], "uncertain_items": [str]}'
    )
    return (
        "You are a senior QHSE manager assisting an automated safety system. "
        "The deterministic engine has already flagged candidate risks; your job is to "
        "EXPLAIN and VERIFY them and note relational/contextual risk a box detector misses "
        "(e.g. object near an edge, person under a suspended load). Reason about object x "
        "position x height x people-exposure x dynamics. Anchor advice to the hierarchy of "
        "controls (elimination->...->ppe). You ADVISE only; you never authorize action.\n\n"
        "STRICT RULES:\n"
        "1. Report ONLY risks that are visible in the CURRENT frame.\n"
        "2. Every risk MUST include at least one of: linked_entity_id, involved_track_ids, "
        "involved_detection_ids, bbox, or approximate_region. Vague/unlinked risks must NOT "
        "be emitted -- place them in uncertain_items instead.\n"
        "3. Return an EMPTY risks list if you are uncertain about any risk.\n"
        "4. Do NOT fabricate risks. Do NOT inherit should_alert or requires_human_review "
        "from the deterministic engine.\n\n"
        "Return STRICT JSON ONLY (no prose, no code fences) matching this schema:\n"
        + schema_hint + "\nContext:\n" + json.dumps(context)[:6000]
    )


def _extract_json(raw: str) -> Optional[Dict[str, Any]]:
    if not raw:
        return None
    s = raw.strip()
    if "```" in s:  # strip code fences
        parts = s.split("```")
        for p in parts:
            p = p.strip()
            if p.startswith("{") or p.startswith("json"):
                s = p[4:].strip() if p.startswith("json") else p
                break
    a, b = s.find("{"), s.rfind("}")
    if a == -1 or b == -1 or b <= a:
        return None
    try:
        return json.loads(s[a:b + 1])
    except (ValueError, TypeError):
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
    """Import torch/transformers and load the model lazily. Returns a dict with
    available + generate(prompt, image) or an error string. Heavy + best-effort:
    on any import/load failure -> available=False so the worker degrades."""
    try:
        import torch  # noqa: F401
        from transformers import AutoProcessor  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        return {"available": False, "error": f"deps unavailable: {exc}", "generate": None}

    model_id = _model_id()
    cache_dir = os.getenv("REASONER_CACHE_DIR", "/runpod-volume/models/qwen-vl")
    device = os.getenv("REASONER_DEVICE", "cuda")
    quant = os.getenv("REASONER_QUANTIZATION", "4bit").strip().lower()

    def _load():
        import torch
        from transformers import AutoProcessor
        kwargs: Dict[str, Any] = {"cache_dir": cache_dir, "trust_remote_code": True}
        dtype = os.getenv("REASONER_DTYPE", "auto")
        if dtype != "auto":
            kwargs["torch_dtype"] = getattr(torch, dtype, "auto")
        else:
            kwargs["torch_dtype"] = "auto"
        if quant in ("4bit", "8bit"):
            try:
                from transformers import BitsAndBytesConfig
                kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=(quant == "4bit"), load_in_8bit=(quant == "8bit"))
            except Exception as exc:  # noqa: BLE001
                log.warning("vlm: quantization unavailable (%s); loading full precision", exc)
        if m == "deepseek_vl2":
            from transformers import AutoModelForCausalLM
            model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
        else:
            try:
                from transformers import Qwen2_5_VLForConditionalGeneration as _Q
            except Exception:  # noqa: BLE001 -- older transformers
                from transformers import AutoModelForImageTextToText as _Q
            model = _Q.from_pretrained(model_id, **kwargs)
        if quant not in ("4bit", "8bit") and device:
            model = model.to(device)
        processor = AutoProcessor.from_pretrained(model_id, cache_dir=cache_dir,
                                                  trust_remote_code=True)
        model.eval()
        return model, processor

    state: Dict[str, Any] = {"available": True, "error": None, "model": None,
                             "processor": None, "lock": threading.Lock()}

    def generate(prompt: str, image) -> str:
        import torch
        with state["lock"]:
            if state["model"] is None:
                state["model"], state["processor"] = _load()
        model, processor = state["model"], state["processor"]
        content = []
        if image is not None:
            content.append({"type": "image", "image": image})
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        proc_kwargs: Dict[str, Any] = {"text": [text], "return_tensors": "pt"}
        if image is not None:
            proc_kwargs["images"] = [image]
        inputs = processor(**proc_kwargs).to(model.device)
        max_new = int(os.getenv("REASONER_MAX_NEW_TOKENS", "768"))
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_new, do_sample=False)
        trimmed = out[:, inputs["input_ids"].shape[1]:]
        return processor.batch_decode(trimmed, skip_special_tokens=True)[0]

    state["generate"] = generate
    return state


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


# -- status (for /debug/state) -------------------------------------------------

def status_snapshot() -> Dict[str, Any]:
    with _LOCK:
        active = len(_CACHE)
        last = dict(_LAST_STATUS)
    return {
        "enabled": enabled(),
        "mode": mode(),
        "model_id": _model_id(),
        "trigger_level": trigger_level(),
        "min_interval_ms": _min_interval_ms(),
        "timeout_ms": _timeout_ms(),
        "max_image_side": _max_image_side(),
        "privacy_blur_enabled": privacy.blur_enabled(),
        "active_sessions": active,
        "last_status": last,
        "note": "AI draft only; requires_human_review=true; never per-frame; never blocks /detect.",
    }
