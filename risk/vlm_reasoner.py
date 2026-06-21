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

from pydantic import ValidationError

from . import controls, gemini_reasoner, privacy
from .gemini_reasoner import GeminiBoxDecisionResponse
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

# Short letter IDs assigned to YOLO box candidates for Gemini
_BOX_LABELS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


def _select_candidate_entities(
    entities: List[Dict[str, Any]],
    deterministic_risks: List[Dict[str, Any]],
    limit: int,
) -> List[Dict[str, Any]]:
    """Return up to `limit` candidate YOLO entities ordered by HSE priority.

    Priority (highest first):
      1. Entities already linked to deterministic risks.
      2. People / vehicles / PPE objects.
      3. Objects near image boundaries (edge ≤ 10 % of frame dimension).
      4. All remaining entities by descending detection confidence.
    """
    if not entities:
        return []

    # Collect entity IDs that have a deterministic risk link.
    linked_ids: set = set()
    for r in (deterministic_risks or []):
        if not isinstance(r, dict):
            continue
        for tid in (r.get("involved_track_ids") or []):
            linked_ids.add(str(tid))
        eid = r.get("linked_entity_id") or r.get("entity_id")
        if eid:
            linked_ids.add(str(eid))

    hse_labels = {
        "person", "worker", "forklift", "vehicle", "truck", "car",
        "helmet", "hardhat", "vest", "ppe", "pedestrian",
    }

    def _entity_id(e: Dict[str, Any], idx: int) -> str:
        return str(
            e.get("track_id") or e.get("id") or e.get("entity_id")
            or e.get("detection_id") or f"entity_{idx}"
        )

    def _near_edge(bbox: Any) -> bool:
        if not isinstance(bbox, dict):
            return False
        x, y = bbox.get("x", 0.5), bbox.get("y", 0.5)
        w, h = bbox.get("w", 0.0), bbox.get("h", 0.0)
        edge = 0.10
        return (x < edge or y < edge or (x + w) > (1.0 - edge) or (y + h) > (1.0 - edge))

    def _priority(item: tuple) -> tuple:
        idx, e = item
        eid = _entity_id(e, idx)
        label = str(
            e.get("semantic_label") or e.get("display_label")
            or e.get("label") or e.get("class_name") or ""
        ).lower()
        bbox = e.get("bbox") or {}
        conf = float(e.get("confidence") or 0.0)
        is_linked = eid in linked_ids
        is_hse = any(h in label for h in hse_labels)
        is_edge = _near_edge(bbox)
        # Lower tuple = higher priority (used for stable ascending sort + reversal)
        return (0 if is_linked else 1, 0 if is_hse else 1, 0 if is_edge else 1, -conf)

    ranked = sorted(enumerate(entities), key=_priority)
    return [e for _, e in ranked[:limit]]


def _build_box_decision_anchors(
    entities: List[Dict[str, Any]],
) -> List[Dict[str, str]]:
    """Assign short letter IDs (A, B, C…) to candidate entities.

    Returns a list of {"box_id": "A", "entity_id": "...", "label": "..."}.
    """
    anchors = []
    for i, e in enumerate(entities):
        if i >= len(_BOX_LABELS):
            break
        if not isinstance(e, dict):
            continue
        box_id = _BOX_LABELS[i]
        label = str(
            e.get("semantic_label") or e.get("display_label")
            or e.get("label") or e.get("class_name") or "object"
        ).strip() or "object"
        entity_id = str(
            e.get("track_id") or e.get("id") or e.get("entity_id")
            or e.get("detection_id") or f"entity_{i}"
        )
        anchors.append({"box_id": box_id, "entity_id": entity_id, "label": label[:50]})
    return anchors


def _render_annotated_reasoner_frame(
    image: Any,
    entities: List[Dict[str, Any]],
    anchors: List[Dict[str, str]],
) -> Any:
    """Draw YOLO box labels (A, B, C…) on a copy of `image` for Gemini reasoning.

    The annotated image is only for Gemini reasoning — it is not saved, logged,
    or returned to the app. Returns the original image unchanged on any error.
    """
    if image is None or not anchors:
        return image
    try:
        from PIL import Image as _PILImage, ImageDraw, ImageFont

        # Build entity_id → bbox mapping for quick lookup
        id_to_entity: Dict[str, Dict[str, Any]] = {}
        for e in entities:
            if not isinstance(e, dict):
                continue
            eid = str(
                e.get("track_id") or e.get("id") or e.get("entity_id")
                or e.get("detection_id") or ""
            )
            if eid:
                id_to_entity[eid] = e

        # Work on a copy so we don't mutate the original.
        annotated = image.copy()
        draw = ImageDraw.Draw(annotated)
        w_img, h_img = annotated.size
        try:
            font = ImageFont.load_default()
        except Exception:  # noqa: BLE001
            font = None

        for anchor in anchors:
            eid = anchor.get("entity_id", "")
            box_id = anchor.get("box_id", "?")
            entity = id_to_entity.get(eid)
            if not entity:
                continue
            bbox = entity.get("bbox") or {}
            if not isinstance(bbox, dict):
                continue
            x = float(bbox.get("x", 0.0))
            y = float(bbox.get("y", 0.0))
            bw = float(bbox.get("w", 0.0))
            bh = float(bbox.get("h", 0.0))
            if bw <= 0 or bh <= 0:
                continue
            # Convert normalized coords to pixel coords.
            px0 = int(x * w_img)
            py0 = int(y * h_img)
            px1 = int((x + bw) * w_img)
            py1 = int((y + bh) * h_img)
            draw.rectangle([px0, py0, px1, py1], outline="white", width=2)
            label_text = box_id
            # Draw label background + text.
            tx, ty = px0 + 2, py0 + 2
            if font:
                draw.text((tx, ty), label_text, fill="white", font=font)
            else:
                draw.text((tx, ty), label_text, fill="white")

        return annotated
    except Exception as exc:  # noqa: BLE001
        log.warning("vlm: annotated frame render failed: %s", exc)
        return image


def _gemini_data_to_reason_response(
    data: Dict[str, Any],
    req: ReasonRequest,
    anchors: List[Dict[str, str]],
) -> ReasonResponse:
    """Map GeminiBoxDecisionResponse data → ReasonResponse via anchor lookup."""
    gem = GeminiBoxDecisionResponse.model_validate(data)

    # Build box_id → anchor mapping for entity resolution.
    box_id_to_anchor = {a["box_id"]: a for a in anchors}
    valid_box_ids = set(box_id_to_anchor.keys())

    # Import risk matrix here to avoid circular at module level.
    from .risk_matrix import get_matrix
    matrix = get_matrix()

    risks: List[VlmRisk] = []
    frame_or_session = req.frame_id or req.session_id or "frame"
    for i, bd in enumerate(gem.box_updates):
        # Silently skip any box_id Gemini hallucinated outside our anchor set.
        if bd.box_id not in valid_box_ids:
            log.debug("gemini: unknown box_id=%r skipped (valid=%s)", bd.box_id, sorted(valid_box_ids))
            continue
        anchor = box_id_to_anchor[bd.box_id]
        evaluated = matrix.evaluate(bd.severity, bd.likelihood)
        level = evaluated["risk_level"]
        risk_score = evaluated["risk_score"]
        risk_state = "active" if level in ("YELLOW", "ORANGE", "RED") else "latent"
        evidence = [bd.evidence_code] if bd.evidence_code else []
        risks.append(VlmRisk(
            risk_id=f"gemini_{frame_or_session}_{i}",
            hazard_type=bd.hazard_type,
            risk_level=level,
            risk_score=risk_score,
            severity=bd.severity,
            likelihood=bd.likelihood,
            risk_state=risk_state,
            evidence=evidence,
            visual_evidence=evidence,
            involved_track_ids=[anchor["entity_id"]],
            linked_entity_id=anchor["entity_id"],
            confidence=float(bd.confidence or 0.0),
            produced_by="vlm_reasoner",
            reasoner_model=gemini_reasoner.model_id(),
            reasoner_status="ok",
            requires_human_review=True,
            should_alert=False,
        ))

    return ReasonResponse(
        reasoner_status="ok",
        reasoner_model=gemini_reasoner.model_id(),
        risks=risks,
        uncertain_items=list(gem.uncertain_box_ids or []),
        session_id=req.session_id,
        frame_id=req.frame_id,
        request_id=req.request_id,
    )


def _gemini_reason(req: ReasonRequest) -> ReasonResponse:
    adapter = _get_adapter("gemini")
    if not adapter["available"]:
        return ReasonResponse(reasoner_status="unavailable",
                              error=adapter.get("error", "Gemini adapter unavailable"))

    # Select candidate boxes (priority-ordered, capped).
    limit = gemini_reasoner.max_box_candidates()
    candidates = _select_candidate_entities(req.entities or [], req.deterministic_risks or [], limit)
    anchors = _build_box_decision_anchors(candidates)

    # Build base frame (decoded + blurred for privacy).
    base_image = _decode_blurred(req)

    # Render annotated frame with A/B/C box labels for Gemini.
    annotated_image = _render_annotated_reasoner_frame(base_image, candidates, anchors)

    prompt = _build_gemini_prompt(req, anchors)
    try:
        log.info("gemini_generate_started", extra={"session_id": req.session_id, "frame_id": req.frame_id})
        raw = adapter["generate"](prompt, annotated_image)
        log.info("gemini_generate_completed", extra={"session_id": req.session_id, "frame_id": req.frame_id})
    except Exception as exc:  # noqa: BLE001
        return ReasonResponse(reasoner_status="error", error=f"gemini generate: {exc}")
    try:
        # Adapter may return a pre-validated dict (GeminiBoxDecisionResponse.model_dump())
        # or a raw string for legacy/fallback paths.
        if isinstance(raw, dict):
            data = raw
        else:
            data = _extract_json(raw)

        if data is None:
            log.warning(
                "gemini_json_parse_failed session_id=%s frame_id=%s raw_excerpt=%r",
                req.session_id, req.frame_id, (raw or "")[:400],
            )
            return ReasonResponse(reasoner_status="json_parse_error",
                                  error="Gemini did not return valid JSON",
                                  scene_summary="", risks=[], uncertain_items=[])

        return _gemini_data_to_reason_response(data, req, anchors)
    except ValidationError as exc:
        log.warning("gemini_schema_failed", extra={"session_id": req.session_id, "frame_id": req.frame_id})
        return ReasonResponse(reasoner_status="schema_error", error=f"schema: {exc}")
    except Exception as exc:  # noqa: BLE001
        return ReasonResponse(reasoner_status="error", error=f"gemini map: {type(exc).__name__}: {exc}")


def _build_gemini_prompt(req: ReasonRequest, anchors: List[Dict[str, str]]) -> str:
    """Build the box-decision prompt.  anchors is the list of {box_id, entity_id, label}."""
    from .risk_matrix import get_matrix
    from .gemini_reasoner import GeminiBoxDecisionResponse

    # Compact anchor list for the prompt (box_id + label only; no entity_id/coords).
    prompt_anchors = [{"box_id": a["box_id"], "label": a["label"]} for a in anchors]

    # Source risk matrix bands from the configured matrix so the prompt always matches.
    matrix = get_matrix()
    max_updates = GeminiBoxDecisionResponse.model_fields["box_updates"].metadata
    # Extract the max_length from the Annotated metadata if available, else use schema default.
    try:
        import annotated_types
        max_box = next((m.max_length for m in max_updates
                        if isinstance(m, annotated_types.MaxLen)), 4)
    except Exception:  # noqa: BLE001
        max_box = 4
    band_lines = "\n".join(
        f"{b['min']}-{b['max']} {b['level']}" for b in matrix.bands
    )

    return (
        "You are an HSE box risk classifier.\n"
        "The image contains YOLO boxes labeled with short IDs such as A, B, C.\n"
        "Return ONLY valid JSON matching the schema.\n"
        "Your job:\n"
        "- choose which existing YOLO boxes should change risk color\n"
        "- assign hazard_type\n"
        "- assign severity 1-5\n"
        "- assign likelihood 1-5\n"
        "- assign confidence 0-1\n"
        "- assign evidence_code\n"
        "Rules:\n"
        "- Use only visible evidence in the current image.\n"
        "- Use only box IDs shown in detected_box_anchors.\n"
        "- Do not output coordinates.\n"
        "- Do not output bbox.\n"
        "- Do not output class IDs.\n"
        "- Do not create new boxes.\n"
        "- Do not explain in sentences.\n"
        f"- If no clear risk exists, return box_updates=[].\n"
        f"- Max {max_box} box_updates.\n"
        "- Prefer no risk over guessing.\n"
        "Risk matrix:\n"
        "risk_score = severity * likelihood\n"
        + band_lines + "\n"
        "Detected box anchors:\n"
        + json.dumps(prompt_anchors, default=str, separators=(",", ":"))
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
    if isinstance(raw, dict):
        return raw
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
        cfg = gemini_reasoner.config()
        snap.update({
            "gemini_max_output_tokens": cfg["max_output_tokens"],
            "gemini_temperature": cfg["temperature"],
            "gemini_max_detected_labels": cfg["max_detected_labels"],
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
        if "box_updates" in o or "scene_summary" in o or "risks" in o or "scene_context" in o:
            return o
    return parsed[0]
