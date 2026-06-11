"""
vision_backend.py -- Backend dispatcher for the SafeLens vision worker.

Backend priority (selected via VISION_BACKEND):

  VISION_BACKEND=yolo26       (default)  -> YOLO26 boxes + poses (+ optional seg)
  VISION_BACKEND=edgecrafter  (fallback) -> EdgeCrafter ECDet-S + optional ECPose-S
  VISION_BACKEND=deimv2       (legacy)   -> DEIMv2 boxes only (debug)

Automatic fallback: when the requested backend fails to LOAD and
AUTO_BACKEND_FALLBACK=true (default), the dispatcher loads
FALLBACK_VISION_BACKEND (default edgecrafter) instead. The actually serving
backend is recorded in _BACKEND_STATE and exposed via backend_status() (used by
/debug/state) and via the InferResponse `backend` + `warning` fields, so the
app/Cloudflare contract is unchanged.

This module is the single integration point used by server.py for:
  * model loading      (load_models)
  * structured summary (model_load_summary)
  * inference          (run_inference -> InferResponse)
  * backend visibility (backend_status)
"""

from __future__ import annotations

import base64
import io
import logging
import os
import time
from typing import Any, Dict, List, Optional

from PIL import Image

from schema import BBox, Entity, Keypoint, Pose, Segment, InferResponse

log = logging.getLogger(__name__)

# Requested vs actually-serving backend (set during load_models).
_BACKEND_STATE: Dict[str, Any] = {
    "requested": None,
    "active": None,
    "fallback_active": False,
    "fallback_reason": None,
}


def active_backend() -> str:
    """Backend requested via VISION_BACKEND (default yolo26)."""
    return os.environ.get("VISION_BACKEND", "yolo26").strip().lower()


def fallback_backend() -> str:
    return os.environ.get("FALLBACK_VISION_BACKEND", "edgecrafter").strip().lower()


def auto_fallback_enabled() -> bool:
    return os.environ.get("AUTO_BACKEND_FALLBACK", "true").strip().lower() in (
        "1", "true", "yes", "on")


def serving_backend() -> str:
    """The backend that actually serves /detect (post-fallback)."""
    return _BACKEND_STATE["active"] or active_backend()


def decode_image(image_b64: str) -> "Image.Image":
    raw = base64.b64decode(image_b64)
    return Image.open(io.BytesIO(raw)).convert("RGB")


# -- Model loading ------------------------------------------------------------

def _load_backend(backend: str) -> Dict[str, Any]:
    """Load one backend's models; raises on failure."""
    if backend == "deimv2":
        from deimv2_infer import get_model
        model, _proc, device = get_model()
        return {
            "backend": "deimv2",
            "tasks_loaded": ["det"],
            "model_classes": {"det": type(model).__name__},
            "checkpoint_paths": {},
            "device": str(device),
        }
    if backend == "yolo26":
        import yolo26_loader
        return yolo26_loader.load()
    # default: edgecrafter
    import edgecrafter_loader as ec
    summary = ec.load()
    return summary


def load_models() -> Dict[str, Any]:
    """Load the configured backend, auto-falling back when enabled.

    Returns a structured summary including requested/active backend. Raises only
    when the requested backend fails AND fallback is disabled/unavailable/also
    failing, so the caller (warmup / /debug/model-load) records the traceback.
    """
    requested = active_backend()
    _BACKEND_STATE["requested"] = requested
    try:
        summary = _load_backend(requested)
        _BACKEND_STATE.update(active=requested, fallback_active=False, fallback_reason=None)
        summary.update(ok=True, requested_backend=requested,
                       active_backend=requested, fallback_active=False)
        return summary
    except Exception as exc:  # noqa: BLE001
        reason = f"{type(exc).__name__}: {exc}"
        fb = fallback_backend()
        if not auto_fallback_enabled() or fb == requested:
            raise
        log.warning("backend %s failed to load (%s); falling back to %s",
                    requested, reason, fb)
        summary = _load_backend(fb)  # raises if the fallback also fails
        _BACKEND_STATE.update(active=fb, fallback_active=True, fallback_reason=reason)
        summary.update(ok=True, requested_backend=requested, active_backend=fb,
                       fallback_active=True, fallback_reason=reason)
        return summary


def model_load_summary() -> Dict[str, Any]:
    """Attempt a model load and ALWAYS return a structured dict (never raises)."""
    result: Dict[str, Any] = {"ok": False, "backend": active_backend()}
    try:
        result.update(load_models())
        result["backend"] = serving_backend()
        result["ok"] = True
    except Exception as exc:
        import traceback
        result["ok"] = False
        result["exception_type"] = type(exc).__name__
        result["exception_message"] = str(exc)
        result["traceback"] = traceback.format_exc()
    return result


def _backend_ready(backend: str) -> bool:
    try:
        if backend == "deimv2":
            import deimv2_infer
            return deimv2_infer._model is not None
        if backend == "yolo26":
            import yolo26_loader
            return yolo26_loader.is_ready()
        import edgecrafter_loader as ec
        return ec.is_ready()
    except Exception:  # noqa: BLE001
        return False


def is_ready() -> bool:
    return _backend_ready(serving_backend())


def backend_status() -> Dict[str, Any]:
    """Non-sensitive backend snapshot for /debug/state and /debug/startup."""
    try:
        import yolo26_loader
        yolo_loaded = yolo26_loader.is_ready()
        yolo_status = yolo26_loader.status()
        yolo_tasks = yolo_status["live_tasks"]
    except Exception:  # noqa: BLE001
        yolo_loaded, yolo_tasks, yolo_status = False, [], {}
    try:
        import edgecrafter_loader as ec
        ec_loaded = ec.is_ready()
        ec_available = True
    except Exception:  # noqa: BLE001
        ec_loaded, ec_available = False, False
    return {
        "requested_backend": active_backend(),
        "active_backend": serving_backend(),
        "fallback_backend": fallback_backend(),
        "auto_backend_fallback": auto_fallback_enabled(),
        "fallback_active": _BACKEND_STATE["fallback_active"],
        "fallback_reason": _BACKEND_STATE["fallback_reason"],
        "yolo26_model_loaded": yolo_loaded,
        "yolo26_tasks": yolo_tasks,
        "yolo26": yolo_status,
        "edgecrafter_available": ec_available,
        "edgecrafter_model_loaded": ec_loaded,
    }


# -- Inference ----------------------------------------------------------------

def _yolo26_response(image_b64: str, conf: float,
                     class_filter: Optional[List[int]]) -> InferResponse:
    import yolo26_loader
    pil = decode_image(image_b64)
    img_w, img_h = pil.size
    raw = yolo26_loader.infer(pil, conf, class_filter)

    entities = [
        Entity(label=d["label"], class_id=d["class_id"], confidence=d["confidence"],
               bbox=BBox(**d["bbox"]), source=d.get("source"))
        for d in raw["entities"]
    ]
    poses = [
        Pose(label=p.get("label", "person"), confidence=p["confidence"],
             keypoints=[Keypoint(**k) for k in p["keypoints"]],
             skeleton=p.get("skeleton", []), source=p.get("source"))
        for p in raw["poses"]
    ]
    segments = [
        Segment(label=s.get("label", ""), class_id=s.get("class_id", -1),
                confidence=s.get("confidence", 0.0),
                maskContour=s.get("maskContour", []), source=s.get("source"))
        for s in raw.get("segments", [])
    ]
    return InferResponse(
        entities=entities, poses=poses, segments=segments,
        inference_ms=raw["inference_ms"],
        model=raw.get("model", "YOLO26"), backend="yolo26",
        tasks=raw.get("tasks", []), img_w=img_w, img_h=img_h,
    )


def _edgecrafter_response(image_b64: str, conf: float,
                          class_filter: Optional[List[int]]) -> InferResponse:
    import edgecrafter_loader as ec
    pil = decode_image(image_b64)
    img_w, img_h = pil.size
    raw = ec.infer(pil, conf, class_filter)

    entities = [
        Entity(
            label=d["label"], class_id=d["class_id"], confidence=d["confidence"],
            bbox=BBox(**d["bbox"]), source=d.get("source"),
        )
        for d in raw["entities"]
    ]
    poses = [
        Pose(
            label=p.get("label", "person"), confidence=p["confidence"],
            keypoints=[Keypoint(**k) for k in p["keypoints"]],
            skeleton=p.get("skeleton", []), source=p.get("source"),
        )
        for p in raw["poses"]
    ]
    return InferResponse(
        entities=entities, poses=poses,
        inference_ms=raw["inference_ms"],
        model="EdgeCrafter", backend="edgecrafter",
        tasks=list(ec._STATE.tasks), img_w=img_w, img_h=img_h,
    )


def _deimv2_response(image_b64: str, conf: float, img_size: int,
                     class_filter: Optional[List[int]]) -> InferResponse:
    from deimv2_infer import run_inference as deimv2_run
    legacy = deimv2_run(
        image_b64=image_b64, conf_threshold=conf,
        img_size=img_size, class_filter=class_filter,
    )
    # Re-tag legacy entities with source='deimv2' and fill the new fields.
    for e in legacy.entities:
        if getattr(e, "source", None) is None:
            e.source = "deimv2"
    legacy.backend = "deimv2"
    legacy.tasks = ["det"]
    if not legacy.model:
        legacy.model = "DEIMv2"
    return legacy


def run_inference(image_b64: str, conf: float = 0.25, img_size: int = 640,
                  class_filter: Optional[List[int]] = None) -> InferResponse:
    """Dispatch inference to the ACTIVE backend and return a unified response."""
    backend = serving_backend()
    t0 = time.perf_counter()
    if backend == "deimv2":
        resp = _deimv2_response(image_b64, conf, img_size, class_filter)
    elif backend == "yolo26":
        resp = _yolo26_response(image_b64, conf, class_filter)
    else:
        resp = _edgecrafter_response(image_b64, conf, class_filter)
    if not resp.inference_ms:
        resp.inference_ms = round((time.perf_counter() - t0) * 1000.0, 2)
    if _BACKEND_STATE["fallback_active"]:
        resp.fallbackUsed = True
        resp.fallbackReason = (str(_BACKEND_STATE["requested"]) + "_load_failed: " +
                               str(_BACKEND_STATE["fallback_reason"]))
        if not resp.warning:
            resp.warning = ("backend_fallback: " + str(_BACKEND_STATE["requested"]) +
                            " -> " + backend + " (" + str(_BACKEND_STATE["fallback_reason"]) + ")")
    return resp
