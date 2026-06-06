"""
vision_backend.py -- Backend dispatcher for the SafeLens vision worker.

Selects the active detection/pose backend via the VISION_BACKEND env var:

  VISION_BACKEND=edgecrafter  (default)  -> EdgeCrafter ECDet-S + optional ECPose-S
  VISION_BACKEND=deimv2       (fallback) -> legacy DEIMv2 boxes only

This module is the single integration point used by server.py for:
  * model loading      (load_models)
  * structured summary (model_load_summary)
  * inference          (run_inference -> InferResponse)

Keeping the dispatcher separate preserves the legacy DEIMv2 code path
(deimv2_infer.py / official_deimv2_loader.py) untouched and available.
"""

from __future__ import annotations

import base64
import io
import logging
import os
import time
from typing import Any, Dict, List, Optional

from PIL import Image

from schema import BBox, Entity, Keypoint, Pose, InferResponse

log = logging.getLogger(__name__)


def active_backend() -> str:
    """Resolve the active backend name from VISION_BACKEND (default edgecrafter)."""
    return os.environ.get("VISION_BACKEND", "edgecrafter").strip().lower()


def decode_image(image_b64: str) -> "Image.Image":
    raw = base64.b64decode(image_b64)
    return Image.open(io.BytesIO(raw)).convert("RGB")


# -- Model loading ------------------------------------------------------------

def load_models() -> Dict[str, Any]:
    """Load the configured backend's models. Returns a structured summary.

    Raises on failure so the caller (warmup / /debug/model-load) records the
    full traceback.
    """
    backend = active_backend()
    if backend == "deimv2":
        from deimv2_infer import get_model
        model, _proc, device = get_model()
        return {
            "ok": True,
            "backend": "deimv2",
            "tasks_loaded": ["det"],
            "model_classes": {"det": type(model).__name__},
            "checkpoint_paths": {},
            "device": str(device),
        }

    # default: edgecrafter
    import edgecrafter_loader as ec
    summary = ec.load()
    summary["ok"] = True
    return summary


def model_load_summary() -> Dict[str, Any]:
    """Attempt a model load and ALWAYS return a structured dict (never raises).

    Used by POST /debug/model-load. On failure returns ok=false plus a
    structured exception payload.
    """
    backend = active_backend()
    result: Dict[str, Any] = {"ok": False, "backend": backend}
    try:
        result.update(load_models())
        result["ok"] = True
    except Exception as exc:
        import traceback
        result["ok"] = False
        result["exception_type"] = type(exc).__name__
        result["exception_message"] = str(exc)
        result["traceback"] = traceback.format_exc()
    return result


def is_ready() -> bool:
    backend = active_backend()
    if backend == "deimv2":
        try:
            import deimv2_infer
            return deimv2_infer._model is not None
        except Exception:
            return False
    try:
        import edgecrafter_loader as ec
        return ec.is_ready()
    except Exception:
        return False


# -- Inference ----------------------------------------------------------------

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
    """Dispatch inference to the active backend and return a unified response."""
    backend = active_backend()
    t0 = time.perf_counter()
    if backend == "deimv2":
        resp = _deimv2_response(image_b64, conf, img_size, class_filter)
    else:
        resp = _edgecrafter_response(image_b64, conf, class_filter)
    if not resp.inference_ms:
        resp.inference_ms = round((time.perf_counter() - t0) * 1000.0, 2)
    return resp
