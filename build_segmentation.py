"""
build_segmentation.py -- optional SAM2-style segmentation for Build Mode.

SAFE, OPTIONAL module. It is only consulted from /build/session/frame when
BUILD_SEGMENTATION_BACKEND=sam2, and it NEVER becomes a hard dependency:

  * SAM2 is imported lazily, only when actually requested.
  * If SAM2 (package or weights) is unavailable, segment_crop() returns
    ok=False with mask_source="fallback-contour", and build_blueprint falls
    back to the cheap Canny/contour pipeline.
  * It never raises into the request path and never touches /detect or startup.

Stable interface (backend-agnostic so the worker doesn't care which engine
produced the mask):

    segment_crop(image_bgr, *, prompt_box=None, session=None, frame_index=0) -> {
        "ok": bool,
        "mask_source": str,                                # "sam2" | "fallback-contour" | ...
        "mask_contour": [{"x": float, "y": float}, ...],   # normalized 0..1
        "mask_b64": Optional[str],                         # small PNG, usually None
        "confidence": float,
        "error": Optional[str],
    }

Preferred backend is Ultralytics SAM/SAM2 (easiest to install + run in a RunPod
image); the official facebookresearch/sam2 package is supported as a fallback.
Contour output is preferred over a full mask to keep the payload small.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

log = logging.getLogger("safelens-vision-worker.build.seg")

# Lazy singleton for a loaded SAM2 predictor (only populated when used).
_SAM2_STATE: Dict[str, Any] = {"loaded": False, "predictor": None, "error": None}

_FALLBACK = {
    "ok": False,
    "mask_source": "fallback-contour",
    "mask_contour": [],
    "mask_b64": None,
    "confidence": 0.0,
    "error": "sam2_unavailable",
}


def _backend() -> str:
    return os.getenv("BUILD_SEGMENTATION_BACKEND", "fallback").strip().lower()


def is_enabled() -> bool:
    """True only when SAM2 is explicitly selected as the segmentation backend."""
    return _backend() == "sam2"


def _device() -> str:
    return os.getenv("BUILD_SAM2_DEVICE", "cuda").strip().lower()


def _weights_path() -> str:
    """Configured weights path; fall back to a bare name so the engine can fetch."""
    weights = os.getenv("BUILD_SAM2_WEIGHTS", "/app/models/sam2_b.pt")
    if os.path.exists(weights):
        return weights
    return os.path.basename(weights) or "sam2_b.pt"


def _load_sam2():
    """Lazily load a SAM2 predictor. Returns None (and records error) on failure.

    Tries Ultralytics SAM first (preferred for RunPod), then the official `sam2`
    package. Tolerant by design: if nothing is installed we return None and the
    caller falls back to the contour pipeline.
    """
    if _SAM2_STATE["loaded"]:
        return _SAM2_STATE["predictor"]
    _SAM2_STATE["loaded"] = True  # only attempt once per process

    errors: List[str] = []

    # Preferred: Ultralytics SAM (also ships a SAM2 variant) -- box/point prompts.
    try:
        from ultralytics import SAM  # type: ignore
        model = SAM(_weights_path())
        try:
            model.to(_device())
        except Exception:  # noqa: BLE001 -- device move is best-effort
            pass
        _SAM2_STATE["predictor"] = ("ultralytics", model)
        log.info("[build.seg] Ultralytics SAM loaded (%s, device=%s)", _weights_path(), _device())
        return _SAM2_STATE["predictor"]
    except Exception as exc:  # noqa: BLE001
        errors.append("ultralytics: " + f"{type(exc).__name__}: {exc}")

    # Fallback: official facebookresearch/sam2 package.
    try:
        from sam2.build_sam import build_sam2  # type: ignore
        from sam2.sam2_image_predictor import SAM2ImagePredictor  # type: ignore
        ckpt = os.getenv("BUILD_SAM2_CHECKPOINT", "") or _weights_path()
        cfg = os.getenv("BUILD_SAM2_CONFIG", "")
        if not cfg or not os.path.exists(ckpt):
            raise RuntimeError("BUILD_SAM2_CONFIG / weights not set for official sam2")
        model = build_sam2(cfg, ckpt, device=_device())
        _SAM2_STATE["predictor"] = ("sam2", SAM2ImagePredictor(model))
        log.info("[build.seg] official SAM2 loaded (device=%s)", _device())
        return _SAM2_STATE["predictor"]
    except Exception as exc:  # noqa: BLE001
        errors.append("sam2: " + f"{type(exc).__name__}: {exc}")

    _SAM2_STATE["error"] = " | ".join(errors)
    log.warning("[build.seg] SAM2 unavailable, using fallback contour: %s", _SAM2_STATE["error"])
    return None


def _mask_to_contour(mask, img_w: int, img_h: int) -> List[Dict[str, float]]:
    """Largest external contour of a boolean/uint8 mask, normalized to 0..1."""
    import cv2
    import numpy as np

    m = (np.asarray(mask) > 0).astype("uint8") * 255
    while m.ndim > 2:
        m = m[0]
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []
    best = max(contours, key=cv2.contourArea)
    peri = cv2.arcLength(best, True)
    approx = cv2.approxPolyDP(best, 0.01 * peri, True).reshape(-1, 2)
    return [{"x": float(max(0.0, min(1.0, px / img_w))),
             "y": float(max(0.0, min(1.0, py / img_h)))} for px, py in approx]


def segment_crop(image_bgr, *, prompt_box: Optional[List[float]] = None,
                 session: Optional[Dict[str, Any]] = None,
                 frame_index: int = 0) -> Dict[str, Any]:
    """Segment the crop with SAM2 if enabled+available; else ok=False (fallback).

    `prompt_box` is an optional [x1, y1, x2, y2] box in pixels (e.g. the Canny
    bbox). Any failure degrades to ok=False so the caller falls back -- this
    function never raises into the request path.
    """
    if not is_enabled():
        return {**_FALLBACK, "error": "sam2_disabled"}

    predictor = _load_sam2()
    if predictor is None:
        return dict(_FALLBACK)

    try:
        import numpy as np

        kind, model = predictor
        h, w = image_bgr.shape[:2]
        if prompt_box:
            x1, y1, x2, y2 = [float(v) for v in prompt_box]
        else:
            x1, y1, x2, y2 = w * 0.1, h * 0.1, w * 0.9, h * 0.9  # most of the crop

        if kind == "ultralytics":
            res = model(image_bgr, bboxes=[[x1, y1, x2, y2]], verbose=False)
            if not res or res[0].masks is None or len(res[0].masks.data) == 0:
                return {**_FALLBACK, "error": "sam2_empty"}
            mask = res[0].masks.data[0].detach().cpu().numpy()
            confidence = 0.85
        else:  # official sam2 -- box prompt
            import cv2
            rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            model.set_image(rgb)
            masks, scores, _ = model.predict(
                box=np.array([x1, y1, x2, y2]), multimask_output=False)
            mask = masks[0]
            confidence = float(scores[0]) if len(scores) else 0.8

        contour = _mask_to_contour(mask, w, h)
        if not contour:
            return {**_FALLBACK, "error": "sam2_empty"}
        return {"ok": True, "mask_source": "sam2", "mask_contour": contour,
                "mask_b64": None, "confidence": confidence, "error": None}
    except Exception as exc:  # noqa: BLE001 -- must never break the frame path
        log.warning("[build.seg] segmentation failed, falling back: %s", exc)
        return {**_FALLBACK, "error": "sam2_error: " + type(exc).__name__}
