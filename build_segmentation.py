"""
build_segmentation.py -- optional SAM2-style segmentation for Build Mode.

This is a SAFE, OPTIONAL module. It is only consulted from /build/session/frame
when BUILD_SEGMENTATION_BACKEND=sam2, and it NEVER becomes a hard dependency:

  * SAM2 is imported lazily, only when actually requested.
  * If SAM2 (or its weights) are unavailable, segment_crop() returns ok=False
    and build_blueprint falls back to the cheap Canny/contour pipeline.
  * It never raises into the request path and never touches /detect or startup.

The contract is intentionally tiny so the rest of Build Mode does not care which
backend produced the mask:

    segment_crop(image_bgr, prompt=None, session=None) -> {
        "ok": bool,
        "mask_contour": [{"x": float, "y": float}, ...],   # normalized 0..1
        "mask_b64": Optional[str],                          # small PNG, usually None
        "mask_source": str,                                 # e.g. "sam2"
        "confidence": float,
    }

Contour output is preferred over a full mask to keep the payload small.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

log = logging.getLogger("safelens-vision-worker.build.seg")

# Lazy singleton for a loaded SAM2 predictor (only populated when used).
_SAM2_STATE: Dict[str, Any] = {"loaded": False, "predictor": None, "error": None}


def _backend() -> str:
    return os.getenv("BUILD_SEGMENTATION_BACKEND", "fallback").strip().lower()


def is_enabled() -> bool:
    """True only when SAM2 is explicitly selected as the segmentation backend."""
    return _backend() == "sam2"


def _device() -> str:
    return os.getenv("BUILD_SAM2_DEVICE", "cuda").strip().lower()


def _load_sam2():
    """Lazily load a SAM2 predictor. Returns None (and records error) on failure.

    Tries the official `sam2` package first, then the Ultralytics SAM API. This
    is intentionally tolerant: if nothing is installed we just return None and
    the caller falls back to the contour pipeline.
    """
    if _SAM2_STATE["loaded"]:
        return _SAM2_STATE["predictor"]
    _SAM2_STATE["loaded"] = True  # only attempt once per process

    ckpt = os.getenv("BUILD_SAM2_CHECKPOINT", "")
    cfg = os.getenv("BUILD_SAM2_CONFIG", "")
    try:
        # Preferred: official facebookresearch/sam2 package.
        from sam2.build_sam import build_sam2  # type: ignore
        from sam2.sam2_image_predictor import SAM2ImagePredictor  # type: ignore
        import torch  # noqa: F401

        if not ckpt or not cfg:
            raise RuntimeError("BUILD_SAM2_CHECKPOINT / BUILD_SAM2_CONFIG not set")
        model = build_sam2(cfg, ckpt, device=_device())
        predictor = SAM2ImagePredictor(model)
        _SAM2_STATE["predictor"] = ("sam2", predictor)
        log.info("[build.seg] SAM2 predictor loaded (device=%s)", _device())
        return _SAM2_STATE["predictor"]
    except Exception as exc:  # noqa: BLE001
        official_err = f"{type(exc).__name__}: {exc}"

    try:
        # Alternative: Ultralytics SAM (also ships a SAM2 variant).
        from ultralytics import SAM  # type: ignore
        weights = os.getenv("BUILD_SAM2_WEIGHTS", "sam2_b.pt")
        predictor = SAM(weights)
        _SAM2_STATE["predictor"] = ("ultralytics", predictor)
        log.info("[build.seg] Ultralytics SAM loaded (%s)", weights)
        return _SAM2_STATE["predictor"]
    except Exception as exc:  # noqa: BLE001
        _SAM2_STATE["error"] = official_err + " | ultralytics: " + f"{type(exc).__name__}: {exc}"
        log.warning("[build.seg] SAM2 unavailable, will use fallback: %s", _SAM2_STATE["error"])
        return None


def _mask_to_contour(mask, img_w: int, img_h: int) -> List[Dict[str, float]]:
    """Largest external contour of a boolean/uint8 mask, normalized to 0..1."""
    import cv2
    import numpy as np

    m = (np.asarray(mask) > 0).astype("uint8") * 255
    if m.ndim > 2:
        m = m[0]
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []
    best = max(contours, key=cv2.contourArea)
    peri = cv2.arcLength(best, True)
    approx = cv2.approxPolyDP(best, 0.01 * peri, True).reshape(-1, 2)
    return [{"x": float(max(0.0, min(1.0, px / img_w))),
             "y": float(max(0.0, min(1.0, py / img_h)))} for px, py in approx]


def segment_crop(image_bgr, prompt: Optional[Dict[str, Any]] = None,
                 session: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Segment the crop with SAM2 if enabled+available; else ok=False.

    `prompt` may carry a normalized point, e.g. {"point": [0.5, 0.5]}, used as a
    foreground click. Any failure degrades to ok=False so the caller falls back.
    """
    result = {"ok": False, "mask_contour": [], "mask_b64": None,
              "mask_source": "disabled", "confidence": 0.0}
    if not is_enabled():
        return result

    predictor = _load_sam2()
    if predictor is None:
        result["mask_source"] = "sam2-unavailable"
        return result

    try:
        import numpy as np

        kind, model = predictor
        h, w = image_bgr.shape[:2]
        point = (prompt or {}).get("point") or [0.5, 0.5]
        px, py = float(point[0]) * w, float(point[1]) * h

        if kind == "sam2":
            import cv2
            rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            model.set_image(rgb)
            masks, scores, _ = model.predict(
                point_coords=np.array([[px, py]]),
                point_labels=np.array([1]),
                multimask_output=True,
            )
            best = int(np.argmax(scores))
            mask = masks[best]
            conf = float(scores[best])
        else:  # ultralytics
            res = model(image_bgr, points=[[px, py]], labels=[1], verbose=False)
            if not res or res[0].masks is None or len(res[0].masks.data) == 0:
                result["mask_source"] = "sam2-empty"
                return result
            mask = res[0].masks.data[0].cpu().numpy()
            conf = 0.9

        contour = _mask_to_contour(mask, w, h)
        if not contour:
            result["mask_source"] = "sam2-empty"
            return result
        return {"ok": True, "mask_contour": contour, "mask_b64": None,
                "mask_source": "sam2", "confidence": conf}
    except Exception as exc:  # noqa: BLE001 -- must never break the frame path
        log.warning("[build.seg] segmentation failed, falling back: %s", exc)
        result["mask_source"] = "sam2-error"
        return result
