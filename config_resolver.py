"""
config_resolver.py -- Backend-specific effective inference configuration resolver.

Resolves what actual conf, img_size, iou, and max_det values should be used for
inference based on:
  1. Explicit payload values (highest priority)
  2. Backend-specific env vars
  3. Safe defaults (lowest priority)

This module ensures:
  - YOLO26 always uses YOLO26_* config, never EdgeCrafter values
  - EdgeCrafter always uses EDGECRAFTER_* config, never YOLO values
  - Payload overrides (from frontend HSE profiles) are respected
  - Fallback backend gets its own resolved config, not reused primary config
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Optional

log = logging.getLogger(__name__)


@dataclass
class EffectiveInferenceConfig:
    """Resolved effective configuration for a backend inference run."""

    backend: str
    conf: float
    img_size: int
    iou: Optional[float] = None
    max_det: Optional[int] = None
    conf_source: str = "default"
    img_size_source: str = "default"
    iou_source: str = "default"
    max_det_source: str = "default"


def resolve_effective_inference_config(
    backend: str,
    payload: Optional[dict] = None,
) -> EffectiveInferenceConfig:
    """Resolve effective config for a backend based on payload + env + defaults.

    Priority order:
      1. Explicit payload values (conf, img_size)
      2. Backend-specific env vars
      3. Backend-specific defaults

    Args:
        backend: one of "yolo26", "edgecrafter", "deimv2"
        payload: optional dict with 'conf', 'img_size' fields

    Returns:
        EffectiveInferenceConfig with resolved values and their sources
    """
    payload = payload or {}
    backend = backend.strip().lower()

    if backend == "yolo26":
        return _resolve_yolo26_config(payload)
    elif backend == "edgecrafter":
        return _resolve_edgecrafter_config(payload)
    else:  # deimv2 or other fallback
        return _resolve_default_config(payload)


def _resolve_yolo26_config(payload: dict) -> EffectiveInferenceConfig:
    """Resolve YOLO26-specific config."""
    # conf resolution
    if "conf" in payload:
        conf = float(payload["conf"])
        conf_source = "payload.conf"
    else:
        conf_env = os.getenv("YOLO26_CONF", "0.20")
        try:
            conf = float(conf_env)
            conf_source = f"env:YOLO26_CONF={conf_env}"
        except (ValueError, TypeError):
            conf = 0.20
            conf_source = "default:YOLO26_CONF_parse_failed"
            log.warning("[config] YOLO26_CONF parse failed (%s), using default 0.20", conf_env)

    # img_size resolution
    if "img_size" in payload:
        img_size = int(payload["img_size"])
        img_size_source = "payload.img_size"
    else:
        img_size_env = os.getenv("YOLO26_IMG_SIZE", "960")
        try:
            img_size = int(img_size_env)
            img_size_source = f"env:YOLO26_IMG_SIZE={img_size_env}"
        except (ValueError, TypeError):
            img_size = 960
            img_size_source = "default:YOLO26_IMG_SIZE_parse_failed"
            log.warning("[config] YOLO26_IMG_SIZE parse failed (%s), using default 960", img_size_env)

    # iou resolution (always from env for YOLO, never from payload)
    iou_env = os.getenv("YOLO26_IOU", "0.50")
    try:
        iou = float(iou_env)
        iou_source = f"env:YOLO26_IOU={iou_env}"
    except (ValueError, TypeError):
        iou = 0.50
        iou_source = "default:YOLO26_IOU_parse_failed"
        log.warning("[config] YOLO26_IOU parse failed (%s), using default 0.50", iou_env)

    # max_det resolution (always from env for YOLO, never from payload)
    max_det_env = os.getenv("YOLO26_MAX_DETECTIONS", "170")
    try:
        max_det = int(max_det_env)
        max_det_source = f"env:YOLO26_MAX_DETECTIONS={max_det_env}"
    except (ValueError, TypeError):
        max_det = 170
        max_det_source = "default:YOLO26_MAX_DETECTIONS_parse_failed"
        log.warning("[config] YOLO26_MAX_DETECTIONS parse failed (%s), using default 170", max_det_env)

    return EffectiveInferenceConfig(
        backend="yolo26",
        conf=conf,
        img_size=img_size,
        iou=iou,
        max_det=max_det,
        conf_source=conf_source,
        img_size_source=img_size_source,
        iou_source=iou_source,
        max_det_source=max_det_source,
    )


def _resolve_edgecrafter_config(payload: dict) -> EffectiveInferenceConfig:
    """Resolve EdgeCrafter-specific config."""
    # conf resolution
    if "conf" in payload:
        conf = float(payload["conf"])
        conf_source = "payload.conf"
    else:
        conf_env = os.getenv("EDGECRAFTER_CONF", "0.25")
        try:
            conf = float(conf_env)
            conf_source = f"env:EDGECRAFTER_CONF={conf_env}"
        except (ValueError, TypeError):
            conf = 0.25
            conf_source = "default:EDGECRAFTER_CONF_parse_failed"
            log.warning("[config] EDGECRAFTER_CONF parse failed (%s), using default 0.25", conf_env)

    # img_size resolution
    if "img_size" in payload:
        img_size = int(payload["img_size"])
        img_size_source = "payload.img_size"
    else:
        img_size_env = os.getenv("EDGECRAFTER_IMG_SIZE", "640")
        try:
            img_size = int(img_size_env)
            img_size_source = f"env:EDGECRAFTER_IMG_SIZE={img_size_env}"
        except (ValueError, TypeError):
            img_size = 640
            img_size_source = "default:EDGECRAFTER_IMG_SIZE_parse_failed"
            log.warning("[config] EDGECRAFTER_IMG_SIZE parse failed (%s), using default 640", img_size_env)

    # EdgeCrafter does not use iou or max_det
    return EffectiveInferenceConfig(
        backend="edgecrafter",
        conf=conf,
        img_size=img_size,
        iou=None,
        max_det=None,
        conf_source=conf_source,
        img_size_source=img_size_source,
        iou_source="n/a",
        max_det_source="n/a",
    )


def _resolve_default_config(payload: dict) -> EffectiveInferenceConfig:
    """Resolve config for legacy/default backends (conf + img_size only)."""
    # conf resolution
    if "conf" in payload:
        conf = float(payload["conf"])
        conf_source = "payload.conf"
    else:
        conf = 0.25
        conf_source = "default"

    # img_size resolution
    if "img_size" in payload:
        img_size = int(payload["img_size"])
        img_size_source = "payload.img_size"
    else:
        img_size = 640
        img_size_source = "default"

    return EffectiveInferenceConfig(
        backend="default",
        conf=conf,
        img_size=img_size,
        iou=None,
        max_det=None,
        conf_source=conf_source,
        img_size_source=img_size_source,
        iou_source="n/a",
        max_det_source="n/a",
    )


def get_effective_config_summary() -> dict:
    """Return a summary of all effective configs for all backends (for /debug/state)."""
    return {
        "primary_backend": os.getenv("VISION_BACKEND", "yolo26").strip().lower(),
        "fallback_backend": os.getenv("FALLBACK_VISION_BACKEND", "edgecrafter").strip().lower(),
        "auto_backend_fallback": os.getenv("AUTO_BACKEND_FALLBACK", "true").strip().lower()
        in ("1", "true", "yes", "on"),
        "yolo26": {
            "conf": _safe_float(os.getenv("YOLO26_CONF", "0.20")),
            "img_size": _safe_int(os.getenv("YOLO26_IMG_SIZE", "960")),
            "iou": _safe_float(os.getenv("YOLO26_IOU", "0.50")),
            "max_det": _safe_int(os.getenv("YOLO26_MAX_DETECTIONS", "170")),
            "env_conf": os.getenv("YOLO26_CONF", ""),
            "env_img_size": os.getenv("YOLO26_IMG_SIZE", ""),
            "env_iou": os.getenv("YOLO26_IOU", ""),
            "env_max_det": os.getenv("YOLO26_MAX_DETECTIONS", ""),
        },
        "edgecrafter": {
            "conf": _safe_float(os.getenv("EDGECRAFTER_CONF", "0.25")),
            "img_size": _safe_int(os.getenv("EDGECRAFTER_IMG_SIZE", "640")),
            "env_conf": os.getenv("EDGECRAFTER_CONF", ""),
            "env_img_size": os.getenv("EDGECRAFTER_IMG_SIZE", ""),
        },
    }


def _safe_float(value: str) -> Optional[float]:
    """Safely convert a string to float, return None on failure."""
    try:
        return float(value) if value else None
    except (ValueError, TypeError):
        return None


def _safe_int(value: str) -> Optional[int]:
    """Safely convert a string to int, return None on failure."""
    try:
        return int(value) if value else None
    except (ValueError, TypeError):
        return None
