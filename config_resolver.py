"""
config_resolver.py -- per-backend effective inference config resolution.

Resolves the conf / img_size / iou / max_det actually used for one /detect
request, given the ACTIVE (serving) backend and the request payload. This keeps
the inference parameters correct per backend instead of always using one
backend's env vars:

  yolo26       -> YOLO26_CONF / YOLO26_IMG_SIZE / YOLO26_IOU / YOLO26_MAX_DETECTIONS
  edgecrafter  -> EDGECRAFTER_CONF / EDGECRAFTER_IMG_SIZE (+ optional iou/max_det env)
  deimv2       -> DEIMV2_CONF / DEIMV2_IMG_SIZE (legacy debug)

Resolution precedence for each value (so HSE profiles can tune per request):

    payload value  ->  backend env var  ->  hard default

Every value is parsed defensively: an unparseable payload value falls through
to the env var, an unparseable env var falls through to the hard default, and
the result is clamped to a safe range. The function never raises and never
reads secrets, so it is safe to call from /detect and to surface in
GET /debug/state (`effective_config`).

When AUTO_BACKEND_FALLBACK swaps the serving backend to EdgeCrafter, the caller
passes the post-fallback backend here, so EdgeCrafter's config is recomputed
from EDGECRAFTER_* and never reuses the YOLO26_* values.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

# -- Per-backend env var names + hard defaults --------------------------------
#
# bounds are (low, high) inclusive clamps applied after resolution.

_BOUNDS = {
    "conf": (0.0, 1.0),
    "img_size": (32, 4096),
    "iou": (0.0, 1.0),
    "max_det": (1, 100000),
}

_BACKENDS = {
    "yolo26": {
        "conf": ("YOLO26_CONF", 0.25, float),
        "img_size": ("YOLO26_IMG_SIZE", 640, int),
        "iou": ("YOLO26_IOU", 0.45, float),
        "max_det": ("YOLO26_MAX_DETECTIONS", 300, int),
    },
    "edgecrafter": {
        "conf": ("EDGECRAFTER_CONF", 0.25, float),
        "img_size": ("EDGECRAFTER_IMG_SIZE", 640, int),
        "iou": ("EDGECRAFTER_IOU", 0.45, float),
        "max_det": ("EDGECRAFTER_MAX_DETECTIONS", 300, int),
    },
    "deimv2": {
        "conf": ("DEIMV2_CONF", 0.35, float),
        "img_size": ("DEIMV2_IMG_SIZE", 640, int),
        "iou": ("DEIMV2_IOU", 0.45, float),
        "max_det": ("DEIMV2_MAX_DETECTIONS", 300, int),
    },
}

# Payload keys accepted for each setting (first valid wins).
_PAYLOAD_KEYS = {
    "conf": ("conf", "confidence"),
    "img_size": ("img_size", "imgsz", "imageSize"),
    "iou": ("iou", "iou_threshold"),
    "max_det": ("max_det", "max_detections", "maxDet"),
}


def normalize_backend(active_backend: Optional[str]) -> str:
    """Map a backend string to a known key, defaulting to yolo26."""
    backend = (active_backend or "").strip().lower()
    return backend if backend in _BACKENDS else "yolo26"


def _coerce(value: Any, caster) -> Optional[float]:
    """Cast to int/float, returning None for missing/empty/unparseable values."""
    if value is None:
        return None
    if isinstance(value, bool):  # bool is an int subclass -- reject explicitly
        return None
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None
    try:
        # int() chokes on "640.0"; route ints through float() first.
        return int(float(value)) if caster is int else float(value)
    except (TypeError, ValueError):
        return None


def _resolve_one(payload: Dict[str, Any], setting: str, env_name: str,
                 default, caster) -> Any:
    """payload -> env -> default, then clamp to the setting's safe bounds."""
    value = None
    for key in _PAYLOAD_KEYS[setting]:
        value = _coerce(payload.get(key), caster)
        if value is not None:
            break
    if value is None:
        value = _coerce(os.getenv(env_name), caster)
    if value is None:
        value = default
    lo, hi = _BOUNDS[setting]
    value = max(lo, min(hi, value))
    return int(value) if caster is int else float(value)


def resolve_effective_inference_config(active_backend: Optional[str],
                                       payload: Optional[Dict[str, Any]] = None
                                       ) -> Dict[str, Any]:
    """Return {backend, conf, img_size, iou, max_det} for the serving backend.

    `active_backend` should be the POST-fallback serving backend so that, when
    AUTO_BACKEND_FALLBACK swaps to EdgeCrafter, EdgeCrafter's own env vars are
    used (never the YOLO26_* values). Never raises.
    """
    backend = normalize_backend(active_backend)
    payload = payload if isinstance(payload, dict) else {}
    spec = _BACKENDS[backend]
    cfg: Dict[str, Any] = {"backend": backend}
    for setting, (env_name, default, caster) in spec.items():
        cfg[setting] = _resolve_one(payload, setting, env_name, default, caster)
    return cfg
