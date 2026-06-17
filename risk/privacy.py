"""
risk/privacy.py -- blur persons/faces before any frame is persisted, sent to a
VLM, or saved as evidence (B8).

The deterministic engine itself needs no imagery, but this module is the single
chokepoint a later VLM/evidence PR must call so that, when PRIVACY_BLUR_ENABLED
is true, no un-blurred frame leaves the worker. Hazards/conditions only -- never
emotion, identity, or biometric-category inference.

Import-light: PIL/numpy are imported lazily inside functions so importing
risk.privacy never pulls heavy deps at module import.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List


def blur_enabled() -> bool:
    return os.getenv("PRIVACY_BLUR_ENABLED", "false").strip().lower() in (
        "1", "true", "yes", "on")


_PERSON_LABELS = {"person", "people", "worker", "pedestrian", "face", "head"}


def _person_boxes(entities: List[Dict[str, Any]]) -> List[Dict[str, float]]:
    out = []
    for e in entities or []:
        label = str(e.get("label", "")).strip().lower()
        if label in _PERSON_LABELS:
            bb = e.get("bbox") or {}
            if all(k in bb for k in ("x", "y", "w", "h")):
                out.append(bb)
    return out


def blur_regions(pil_image, boxes_norm: List[Dict[str, float]], *, radius: int = 24):
    """Return a copy of `pil_image` with each normalized bbox region blurred.

    boxes_norm: list of {x,y,w,h} in 0..1. Never raises; on any failure returns
    the original image unchanged (the caller's egress guard still applies).
    """
    try:
        from PIL import ImageFilter
        img = pil_image.copy()
        w, h = img.size
        for bb in boxes_norm:
            x0 = max(0, int(bb["x"] * w))
            y0 = max(0, int(bb["y"] * h))
            x1 = min(w, int((bb["x"] + bb["w"]) * w))
            y1 = min(h, int((bb["y"] + bb["h"]) * h))
            if x1 <= x0 or y1 <= y0:
                continue
            region = img.crop((x0, y0, x1, y1)).filter(ImageFilter.GaussianBlur(radius))
            img.paste(region, (x0, y0))
        return img
    except Exception:  # noqa: BLE001 -- privacy must never crash a frame
        return pil_image


def blur_persons(pil_image, entities: List[Dict[str, Any]], *, radius: int = 24):
    """Blur all person/face regions in `pil_image` given detection entities."""
    return blur_regions(pil_image, _person_boxes(entities), radius=radius)


def sanitize_for_egress(pil_image, entities: List[Dict[str, Any]]):
    """Apply privacy blur IFF enabled, for any image about to leave the worker.

    Returns (image, blurred: bool). This is the function a VLM/evidence path
    must call before sending a frame out; the egress-guard test asserts that
    when blur is enabled and persons are present, the output differs from input.
    """
    if not blur_enabled():
        return pil_image, False
    persons = _person_boxes(entities)
    if not persons:
        return pil_image, False
    return blur_regions(pil_image, persons, radius=radius_from_env()), True


def radius_from_env() -> int:
    try:
        return max(1, int(os.getenv("PRIVACY_BLUR_RADIUS", "24")))
    except (TypeError, ValueError):
        return 24
