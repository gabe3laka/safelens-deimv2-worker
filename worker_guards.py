"""
worker_guards.py -- input protection for image-bearing routes (B7).

Rejects oversized payloads and decompression-bomb / too-large images with a
structured 4xx code (never a 500 for bad user input). Header parsing is used to
read image dimensions WITHOUT decoding all pixels, so a 20000x20000 file is
rejected before it can allocate memory.

Structured error codes (consistent with the existing style):
  missing_image_b64 | invalid_base64 | decode_failure | image_too_large |
  payload_too_large
"""

from __future__ import annotations

import base64
import binascii
import io
import os
from typing import Any, Dict, Optional, Tuple


def max_body_bytes() -> int:
    try:
        return int(os.getenv("MAX_REQUEST_BYTES", str(10_000_000)))  # ~10 MB
    except (TypeError, ValueError):
        return 10_000_000


def max_megapixels() -> float:
    try:
        return float(os.getenv("MAX_IMAGE_MEGAPIXELS", "16"))
    except (TypeError, ValueError):
        return 16.0


def content_length_error(headers) -> Optional[str]:
    """Return 'payload_too_large' if Content-Length exceeds the cap, else None."""
    try:
        cl = headers.get("content-length")
        if cl is not None and int(cl) > max_body_bytes():
            return "payload_too_large"
    except (TypeError, ValueError):
        return None
    return None


def validate_image_b64(image_b64: Optional[str]) -> Tuple[bool, Optional[str], Dict[str, Any]]:
    """Validate a base64 image: presence, decodability, byte size, megapixels.

    Returns (ok, error_code, info). info carries {width,height,megapixels} on
    success. Never raises. Reads only the image header for dimensions.
    """
    if not image_b64 or not isinstance(image_b64, str):
        return False, "missing_image_b64", {}
    # base64 text length bound (4/3 expansion) before we even decode.
    if len(image_b64) > max_body_bytes() * 4 // 3 + 4:
        return False, "payload_too_large", {}
    try:
        raw = base64.b64decode(image_b64, validate=True)
    except (binascii.Error, ValueError):
        return False, "invalid_base64", {}
    if len(raw) > max_body_bytes():
        return False, "payload_too_large", {}
    try:
        from PIL import Image
        # Header-only: .size reads the declared dimensions WITHOUT decoding the
        # raster, so a decompression bomb (tiny file, huge declared size) is
        # rejected below before any pixels are allocated.
        with Image.open(io.BytesIO(raw)) as im:
            w, h = im.size
    except Exception:  # noqa: BLE001 -- truncated / unsupported / corrupt
        return False, "decode_failure", {}
    mp = (w * h) / 1_000_000.0
    if w * h > int(max_megapixels() * 1_000_000):
        return False, "image_too_large", {"width": w, "height": h, "megapixels": round(mp, 2)}
    return True, None, {"width": w, "height": h, "megapixels": round(mp, 2)}


_STATUS = {
    "missing_image_b64": 400,
    "invalid_base64": 400,
    "decode_failure": 400,
    "image_too_large": 413,
    "payload_too_large": 413,
}


def status_for(code: str) -> int:
    return _STATUS.get(code, 400)


def config() -> Dict[str, Any]:
    return {"max_request_bytes": max_body_bytes(), "max_image_megapixels": max_megapixels()}
