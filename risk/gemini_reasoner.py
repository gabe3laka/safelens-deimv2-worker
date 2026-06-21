"""
risk/gemini_reasoner.py -- Google Gemini API vision reasoner adapter.

Implements the same adapter contract as vlm_reasoner._build_adapter():
  {"available": bool, "generate": callable(prompt, image) -> str,
   "model_id": str, "diagnostics": dict, "error": str|None}

Design rules (inherited from vlm_reasoner):
  * Never log or return GEMINI_API_KEY.
  * Scene-level only: Gemini must NOT output bbox, class_id, confidence, or
    raw detector coordinates. YOLO remains the coordinate/detection truth.
  * On any import/API failure -> available=False, never raises into the caller.
  * Structured output requests JSON via response_mime_type; falls back
    gracefully if the installed SDK version uses a different config shape.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

log = logging.getLogger("safelens-vision-worker.gemini")

# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def model_id() -> str:
    return os.getenv("GEMINI_MODEL_ID", "gemini-2.5-flash").strip()


def _api_key() -> Optional[str]:
    return os.getenv("GEMINI_API_KEY", "").strip() or None


def _timeout_ms() -> int:
    try:
        return max(1000, int(os.getenv("GEMINI_TIMEOUT_MS", "12000")))
    except (TypeError, ValueError):
        return 12000


def _max_output_tokens() -> int:
    try:
        return max(64, int(os.getenv("GEMINI_MAX_OUTPUT_TOKENS", "512")))
    except (TypeError, ValueError):
        return 512


def _temperature() -> float:
    try:
        return float(os.getenv("GEMINI_TEMPERATURE", "0"))
    except (TypeError, ValueError):
        return 0.0


def _max_image_side() -> int:
    try:
        return max(64, int(os.getenv("GEMINI_MAX_IMAGE_SIDE", "512")))
    except (TypeError, ValueError):
        return 512


def _max_detected_labels() -> int:
    try:
        return max(1, int(os.getenv("GEMINI_MAX_DETECTED_LABELS", "20")))
    except (TypeError, ValueError):
        return 20


def _request_retries() -> int:
    try:
        return max(0, int(os.getenv("GEMINI_REQUEST_RETRIES", "1")))
    except (TypeError, ValueError):
        return 1


# ---------------------------------------------------------------------------
# Adapter builder
# ---------------------------------------------------------------------------

def build_adapter() -> Dict[str, Any]:
    """Build and return a Gemini generate adapter dict.

    Returns:
        {available, generate, model_id, diagnostics, error}
    Never raises.
    """
    key = _api_key()
    if not key:
        return {
            "available": False,
            "error": "GEMINI_API_KEY is not set",
            "generate": None,
            "model_id": model_id(),
            "diagnostics": {"serve_backend": "google_genai"},
        }

    try:
        from google import genai  # type: ignore
        from google.genai import types  # type: ignore
    except ImportError as exc:
        return {
            "available": False,
            "error": f"google-genai not installed: {exc}",
            "generate": None,
            "model_id": model_id(),
            "diagnostics": {"serve_backend": "google_genai"},
        }

    try:
        client = genai.Client(api_key=key)
    except Exception as exc:  # noqa: BLE001
        return {
            "available": False,
            "error": f"Gemini client init failed: {type(exc).__name__}",
            "generate": None,
            "model_id": model_id(),
            "diagnostics": {"serve_backend": "google_genai"},
        }

    mid = model_id()
    max_side = _max_image_side()
    retries = _request_retries()

    def generate(prompt: str, image: Any) -> str:
        """Call Gemini and return raw text. Raises on API error (caller handles)."""
        import io

        parts: List[Any] = []

        # Encode image to JPEG bytes if provided.
        if image is not None:
            try:
                from PIL import Image as _PILImage
                if not isinstance(image, _PILImage.Image):
                    raise TypeError(f"expected PIL Image, got {type(image)}")
                # Resize if needed (already done in _decode_blurred, but guard here too)
                if max(image.size) > max_side:
                    image.thumbnail((max_side, max_side))
                buf = io.BytesIO()
                image.save(buf, format="JPEG", quality=85)
                image_bytes = buf.getvalue()
                parts.append(
                    types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg")
                )
            except Exception as exc:  # noqa: BLE001
                log.warning("gemini: image encode failed: %s", exc)
                # Continue with text-only prompt.

        parts.append(prompt)

        last_exc: Optional[Exception] = None
        for attempt in range(retries + 1):
            try:
                response = _call_generate(client, types, mid, parts)
                return response.text or ""
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if attempt < retries:
                    log.warning("gemini: attempt %d failed: %s", attempt + 1, exc)
        raise RuntimeError(f"Gemini generate failed after {retries + 1} attempt(s): {last_exc}")

    return {
        "available": True,
        "error": None,
        "generate": generate,
        "model_id": mid,
        "diagnostics": {"serve_backend": "google_genai"},
    }


def _call_generate(client: Any, types: Any, mid: str, parts: List[Any]) -> Any:
    """Call client.models.generate_content with structured JSON config.

    Tries the documented response_format dict first; falls back to
    types.GenerateContentConfig if the SDK rejects the dict shape.
    """
    # Attempt 1: response_format dict (documented pattern for newer SDKs)
    try:
        response = client.models.generate_content(
            model=mid,
            contents=parts,
            config={
                "response_mime_type": "application/json",
                "temperature": _temperature(),
                "max_output_tokens": _max_output_tokens(),
            },
        )
        return response
    except (TypeError, ValueError, AttributeError):
        pass  # SDK may not accept plain dict; fall through to types.GenerateContentConfig

    # Attempt 2: types.GenerateContentConfig (older SDK style)
    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        temperature=_temperature(),
        max_output_tokens=_max_output_tokens(),
    )
    return client.models.generate_content(
        model=mid,
        contents=parts,
        config=config,
    )
