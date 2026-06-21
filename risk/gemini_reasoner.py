"""Gemini API adapter for live scene-level vision reasoning."""
from __future__ import annotations

import importlib.util
import io
import json
import os
from typing import Any, Dict, Optional

from .reason_schema import ReasonResponse as GeminiReasonResponse


def model_id() -> str:
    return os.getenv("GEMINI_MODEL_ID", "gemini-2.5-flash").strip() or "gemini-2.5-flash"


def timeout_ms() -> int:
    try:
        return int(os.getenv("GEMINI_TIMEOUT_MS", "12000"))
    except (TypeError, ValueError):
        return 12000


def max_output_tokens() -> int:
    try:
        return max(1, int(os.getenv("GEMINI_MAX_OUTPUT_TOKENS", "512")))
    except (TypeError, ValueError):
        return 512


def temperature() -> float:
    try:
        return float(os.getenv("GEMINI_TEMPERATURE", "0"))
    except (TypeError, ValueError):
        return 0.0


def max_detected_labels() -> int:
    try:
        return max(1, int(os.getenv("GEMINI_MAX_DETECTED_LABELS", "20")))
    except (TypeError, ValueError):
        return 20


def _image_bytes(image: Any) -> Optional[bytes]:
    if image is None:
        return None
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def _text_from_response(response: Any) -> str:
    text = getattr(response, "text", None)
    if text is not None:
        return str(text)
    try:
        return response.candidates[0].content.parts[0].text
    except Exception:  # noqa: BLE001
        return str(response)


def build_adapter() -> Dict[str, Any]:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return {
            "available": False,
            "error": "GEMINI_API_KEY missing",
            "generate": None,
            "model_id": model_id(),
            "diagnostics": {},
        }
    if importlib.util.find_spec("google.genai") is None:
        return {
            "available": False,
            "error": "google-genai unavailable",
            "generate": None,
            "model_id": model_id(),
            "diagnostics": {},
        }
    from google import genai
    from google.genai import types
    try:
        client = genai.Client(api_key=api_key)
    except Exception as exc:  # noqa: BLE001
        return {
            "available": False,
            "error": f"google-genai client unavailable: {exc}",
            "generate": None,
            "model_id": model_id(),
            "diagnostics": {},
        }

    def generate(prompt: str, image: Any = None) -> str:
        data = _image_bytes(image)
        image_part = types.Part.from_bytes(data=data, mime_type="image/jpeg") if data else None
        contents = [image_part, prompt] if image_part else [prompt]
        config = {
            "response_format": {
                "text": {
                    "mime_type": "application/json",
                    "schema": GeminiReasonResponse.model_json_schema(),
                }
            },
            "temperature": temperature(),
            "max_output_tokens": max_output_tokens(),
        }
        try:
            response = client.models.generate_content(model=model_id(), contents=contents, config=config)
        except (TypeError, ValueError):
            response = client.models.generate_content(
                model=model_id(),
                contents=contents,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=GeminiReasonResponse,
                    temperature=temperature(),
                    max_output_tokens=max_output_tokens(),
                ),
            )
        text = _text_from_response(response)
        # Keep output scene-level only; schema enforcement in vlm_reasoner drops unknown fields.
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                for key in ("entities", "bbox", "class_id", "confidence"):
                    parsed.pop(key, None)
                return json.dumps(parsed, separators=(",", ":"))
        except Exception:  # noqa: BLE001
            pass
        return text

    return {
        "available": True,
        "error": None,
        "generate": generate,
        "model_id": model_id(),
        "diagnostics": {},
    }
