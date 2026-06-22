"""
risk/gemini_reasoner.py -- Google Gemini API vision reasoner adapter.

Implements the same adapter contract as vlm_reasoner._build_adapter():
  {"available": bool, "generate": callable(prompt, image) -> dict | str,
   "model_id": str, "diagnostics": dict, "error": str|None}

Design rules (inherited from vlm_reasoner):
  * Never log or return GEMINI_API_KEY.
  * Box-decision only: Gemini assigns risk to existing YOLO boxes (identified
    by short letter IDs A/B/C) and must NOT output bbox, class_id, confidence,
    or raw detector coordinates. YOLO remains the coordinate/detection truth.
  * On any import/API failure -> available=False, never raises into the caller.
  * generate() returns a validated dict (GeminiBoxDecisionResponse.model_dump())
    when Pydantic parse succeeds; falls back to raw str for _extract_json().
  * _call_generate() uses response_schema structured output; falls back
    gracefully if the installed SDK version uses a different config shape.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

log = logging.getLogger("safelens-vision-worker.gemini")

# ---------------------------------------------------------------------------
# Custom exception for Gemini 503 / service unavailable
# ---------------------------------------------------------------------------


class GeminiUnavailableError(RuntimeError):
    """Raised when Gemini returns HTTP 503 or an equivalent service-unavailable error.

    Distinct from generic RuntimeError so callers can map it to
    reasoner_status='unavailable' rather than 'error' or 'json_parse_error'.
    """
    pass


def _is_503(exc: Exception) -> bool:
    """Return True when *exc* looks like a Gemini / Google API 503 error."""
    exc_type = type(exc).__name__
    exc_str = str(exc)
    return (
        "503" in exc_str
        or "ServiceUnavailable" in exc_type
        or "UNAVAILABLE" in exc_str
        or "service_unavailable" in exc_str.lower()
    )

# ---------------------------------------------------------------------------
# Gemini structured-output schema
# (box-decision only; no bbox / class_id / detector confidence / narratives)
# ---------------------------------------------------------------------------

RiskLevel = Literal["GREEN", "YELLOW", "ORANGE", "RED"]
HazardType = Literal[
    "object_near_edge",
    "slip_trip",
    "blocked_path",
    "falling_object",
    "ppe_missing",
    "unsafe_interaction",
    "worker_near_vehicle",
    "broken_object",
    "other",
]
EvidenceCode = Literal[
    "near_edge",
    "unstable_position",
    "blocked_path_visible",
    "spill_or_wet_surface",
    "person_vehicle_proximity",
    "ppe_absent_visible",
    "broken_or_sharp_object",
    "falling_object_potential",
    "other_visible",
]


class GeminiBoxDecision(BaseModel):
    """Risk decision for a single existing YOLO box (identified by short letter ID).

    Deliberately omits bbox, class_id, and detector confidence — YOLO remains
    the coordinate/detection truth.  box_id must match one of the short IDs
    (A, B, C…) shown on the annotated frame passed to Gemini.
    """

    box_id: str = Field(
        description="Short YOLO box id shown on the annotated frame, e.g. A, B, C.",
        pattern=r"^[A-Z]$",
    )
    hazard_type: HazardType
    severity: int = Field(ge=1, le=5)
    likelihood: int = Field(ge=1, le=5)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    evidence_code: EvidenceCode = "other_visible"


class GeminiBoxDecisionResponse(BaseModel):
    """Top-level structured output schema for Gemini's HSE box risk decisions.

    Gemini returns only box_updates (which existing YOLO boxes to flag) and
    uncertain_box_ids (boxes it cannot assess confidently).  The worker maps
    box_id back to the YOLO entity/track, computes risk_score = severity *
    likelihood, and applies the risk matrix for the final color.
    """

    box_updates: List[GeminiBoxDecision] = Field(default_factory=list, max_length=4)
    uncertain_box_ids: List[str] = Field(default_factory=list, max_length=4)

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
        return max(64, int(os.getenv("GEMINI_MAX_OUTPUT_TOKENS", "1024")))
    except (TypeError, ValueError):
        return 1024


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


def _max_box_candidates() -> int:
    try:
        return max(1, int(os.getenv("GEMINI_MAX_BOX_CANDIDATES", "8")))
    except (TypeError, ValueError):
        return 8


def _request_retries() -> int:
    try:
        return max(0, int(os.getenv("GEMINI_REQUEST_RETRIES", "1")))
    except (TypeError, ValueError):
        return 1


# ---------------------------------------------------------------------------
# Adapter builder
# ---------------------------------------------------------------------------


def config() -> Dict[str, Any]:
    """Return public Gemini configuration values (safe to expose in status snapshots)."""
    return {
        "max_output_tokens": _max_output_tokens(),
        "temperature": _temperature(),
        "max_detected_labels": _max_detected_labels(),
        "max_box_candidates": _max_box_candidates(),
        "max_image_side": _max_image_side(),
        "timeout_ms": _timeout_ms(),
        "request_retries": _request_retries(),
    }


def max_detected_labels() -> int:
    """Public accessor for GEMINI_MAX_DETECTED_LABELS (number of entity labels in prompts)."""
    return _max_detected_labels()


def max_box_candidates() -> int:
    """Public accessor for GEMINI_MAX_BOX_CANDIDATES (max YOLO boxes sent to Gemini)."""
    return _max_box_candidates()


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

    def generate(prompt: str, image: Any) -> Any:
        """Call Gemini and return a validated dict or raw text on fallback.

        Returns GeminiBoxDecisionResponse.model_dump() when Pydantic validation
        succeeds so the caller can skip JSON re-parsing.  Falls back to raw
        str so _extract_json() in vlm_reasoner can still recover.  Raises on
        API error (caller handles).
        """
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
        last_is_unavailable = False
        for attempt in range(retries + 1):
            try:
                response = _call_generate(client, types, mid, parts)
                raw_text = response.text or ""
                # Prefer validated Pydantic dict; fall back to raw str.
                try:
                    return GeminiBoxDecisionResponse.model_validate_json(raw_text).model_dump()
                except Exception:  # noqa: BLE001
                    return raw_text
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if _is_503(exc):
                    last_is_unavailable = True
                if attempt < retries:
                    log.warning("gemini: attempt %d failed: %s", attempt + 1, exc)
        if last_is_unavailable:
            raise GeminiUnavailableError(
                f"Gemini unavailable (503) after {retries + 1} attempt(s): {last_exc}"
            ) from last_exc
        raise RuntimeError(f"Gemini generate failed after {retries + 1} attempt(s): {last_exc}")

    return {
        "available": True,
        "error": None,
        "generate": generate,
        "model_id": mid,
        "diagnostics": {"serve_backend": "google_genai"},
    }


def _call_generate(client: Any, types: Any, mid: str, parts: List[Any]) -> Any:
    """Call client.models.generate_content with structured JSON schema output.

    Tries the documented response_format dict first (newer SDK); falls back to
    types.GenerateContentConfig with response_schema (older SDK style).
    Both paths enforce the GeminiBoxDecisionResponse schema so Gemini is
    constrained to emit only the box-decision fields and nothing else.
    """
    schema_dict = GeminiBoxDecisionResponse.model_json_schema()

    # Attempt 1: response_format dict (documented pattern for newer SDKs)
    try:
        response = client.models.generate_content(
            model=mid,
            contents=parts,
            config={
                "response_format": {
                    "text": {
                        "mime_type": "application/json",
                        "schema": schema_dict,
                    }
                },
                "temperature": _temperature(),
                "max_output_tokens": _max_output_tokens(),
            },
        )
        return response
    except (TypeError, ValueError, AttributeError):
        pass  # SDK may not accept this dict shape; fall through

    # Attempt 2: types.GenerateContentConfig with response_schema
    try:
        cfg = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=GeminiBoxDecisionResponse,
            temperature=_temperature(),
            max_output_tokens=_max_output_tokens(),
        )
    except (TypeError, ValueError, AttributeError):
        # Older SDK may not accept a Pydantic class; pass the JSON Schema dict.
        cfg = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=schema_dict,
            temperature=_temperature(),
            max_output_tokens=_max_output_tokens(),
        )
    return client.models.generate_content(
        model=mid,
        contents=parts,
        config=cfg,
    )
