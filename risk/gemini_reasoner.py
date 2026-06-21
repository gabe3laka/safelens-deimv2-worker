"""
risk/gemini_reasoner.py -- Gemini API live HSE vision reasoner (REASONER_MODE=gemini).

This is the recommended LIVE scene-reasoning backend. It replaces only the live
vision-reasoning model: YOLO stays the coordinate authority, and the Qwen-VL /
DeepSeek-VL2 transformer paths remain available as legacy/fallback modes in
risk.vlm_reasoner. Gemini NEVER creates boxes and NEVER emits detector
entities / bbox / class_id / confidence -- it only produces scene-level HSE
reasoning that flows through the existing draft contract (human-review, no alert).

Uses the current official Google GenAI SDK style:

    from google import genai
    from google.genai import types
    client = genai.Client(api_key=GEMINI_API_KEY)
    client.models.generate_content(model=..., contents=[image_part, prompt], config=...)

The SDK is imported lazily (via ``_load_genai``) so importing this module never
requires google-genai to be installed (CI / mock / qwen paths stay light), and
tests can inject a fake client.

Secrets: GEMINI_API_KEY is read from the environment only. It is never logged,
never returned in /debug/state, and never baked into the image.
"""

from __future__ import annotations

import io
import json
import logging
import os
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, ValidationError

from .reason_schema import ReasonResponse, VlmRisk

log = logging.getLogger("safelens-vision-worker.vlm")

SERVE_BACKEND = "google_genai"


# -- Gemini structured-output schema (scene-level only; never detector data) ---

class GeminiRisk(BaseModel):
    hazard_type: Literal[
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
    risk_level: Literal["GREEN", "YELLOW", "ORANGE", "RED"]
    reason: str = ""
    recommended_action: str = ""
    visual_evidence: List[str] = Field(default_factory=list, max_length=3)
    involved_track_ids: List[str] = Field(default_factory=list, max_length=3)
    linked_entity_id: Optional[str] = None
    approximate_region: Optional[str] = None
    confidence: float = 0.0


class GeminiReasonResponse(BaseModel):
    scene_summary: str = ""
    risks: List[GeminiRisk] = Field(default_factory=list, max_length=3)
    uncertain_items: List[str] = Field(default_factory=list, max_length=5)


# -- config (all env-driven; model is NOT hardcoded) --------------------------

def model_id() -> str:
    return os.getenv("GEMINI_MODEL_ID", "gemini-2.5-flash").strip()


def _timeout_ms() -> int:
    try:
        return max(1, int(os.getenv("GEMINI_TIMEOUT_MS", "12000")))
    except (TypeError, ValueError):
        return 12000


def _max_output_tokens() -> int:
    try:
        return max(1, int(os.getenv("GEMINI_MAX_OUTPUT_TOKENS", "512")))
    except (TypeError, ValueError):
        return 512


def _temperature() -> float:
    try:
        return float(os.getenv("GEMINI_TEMPERATURE", "0"))
    except (TypeError, ValueError):
        return 0.0


def max_image_side() -> int:
    try:
        return max(64, int(os.getenv("GEMINI_MAX_IMAGE_SIDE", "512")))
    except (TypeError, ValueError):
        return 512


def max_detected_labels() -> int:
    try:
        return max(1, int(os.getenv("GEMINI_MAX_DETECTED_LABELS", "20")))
    except (TypeError, ValueError):
        return 20


def _retries() -> int:
    try:
        return max(0, int(os.getenv("GEMINI_REQUEST_RETRIES", "1")))
    except (TypeError, ValueError):
        return 1


# -- SDK seam (lazy import; monkeypatched in tests) ---------------------------

def _load_genai():
    """Import the current Google GenAI SDK. Raises if not installed."""
    from google import genai
    from google.genai import types
    return genai, types


def _is_timeout_error(exc: BaseException) -> bool:
    name = type(exc).__name__.lower()
    msg = str(exc).lower()
    return ("timeout" in name or "deadline" in name
            or "timeout" in msg or "deadline" in msg)


# -- adapter (same shape risk.vlm_reasoner expects: available/generate/...) ----

def build_adapter() -> Dict[str, Any]:
    """Build the Gemini adapter. Never raises: on any problem it returns
    available=False with a clear, non-secret error string so the worker degrades
    to reasoner_status=unavailable instead of crashing."""
    diagnostics = {"vlm.serve_backend": SERVE_BACKEND, "vlm.gemini_model": model_id()}
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return {"available": False, "error": "GEMINI_API_KEY missing", "generate": None,
                "model_id": model_id(), "diagnostics": diagnostics}
    try:
        genai, types = _load_genai()
    except Exception as exc:  # noqa: BLE001
        return {"available": False, "error": f"google-genai not installed: {exc}",
                "generate": None, "model_id": model_id(), "diagnostics": diagnostics}
    try:
        client = genai.Client(api_key=api_key)
    except Exception as exc:  # noqa: BLE001
        return {"available": False, "error": f"gemini client init failed: {exc}",
                "generate": None, "model_id": model_id(), "diagnostics": diagnostics}

    def generate(prompt: str, image: Any, *, response_schema: Any = None) -> str:
        """Call Gemini and return the model's text (JSON). Reuses the same
        (prompt, image) interface as the transformer adapters so generate_json
        and the HSE reason path both work. Image is sent as inline JPEG bytes."""
        contents: List[Any] = []
        if image is not None:
            buf = io.BytesIO()
            image.save(buf, format="JPEG", quality=85)
            contents.append(types.Part.from_bytes(data=buf.getvalue(), mime_type="image/jpeg"))
        contents.append(prompt)
        cfg: Dict[str, Any] = {
            "response_mime_type": "application/json",
            "temperature": _temperature(),
            "max_output_tokens": _max_output_tokens(),
        }
        if response_schema is not None:
            cfg["response_schema"] = response_schema
        try:
            cfg["http_options"] = types.HttpOptions(timeout=_timeout_ms())
        except Exception:  # noqa: BLE001 -- older/newer SDKs: timeout still bounded upstream
            pass
        config = types.GenerateContentConfig(**cfg)
        resp = client.models.generate_content(
            model=model_id(), contents=contents, config=config)
        return getattr(resp, "text", "") or ""

    return {"available": True, "error": None, "generate": generate,
            "model_id": model_id(), "client": client, "diagnostics": diagnostics}


# -- output mapping: Gemini JSON -> ReasonResponse (draft contract enforced) ---

def _loads_lenient(raw_text: str) -> Optional[Any]:
    """Parse JSON, tolerating a stray ```json fence if the API returned one."""
    try:
        return json.loads(raw_text)
    except (ValueError, TypeError):
        pass
    s = raw_text.strip()
    if "```" in s:
        for body in s.split("```")[1:]:
            body = body.strip()
            if body.startswith("json"):
                body = body[4:].strip()
            if body.startswith("{"):
                try:
                    return json.loads(body)
                except (ValueError, TypeError):
                    continue
    return None


def to_reason_response(raw_text: Any, *, model_id_str: str) -> ReasonResponse:
    """Validate Gemini output and map it into the strict ReasonResponse draft.

    Status mapping: empty/non-JSON -> json_parse_error; valid JSON that fails the
    Gemini schema -> schema_error; valid -> ok (even with empty risks).
    """
    text = "" if raw_text is None else str(raw_text)
    if not text.strip():
        return ReasonResponse(reasoner_status="json_parse_error", reasoner_model=model_id_str,
                              error="gemini returned empty output",
                              scene_summary="", risks=[], uncertain_items=[])
    obj = _loads_lenient(text)
    if obj is None or not isinstance(obj, dict):
        return ReasonResponse(reasoner_status="json_parse_error", reasoner_model=model_id_str,
                              error="gemini did not return valid JSON",
                              scene_summary="", risks=[], uncertain_items=[])
    try:
        parsed = GeminiReasonResponse.model_validate(obj)
    except ValidationError as exc:
        return ReasonResponse(reasoner_status="schema_error", reasoner_model=model_id_str,
                              error=f"schema: {exc}", scene_summary="",
                              risks=[], uncertain_items=[])
    risks: List[VlmRisk] = []
    for i, r in enumerate(parsed.risks):
        risks.append(VlmRisk(
            risk_id=f"gemini_{i + 1}",
            hazard_type=r.hazard_type,
            risk_level=r.risk_level,
            reason=r.reason,
            risk_reason=r.reason or None,
            recommended_action=r.recommended_action or None,
            visual_evidence=list(r.visual_evidence),
            involved_track_ids=list(r.involved_track_ids),
            linked_entity_id=r.linked_entity_id,
            approximate_region=r.approximate_region,
            confidence=float(r.confidence),
        ))
    resp = ReasonResponse(
        reasoner_status="ok", reasoner_model=model_id_str,
        scene_summary=parsed.scene_summary, risks=risks,
        uncertain_items=list(parsed.uncertain_items))
    # YOLO stays the authority: a Gemini draft can never self-authorize.
    return resp.enforce_draft_contract()
