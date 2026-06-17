"""
temporal_reasoning/scene_context.py -- latest scene understanding.

Produces a SceneContext (scene_type / environment_type / confidence / reason).
In mock mode it derives a plausible context deterministically from the scene_hint
+ detected entities (no weights, used by tests + CPU integration). In qwen_vl /
deepseek_vl2 mode it asks the real VLM (via risk.vlm_reasoner.generate_json,
which blurs people first) and parses strict JSON, degrading to mock-style context
if the model is unavailable.

Scene context is ADVISORY perception, not a safety action.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from .session_memory import _bool_env

_INDOOR_LABELS = {"chair", "dining table", "table", "desk", "couch", "tv", "laptop",
                  "cup", "bottle", "wine glass", "bowl", "book", "potted plant",
                  "keyboard", "mouse", "cell phone", "microwave", "oven",
                  "refrigerator", "sink", "clock", "vase"}
_OUTDOOR_LABELS = {"car", "bus", "truck", "traffic light", "fire hydrant",
                   "stop sign", "bench", "bicycle", "motorcycle", "boat", "airplane"}
_HINT_TO_ENV = {
    "cafe": "indoor_public", "restaurant": "indoor_public", "shop": "indoor_public",
    "store": "indoor_public", "office": "indoor_workplace", "classroom": "indoor_public",
    "warehouse": "indoor_industrial", "kitchen": "indoor_workplace",
    "lab": "indoor_workplace", "home": "indoor_private", "house": "indoor_private",
    "construction": "outdoor_industrial",
}


def enabled() -> bool:
    return _bool_env("SCENE_CONTEXT_ENABLED", True)


def _hint(payload: Dict[str, Any]) -> Optional[str]:
    hint = (payload or {}).get("scene_hint")
    site = (payload or {}).get("site_context") or {}
    raw = site.get("environment_type") or hint
    if not raw:
        return None
    raw = str(raw).lower()
    for key in _HINT_TO_ENV:
        if key in raw:
            return key
    return raw


def mock_scene_context(entities: List[Dict[str, Any]], payload: Dict[str, Any]) -> Dict[str, Any]:
    labels = [str(e.get("label", "")).lower() for e in entities or []]
    indoor = sum(1 for x in labels if x in _INDOOR_LABELS)
    outdoor = sum(1 for x in labels if x in _OUTDOOR_LABELS)
    hint = _hint(payload)
    if hint:
        scene_type = hint
        environment_type = _HINT_TO_ENV.get(hint, "indoor_public")
        confidence = 0.82
        reason = (f"Scene hint '{hint}' corroborated by {indoor} indoor-typical "
                  f"object(s); treating apparent outdoor/vehicle detections as context noise.")
    elif indoor >= outdoor:
        scene_type = "indoor"
        environment_type = "indoor_public"
        confidence = 0.6 if indoor else 0.4
        reason = f"{indoor} indoor-typical object(s) vs {outdoor} outdoor-typical."
    else:
        scene_type = "outdoor"
        environment_type = "outdoor"
        confidence = 0.55
        reason = f"{outdoor} outdoor-typical object(s) vs {indoor} indoor-typical."
    return {
        "scene_type": scene_type,
        "environment_type": environment_type,
        "confidence": round(confidence, 3),
        "source": "vlm_reasoner",
        "reason": reason,
        "last_checked_ms": int(time.time() * 1000),
    }


def from_vlm_json(data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Parse a SceneContext from a VLM JSON payload, or None if unusable."""
    if not isinstance(data, dict):
        return None
    sc = data.get("scene_context") if isinstance(data.get("scene_context"), dict) else data
    if not sc.get("scene_type") and not sc.get("environment_type"):
        return None
    try:
        return {
            "scene_type": sc.get("scene_type"),
            "environment_type": sc.get("environment_type"),
            "confidence": float(sc.get("confidence", 0.5) or 0.5),
            "source": "vlm_reasoner",
            "reason": str(sc.get("reason", ""))[:500],
            "last_checked_ms": int(time.time() * 1000),
        }
    except Exception:  # noqa: BLE001
        return None
