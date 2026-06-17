"""
agentic_cpu/agents/safety_observation.py -- draft a safety observation that, if
approved, becomes an incident (incident_create is approval-required).
"""

from __future__ import annotations

from typing import Any, Dict

from ..tools import vision_tools
from . import new_action


def draft(req: Dict[str, Any]) -> Dict[str, Any]:
    dc = req.get("detection_context") or {}
    summary_block = vision_tools.summarize_detection(dc)
    hazards = vision_tools.extract_hazards(dc)
    top_hazard = hazards[0]["hazard_type"] if hazards else "general_observation"
    payload = {
        "observation_type": "safety_observation",
        "scene_context": summary_block.get("scene_context", {}),
        "highest_risk_level": summary_block.get("highest_risk_level", "GREEN"),
        "hazards": hazards,
        "narrative": req.get("notes") or (
            f"Vision system observed {summary_block.get('entity_count', 0)} object(s); "
            f"primary concern: {top_hazard.replace('_', ' ')}."),
        "disclaimer": "AI-drafted observation; approve to raise an incident.",
    }
    return new_action(
        "incident_create", req,
        title=f"Safety observation: {top_hazard.replace('_', ' ')}",
        summary=payload["narrative"],
        payload=payload,
        preview_extra={"highest_risk_level": payload["highest_risk_level"],
                       "hazard_count": len(hazards)},
    )
