"""
agentic_cpu/tools/training_tools.py -- toolbox-talk + training-record drafts.

Generating an OFFICIAL completion record is approval-required; these helpers only
produce DRAFT content/records.
"""

from __future__ import annotations

from typing import Any, Dict, List


def build_toolbox_talk(hazards: List[Dict[str, Any]],
                       company_profile: Dict[str, Any]) -> Dict[str, Any]:
    topics = sorted({str(h.get("hazard_type", "unknown")).replace("_", " ")
                     for h in hazards}) or ["general workplace safety"]
    return {
        "title": "Toolbox talk: " + ", ".join(topics[:3]),
        "learning_objectives": [
            f"Recognise the {t} hazard and its triggers." for t in topics[:3]
        ],
        "key_messages": [
            "Stop and report unsafe conditions.",
            "Apply controls in order: eliminate, engineer, then PPE.",
            "Keep walkways and edges clear.",
        ],
        "duration_minutes": 10,
        "audience": company_profile.get("site_type") or "all site staff",
    }


def build_completion_record_draft(topic: str, trainee_placeholder: str = "<trainee>") -> Dict[str, Any]:
    """A DRAFT completion record. It NEVER asserts a named person completed
    training -- that requires an approved execute call."""
    return {
        "topic": topic,
        "trainee": trainee_placeholder,
        "status": "draft_not_certified",
        "completed": False,
        "note": "Draft only; an authorized human must approve before this becomes official.",
        "valid_horizon_months": 12,
    }
