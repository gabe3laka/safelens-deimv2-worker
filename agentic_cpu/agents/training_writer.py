"""
agentic_cpu/agents/training_writer.py -- draft toolbox talks + training records.
Generating an official completion record is approval-required (training_record_create).
"""

from __future__ import annotations

from typing import Any, Dict

from ..tools import training_tools, vision_tools
from . import new_action


def draft(req: Dict[str, Any]) -> Dict[str, Any]:
    dc = req.get("detection_context") or {}
    profile = req.get("company_profile") or {}
    hazards = vision_tools.extract_hazards(dc)
    talk = training_tools.build_toolbox_talk(hazards, profile)
    record = training_tools.build_completion_record_draft(topic=talk["title"])
    payload = {
        "training_type": "toolbox_talk",
        "toolbox_talk": talk,
        "completion_record_draft": record,
        "notes": req.get("notes"),
        "disclaimer": ("AI-drafted. Completion records are NOT certified until an "
                       "authorized human approves them."),
    }
    return new_action(
        "training_record_create", req,
        title=talk["title"],
        summary=(f"{len(talk['learning_objectives'])} objective(s); draft completion "
                 f"record is uncertified until approved."),
        payload=payload,
    )
