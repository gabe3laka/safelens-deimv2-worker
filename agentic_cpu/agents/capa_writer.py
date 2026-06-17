"""
agentic_cpu/agents/capa_writer.py -- draft a CAPA from a finding/incident/risk.
Creating a CAPA is approval-required (capa_create).
"""

from __future__ import annotations

from typing import Any, Dict

from ..tools import capa_tools, vision_tools
from . import new_action


def draft(req: Dict[str, Any]) -> Dict[str, Any]:
    dc = req.get("detection_context") or {}
    payload_in = req.get("payload") or {}
    hazards = vision_tools.extract_hazards(dc)
    # Prefer an explicit hazard passed in (e.g. from an audit finding); else the
    # top hazard from the detection context.
    hazard = payload_in.get("hazard") or (hazards[0] if hazards else
                                          {"hazard_type": "reported_condition"})
    source = {"type": payload_in.get("source_type", "risk_assessment"),
              "id": payload_in.get("source_id")}
    capa = capa_tools.build_capa(source, hazard)
    payload = {
        "capa_type": "corrective_preventive_action",
        "capa": capa,
        "notes": req.get("notes"),
        "disclaimer": "AI-drafted CAPA; root causes are hypotheses. Approve to create.",
    }
    return new_action(
        "capa_create", req,
        title=f"CAPA draft: {str(hazard.get('hazard_type', 'hazard')).replace('_', ' ')}",
        summary=(f"{len(capa.get('corrective_actions', []))} corrective + "
                 f"{len(capa.get('preventive_actions', []))} preventive action(s) proposed."),
        payload=payload,
    )
