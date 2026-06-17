"""
agentic_cpu/agents/risk_assessment.py -- draft a formal risk assessment.

Consumes detection JSON + company profile, scores each hazard on the 5x5 matrix
with residual-after-controls, and returns a DRAFT (risk_assessment_approve)
requiring human approval.
"""

from __future__ import annotations

from typing import Any, Dict

from ..tools import risk_tools, vision_tools
from . import new_action


def draft(req: Dict[str, Any]) -> Dict[str, Any]:
    dc = req.get("detection_context") or {}
    profile = req.get("company_profile") or {}
    hazards = vision_tools.extract_hazards(dc)
    if not hazards and req.get("notes"):
        hazards = [{"hazard_type": "reported_condition", "severity": 2, "likelihood": 2}]

    rows = []
    max_score = 0
    for h in hazards:
        sc = risk_tools.score(h.get("severity", 2), h.get("likelihood", 2))
        residual = risk_tools.residual_after_controls(int(sc["severity"]), int(sc["likelihood"]))
        controls = h.get("recommended_controls") or risk_tools.controls_for(h.get("hazard_type"))
        max_score = max(max_score, int(sc["risk_score"]))
        rows.append({
            "hazard_type": h.get("hazard_type", "unknown"),
            "where_observed": dc.get("session_id") or "live camera",
            "initial": sc,
            "recommended_controls": controls,
            "residual": residual,
            "standard_reference": "to verify",
        })

    payload = {
        "assessment_type": "risk_assessment",
        "company": {"name": profile.get("company_name"),
                    "industry": profile.get("industry"),
                    "site_type": profile.get("site_type")},
        "rows": rows,
        "notes": req.get("notes"),
        "disclaimer": "AI-drafted; must be reviewed and approved by a competent person.",
    }
    summary = (f"{len(rows)} hazard(s) assessed; highest initial risk score "
               f"{max_score}. Review controls and approve to add to the register.")
    return new_action(
        "risk_assessment_approve", req,
        title="Risk assessment draft",
        summary=summary,
        payload=payload,
        preview_extra={"risk_score": max_score, "hazard_count": len(rows),
                       "recommended_controls": (rows[0]["recommended_controls"] if rows else [])},
    )
