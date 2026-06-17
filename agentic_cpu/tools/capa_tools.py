"""
agentic_cpu/tools/capa_tools.py -- build CAPA structures (draft only).
"""

from __future__ import annotations

from typing import Any, Dict

from . import risk_tools


def build_capa(source: Dict[str, Any], hazard: Dict[str, Any]) -> Dict[str, Any]:
    """One DRAFT CAPA from a finding/incident/risk + the hazard it addresses.

    Root causes are presented as HYPOTHESES, never conclusions. Preventive actions
    follow the hierarchy of controls.
    """
    ht = str(hazard.get("hazard_type", "unknown"))
    controls = hazard.get("recommended_controls") or risk_tools.controls_for(ht)
    corrective = next((c.get("action") for c in controls
                       if c.get("level") in ("elimination", "engineering")), None)
    preventive = [c.get("action") for c in controls
                  if c.get("level") in ("engineering", "administrative", "ppe")]
    return {
        "linked_source": {
            "type": source.get("type", "risk_assessment"),
            "id": source.get("id"),
        },
        "root_cause_hypotheses": [
            f"Condition '{ht}' was allowed to develop (hypothesis -- verify on site).",
            "Existing controls may be missing or not followed (hypothesis).",
        ],
        "corrective_actions": [corrective] if corrective else [],
        "preventive_actions": preventive[:3],
        "suggested_owner_role": "site HSE lead",
        "due_date_horizon_days": 14,
        "verification_method": "Re-inspection + sign-off by a competent person.",
    }
