"""Agent 1 - Company Setup."""
from __future__ import annotations

from typing import Any

DEFAULT_RISK_MATRIX = {"size": 5, "approval_threshold": 10, "halt_threshold": 17}


def run_setup_agent(state: dict[str, Any]) -> dict[str, Any]:
    profile = dict(state.get("company_profile") or {})
    site_rules = dict(state.get("site_rules") or {})
    documents = state.get("document_context") or []
    rag_context = state.get("rag_context") or []

    profile.setdefault("company", profile.pop("name", ""))
    profile.setdefault("sites", [])
    profile.setdefault("ppe_required", ["hardhat", "safety_vest"])
    profile.setdefault("high_risk_work", site_rules.get("high_risk_work", []))
    profile.setdefault("permit_rules", site_rules.get("permit_rules", {}))
    profile.setdefault("risk_matrix", DEFAULT_RISK_MATRIX)
    profile.setdefault("custom_hazards", site_rules.get("custom_hazards", []))
    profile.setdefault("inspection_frequency", site_rules.get("inspection_frequency", {}))
    profile.setdefault("training_requirements", site_rules.get("training_requirements", []))
    site_rules.setdefault("restricted_zones", [])
    site_rules.setdefault("high_risk_work", [])
    profile.setdefault("restricted_zones", site_rules["restricted_zones"])
    profile.setdefault(
        "source_documents",
        [
            str(item.get("path") or item.get("title") or item.get("document_id"))
            for item in documents + rag_context
            if item.get("path") or item.get("title") or item.get("document_id")
        ],
    )

    return {
        "company_profile": profile,
        "site_rules": site_rules,
        "action_log": (state.get("action_log") or [])
        + [{
            "agent": "setup",
            "status": "preview",
            "summary": (
                "company safety profile normalized from supplied context; "
                f"{len(documents) + len(rag_context)} document reference(s)"
            ),
        }],
    }
