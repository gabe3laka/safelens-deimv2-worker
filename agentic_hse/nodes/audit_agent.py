"""Agent 4 - Audit Writing.

Converts the top hazard event into an audit-ready observation draft (NC/OFI,
objective evidence, corrective + preventive action, owner, verification).
"""
from __future__ import annotations

from typing import Any


def run_audit_agent(state: dict[str, Any]) -> dict[str, Any]:
    events = state.get("events") or []
    if not events:
        return {"action_log": (state.get("action_log") or [])
                + [{"agent": "audit", "status": "ok", "summary": "no events to audit"}]}

    top = max(events, key=lambda e: int(e.get("score", 0)))
    ra = state.get("risk_assessment") or {}
    controls = ra.get("recommended_controls") or []
    corrective = controls[0]["action"] if controls else "implement controls per hierarchy of controls"
    band = top.get("matrix_band", "low")

    draft = {
        "observation": f"{top.get('object_or_condition', 'hazard')} creating {top.get('hazard', 'a hazard')}",
        "classification": "nonconformance" if band in ("high", "critical") else "ofi",
        "objective_evidence": top.get("evidence_ref", ""),
        "checklist_question_violated": (
            (state.get("site_rules") or {}).get("checklist_question")
            or f"Are controls effective for {top.get('hazard', 'the identified hazard')}?"
        ),
        "possible_root_cause": "confirm on site (supervision / barriers / signage / training)",
        "risk_rating": band,
        "corrective_action": corrective,
        "preventive_action": "add to inspection checklist + toolbox talk",
        "responsible_role": "area supervisor",
        "due_date": ra.get("due_date"),
        "verification_method": "re-inspection + photo evidence",
        "standard_reference": ra.get("standard_reference", ""),
    }
    return {
        "audit_draft": draft,
        "action_log": (state.get("action_log") or [])
        + [{"agent": "audit", "status": "drafted", "summary": "audit observation drafted"}],
    }
