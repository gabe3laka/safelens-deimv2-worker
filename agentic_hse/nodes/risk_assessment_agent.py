"""Agent 3 - Risk Assessment.

Pre-fills a risk assessment from the reasoning records + company matrix. The
draft always carries ``requires_human_approval=True`` (per spec), and the highest
scoring event is staged for the LangGraph ``interrupt()`` approval gate when its
score crosses the >=10 threshold.
"""
from __future__ import annotations

from datetime import date, timedelta
from typing import Any

from ..approval import band_for_score, build_approval_request, requires_approval


def run_risk_assessment_agent(state: dict[str, Any]) -> dict[str, Any]:
    events = state.get("events") or []
    if not events:
        return {"action_log": (state.get("action_log") or [])
                + [{"agent": "risk_assessment", "status": "ok", "summary": "no hazard events"}]}

    top = max(events, key=lambda e: int(e.get("score", 0)))
    score = int(top.get("score", 0))
    records = (state.get("reasoning") or {}).get("records") or []
    record = _matching_record(top, records)
    controls = record.get("hierarchy_of_controls_recommendation") or top.get("recommended_controls") or []
    residual_likelihood = max(1, int(top.get("likelihood") or 1) - (1 if controls else 0))
    residual_severity = max(1, int(top.get("severity") or 1) - (1 if controls else 0))
    residual_score = residual_likelihood * residual_severity
    responsible = (
        (state.get("site_rules") or {}).get("default_action_owner")
        or (state.get("company_profile") or {}).get("default_action_owner")
        or "area supervisor"
    )
    due_days = 1 if score >= 17 else 3 if score >= 10 else 7

    draft = {
        "hazard": top.get("hazard"),
        "persons_at_risk": record.get("people_exposed") or ["workers in area"],
        "likelihood": top.get("likelihood"),
        "severity": top.get("severity"),
        "score": score,
        "matrix_band": top.get("matrix_band", band_for_score(score)),
        "existing_controls": [],
        "recommended_controls": controls,
        "residual_likelihood": residual_likelihood,
        "residual_severity": residual_severity,
        "residual_score": residual_score,
        "residual_matrix_band": band_for_score(residual_score),
        "responsible_person": responsible,
        "due_date": (date.today() + timedelta(days=due_days)).isoformat(),
        "standard_reference": record.get("standard_reference", ""),
        "requires_human_approval": True,
    }

    update: dict[str, Any] = {
        "risk_assessment": draft,
        "action_log": (state.get("action_log") or [])
        + [{"agent": "risk_assessment", "status": "drafted",
            "summary": f"risk assessment drafted, top score {score}"}],
    }
    update["pending_approval"] = build_approval_request("risk_assessment", draft, score)
    return update


def _matching_record(event: dict[str, Any], records: list[dict[str, Any]]) -> dict[str, Any]:
    event_object = str(event.get("object_or_condition") or "").casefold()
    event_hazard = str(event.get("hazard") or "").casefold()
    for rec in records:
        if (
            str(rec.get("object_or_condition") or "").casefold() == event_object
            or str(rec.get("hazard") or "").casefold() == event_hazard
        ):
            return rec
    return records[0] if len(records) == 1 else {}
