"""Agent 2 - Safety Observation.

Turns raw detections into context-aware hazard events by calling the RunPod
Senior-QHSE-Manager reasoning engine for the risk-sensitive classes, then records
typed events on the state. Falls back to a conservative detector-only event when
the reasoning service is unreachable so the graph never hard-crashes.
"""
from __future__ import annotations

from typing import Any

from ..approval import band_for_score, requires_approval, risk_sensitive_detections


async def run_observation_agent(state: dict[str, Any]) -> dict[str, Any]:
    detections = state.get("detections") or []
    sensitive = risk_sensitive_detections(detections)
    reasoning_url = state.get("reasoning_url")

    records: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    for det in sensitive:
        record = await _reason(reasoning_url, det, state)
        records.append(record)
        score = int(record.get("score", 0))
        events.append({
            "event_id": f"event-{len(state.get('events') or []) + len(events) + 1}",
            "hazard": record.get("hazard", "unspecified"),
            "object_or_condition": record.get("object_or_condition", det.get("label", "")),
            "risk_state": record.get("risk_state", "latent"),
            "likelihood": record.get("likelihood"),
            "severity": record.get("severity"),
            "score": score,
            "matrix_band": record.get("matrix_band", band_for_score(score)),
            "evidence_ref": det.get("evidence_ref")
            or (state.get("frame_context") or {}).get("frame_ref", ""),
            "immediate_action": _immediate_action(record),
            "recommended_controls": record.get("hierarchy_of_controls_recommendation") or [],
            "create_observation": True,
            "create_audit": score >= 5,
            "create_capa": score >= 10,
            "create_incident": score >= 17 or record.get("risk_state") == "active",
            "requires_human_approval": bool(record.get("requires_human_approval")) or requires_approval(score),
        })

    return {
        "reasoning": {"records": records},
        "events": (state.get("events") or []) + events,
        "action_log": (state.get("action_log") or [])
        + [{"agent": "observation", "status": "ok",
            "summary": f"{len(events)} hazard event(s) from {len(detections)} detection(s)"}],
    }


async def _reason(reasoning_url: str | None, det: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    payload = {
        "detections": [det],
        "frame_ref": (state.get("frame_context") or {}).get("frame_ref"),
        "company_profile": state.get("company_profile", {}),
        "zone_context": state.get("zone_context", {}),
    }
    if reasoning_url:
        try:
            from ..reasoning_client import reason_over_hazard
            return await reason_over_hazard(reasoning_url, payload)
        except Exception as exc:  # noqa: BLE001 -- reasoning outage must not crash the graph
            return _fallback(det, note=f"reasoning_unavailable:{type(exc).__name__}")
    try:
        from ..typed_agents import reason_with_pydantic_ai

        typed = await reason_with_pydantic_ai(payload)
        if typed:
            return typed
    except Exception as exc:  # noqa: BLE001 -- optional provider failures must fail into review
        return _fallback(det, note=f"typed_reasoning_unavailable:{type(exc).__name__}")
    return _fallback(det, note="no_reasoning_url")


def _fallback(det: dict[str, Any], note: str) -> dict[str, Any]:
    """Fail-safe estimate: unknown context must pause for human review."""
    likelihood, severity = 4, 4
    score = likelihood * severity
    return {
        "hazard": det.get("label", "unspecified_hazard"),
        "object_or_condition": det.get("label", ""),
        "location_context": "unknown (detector-only fallback)",
        "is_elevated": False,
        "people_exposed": [],
        "risk_state": "active",
        "trigger_condition": "reasoning context is unavailable, so exposure cannot be ruled out",
        "likelihood": likelihood,
        "severity": severity,
        "score": score,
        "matrix_band": band_for_score(score),
        "hierarchy_of_controls_recommendation": [
            {"control_type": "engineering", "action": "isolate or barricade the suspected hazard area"},
            {"control_type": "administrative", "action": "obtain an authorized on-site assessment before work continues"},
        ],
        "reasoning": f"Fail-safe estimate ({note}). Contextual reasoning is unavailable; human review is mandatory.",
        "standard_reference": "review_required",
        "requires_human_approval": requires_approval(score),
    }


def _immediate_action(record: dict[str, Any]) -> str:
    controls = record.get("hierarchy_of_controls_recommendation") or []
    if controls:
        return str(controls[0].get("action") or "isolate the hazard and obtain supervisor review")
    return "isolate the hazard and obtain supervisor review"
