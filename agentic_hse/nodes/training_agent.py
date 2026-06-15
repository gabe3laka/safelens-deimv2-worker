"""Agent 6 - Training.

Drafts a toolbox-talk seed from the top hazard event using anonymized references
only. Real training-module rendering uses ``training-materials/`` templates +
incident history; this node produces the structured seed the renderer consumes.
"""
from __future__ import annotations

from typing import Any


def run_training_agent(state: dict[str, Any]) -> dict[str, Any]:
    events = state.get("events") or []
    if not events:
        return {"action_log": (state.get("action_log") or [])
                + [{"agent": "training", "status": "ok", "summary": "no events for training"}]}

    top = max(events, key=lambda e: int(e.get("score", 0)))
    topic = top.get("hazard", "site hazard")
    controls = (state.get("risk_assessment") or {}).get("recommended_controls", [])
    evidence = [top.get("evidence_ref")] if top.get("evidence_ref") else []
    common = {
        "hazard": topic,
        "why_it_matters": f"Observed {top.get('object_or_condition', '')} at band {top.get('matrix_band')}",
        "controls": controls,
    }
    module_types = [
        "toolbox_talk",
        "micro_learning",
        "quiz",
        "worker_briefing",
        "supervisor_briefing",
        "training_record",
        "before_after_explanation",
        "method_statement_summary",
        "refresher_training",
    ]
    modules = [
        {
            "type": module_type,
            "topic": topic,
            "content": {**common, "template_status": "preview"},
            "evidence_refs": evidence,
            "anonymized": True,
        }
        for module_type in module_types
    ]
    draft = {"topic": topic, "modules": modules, "anonymized": True}
    return {
        "training_draft": draft,
        "action_log": (state.get("action_log") or [])
        + [{
            "agent": "training",
            "status": "drafted",
            "summary": f"{len(modules)} anonymized training module preview(s) drafted",
        }],
    }
