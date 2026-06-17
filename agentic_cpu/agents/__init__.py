"""
agentic_cpu/agents/ -- the CPU agents. Each produces a DRAFT (AgentAction) from
structured JSON; none finalize a record. Dispatch maps an approval-required
action_type to the agent that drafts it.
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Callable, Dict

from .. import config
from ..schemas import ActionPreview, ActionProvenance, ActionSource, AgentAction


def _now_ms() -> int:
    return int(time.time() * 1000)


def _source(req: Dict[str, Any]) -> ActionSource:
    src = (req or {}).get("source") or {}
    dc = (req or {}).get("detection_context") or {}
    return ActionSource(
        vision_session_id=src.get("vision_session_id") or dc.get("session_id"),
        frame_id=src.get("frame_id") or dc.get("frame_id"),
        detect_response_id=src.get("detect_response_id") or dc.get("detect_response_id"),
    )


def _provenance(req: Dict[str, Any]) -> ActionProvenance:
    dc = (req or {}).get("detection_context") or {}
    return ActionProvenance(
        detector_model=dc.get("model") or dc.get("backend"),
        reasoner_model=(dc.get("reasoner_status") or {}).get("mode")
        if isinstance(dc.get("reasoner_status"), dict) else None,
        llm_provider=config.llm_provider(),
        llm_model=config.llm_model(),
        produced_by="agentic_cpu",
    )


def new_action(action_type: str, req: Dict[str, Any], *, title: str, summary: str,
               payload: Dict[str, Any], preview_extra: Dict[str, Any] | None = None,
               requires_human_approval: bool = True) -> Dict[str, Any]:
    """Build a registered-ready AgentAction dict (status=pending_approval)."""
    preview = {"title": title, "summary": summary}
    if preview_extra:
        preview.update(preview_extra)
    action = AgentAction(
        action_id="act_" + uuid.uuid4().hex[:16],
        action_type=action_type,
        status="pending_approval",
        requires_human_approval=requires_human_approval,
        created_by="agentic_cpu",
        created_at_ms=_now_ms(),
        preview=ActionPreview(**preview),
        payload=payload or {},
        source=_source(req),
        provenance=_provenance(req),
    )
    return action.model_dump()


# Registry: approval-required action_type -> drafting agent.
def _registry() -> Dict[str, Callable[[Dict[str, Any]], Dict[str, Any]]]:
    from . import (audit_writer, capa_writer, risk_assessment,
                   safety_observation, training_writer, vision_improvement)
    return {
        "incident_create": safety_observation.draft,
        "risk_assessment_approve": risk_assessment.draft,
        "audit_finding_send": audit_writer.draft,
        "capa_create": capa_writer.draft,
        "training_record_create": training_writer.draft,
        "dataset_candidate_approve": vision_improvement.candidate,
    }


def dispatch(action_type: str, req: Dict[str, Any]) -> Dict[str, Any]:
    """Run the agent that drafts `action_type`. Raises ValueError if unknown."""
    fn = _registry().get(action_type)
    if fn is None:
        raise ValueError(f"unknown action_type: {action_type}")
    return fn(req)


__all__ = ["dispatch", "new_action"]
