"""
shared/schemas/agent_actions.py -- the CPU agent's action/draft contract.

Hard rule: the CPU agent NEVER finalizes a serious record on its own. Serious
actions are produced as DRAFTS (status="pending_approval",
requires_human_approval=True) and only become final via an explicit approved
execute call (agentic_cpu.approvals). Provenance is always attached.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import BaseModel, ConfigDict, Field

AGENT_ACTION_SCHEMA_VERSION = "agent_action.v1"

# Action types that MUST go through Preview -> Human approval -> Execute -> Log.
APPROVAL_REQUIRED_ACTIONS = frozenset({
    "incident_create",
    "incident_close",
    "capa_create",
    "risk_register_write",
    "risk_assessment_approve",
    "audit_finding_send",
    "training_record_create",
    "dataset_candidate_approve",
    "model_deployment_approve",
})


class ActionSource(BaseModel):
    """Where the draft came from (links a draft back to a vision frame)."""

    vision_session_id: Optional[str] = None
    frame_id: Optional[str] = None
    detect_response_id: Optional[str] = None


class ActionProvenance(BaseModel):
    """Model lineage for the draft. No secrets, no weights -- identifiers only."""

    model_config = ConfigDict(protected_namespaces=())

    detector_model: Optional[str] = None
    reasoner_model: Optional[str] = None
    llm_provider: str = "mock"
    llm_model: str = "mock"
    produced_by: str = "agentic_cpu"


class ActionPreview(BaseModel):
    """Human-readable summary the approver reviews before executing."""

    model_config = ConfigDict(extra="allow")

    title: str = ""
    summary: str = ""


class AgentAction(BaseModel):
    """A single agent action/draft. Serious types require human approval."""

    model_config = ConfigDict(protected_namespaces=())

    schema_version: str = AGENT_ACTION_SCHEMA_VERSION
    action_id: str
    action_type: str
    # pending_approval | approved | executed | rejected | failed
    status: str = "pending_approval"
    requires_human_approval: bool = True
    created_by: str = "agentic_cpu"
    created_at_ms: int = 0
    preview: ActionPreview = Field(default_factory=ActionPreview)
    payload: Dict[str, Any] = Field(default_factory=dict)
    source: ActionSource = Field(default_factory=ActionSource)
    provenance: ActionProvenance = Field(default_factory=ActionProvenance)
    # Set once executed (audit trail); never pre-populated.
    executed_at_ms: Optional[int] = None
    approved_by: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
