"""LangGraph state for the SafeLens agentic HSE layer.

Canonical definition. ``draft-branch/agents/langgraph_state.py`` is a thin
public re-export; edit this file only.

Pure ``typing`` so this stays importable on any host (bare VPS, worker, RunPod)
without pydantic/langgraph installed.
"""
from typing import Any, Literal, TypedDict


class HazardEvent(TypedDict, total=False):
    """One context-aware hazard derived from a detection by the reasoning engine."""
    hazard: str
    object_or_condition: str
    risk_state: Literal["latent", "active"]
    likelihood: int
    severity: int
    score: int
    matrix_band: Literal["low", "medium", "high", "critical"]
    evidence_ref: str
    immediate_action: str
    recommended_controls: list[dict[str, Any]]
    create_observation: bool
    create_audit: bool
    create_capa: bool
    create_incident: bool
    requires_human_approval: bool


class AgenticHSEState(TypedDict, total=False):
    # --- inputs ---
    company_profile: dict[str, Any]
    site_rules: dict[str, Any]
    document_context: list[dict[str, Any]]
    rag_context: list[dict[str, Any]]
    frame_context: dict[str, Any]
    zone_context: dict[str, Any]
    detections: list[dict[str, Any]]
    reasoning_url: str
    thread_id: str
    # --- working memory ---
    reasoning: dict[str, Any]
    events: list[HazardEvent]
    risk_assessment: dict[str, Any]
    audit_draft: dict[str, Any]
    training_draft: dict[str, Any]
    dataset_candidates: list[dict[str, Any]]
    vision_improvement_plan: dict[str, Any]
    # --- control / audit trail ---
    pending_approval: dict[str, Any]
    approvals: list[dict[str, Any]]
    action_log: list[dict[str, Any]]
