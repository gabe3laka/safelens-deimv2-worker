"""Typed contracts for the agentic HSE layer (Pydantic AI / pydantic v2).

``ReasoningRecord`` mirrors ``runpod_reasoning/reasoning_schema.json`` field-for-
field so the RunPod reasoning engine output and the worker-side parsing cannot
drift. Pydantic validation prevents malformed agent output from propagating
through the LangGraph pipeline.
"""
from __future__ import annotations

from datetime import date
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, model_validator

ControlType = Literal["elimination", "substitution", "engineering", "administrative", "ppe"]
RiskState = Literal["latent", "active"]
MatrixBand = Literal["low", "medium", "high", "critical"]


class ControlRecommendation(BaseModel):
    control_type: ControlType
    action: str


class CompanySafetyProfile(BaseModel):
    company: str = ""
    sites: list[dict[str, Any]] = Field(default_factory=list)
    ppe_required: list[str] = Field(default_factory=list)
    high_risk_work: list[str] = Field(default_factory=list)
    permit_rules: dict[str, Any] = Field(default_factory=dict)
    risk_matrix: dict[str, Any] = Field(default_factory=dict)
    custom_hazards: list[str] = Field(default_factory=list)
    inspection_frequency: dict[str, Any] = Field(default_factory=dict)
    training_requirements: list[str] = Field(default_factory=list)
    restricted_zones: list[str] = Field(default_factory=list)
    source_documents: list[str] = Field(default_factory=list)


class ReasoningRecord(BaseModel):
    """Senior-QHSE-Manager contextual reasoning output (relational risk)."""
    hazard: str
    object_or_condition: str
    location_context: str
    is_elevated: bool
    people_exposed: list[str] = Field(default_factory=list)
    risk_state: RiskState
    trigger_condition: str
    likelihood: int = Field(ge=1, le=5)
    severity: int = Field(ge=1, le=5)
    score: int = Field(ge=1, le=25)
    matrix_band: MatrixBand
    hierarchy_of_controls_recommendation: list[ControlRecommendation] = Field(default_factory=list)
    reasoning: str
    standard_reference: str
    requires_human_approval: bool = False

    @model_validator(mode="after")
    def _enforce_matrix_and_approval(self) -> "ReasoningRecord":
        # score = likelihood x severity; band + approval follow deterministically.
        from .approval import band_for_score, requires_approval
        self.score = self.likelihood * self.severity
        self.matrix_band = band_for_score(self.score)  # type: ignore[assignment]
        if requires_approval(self.score):
            self.requires_human_approval = True
        return self


class RiskAssessmentDraft(BaseModel):
    hazard: str
    persons_at_risk: list[str] = Field(default_factory=list)
    likelihood: int = Field(ge=1, le=5)
    severity: int = Field(ge=1, le=5)
    score: int = Field(ge=1, le=25)
    matrix_band: MatrixBand
    existing_controls: list[str] = Field(default_factory=list)
    recommended_controls: list[ControlRecommendation] = Field(default_factory=list)
    residual_likelihood: Optional[int] = None
    residual_severity: Optional[int] = None
    residual_score: Optional[int] = None
    residual_matrix_band: Optional[MatrixBand] = None
    responsible_person: str = ""
    due_date: Optional[date] = None
    standard_reference: str = ""
    requires_human_approval: bool = True  # risk assessments always need sign-off

    @model_validator(mode="after")
    def _enforce_risk_matrix(self) -> "RiskAssessmentDraft":
        from .approval import band_for_score

        self.score = self.likelihood * self.severity
        self.matrix_band = band_for_score(self.score)  # type: ignore[assignment]
        if self.residual_likelihood is not None or self.residual_severity is not None:
            if self.residual_likelihood is None or self.residual_severity is None:
                raise ValueError("residual likelihood and severity must be supplied together")
            if not 1 <= self.residual_likelihood <= 5 or not 1 <= self.residual_severity <= 5:
                raise ValueError("residual likelihood and severity must be between 1 and 5")
            self.residual_score = self.residual_likelihood * self.residual_severity
            self.residual_matrix_band = band_for_score(self.residual_score)  # type: ignore[assignment]
        self.requires_human_approval = True
        return self


class AuditObservationDraft(BaseModel):
    observation: str
    classification: Literal["nonconformance", "ofi", "observation"] = "observation"
    objective_evidence: str = ""
    checklist_question_violated: str = ""
    possible_root_cause: str = ""
    risk_rating: MatrixBand = "low"
    corrective_action: str = ""
    preventive_action: str = ""
    responsible_role: str = ""
    due_date: Optional[str] = None
    verification_method: str = ""
    standard_reference: str = ""


TrainingType = Literal[
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


class TrainingModuleDraft(BaseModel):
    type: TrainingType
    topic: str
    content: dict[str, Any] = Field(default_factory=dict)
    evidence_refs: list[str] = Field(default_factory=list)
    anonymized: bool = True


class DetectionInput(BaseModel):
    label: str
    confidence: float = Field(ge=0, le=1)
    evidence_ref: str = ""
    bbox: list[float] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class SessionStartRequest(BaseModel):
    thread_id: str | None = None
    company_profile: dict[str, Any] = Field(default_factory=dict)
    site_rules: dict[str, Any] = Field(default_factory=dict)
    document_context: list[dict[str, Any]] = Field(default_factory=list)
    rag_context: list[dict[str, Any]] = Field(default_factory=list)
    frame_context: dict[str, Any] = Field(default_factory=dict)
    zone_context: dict[str, Any] = Field(default_factory=dict)
    detections: list[DetectionInput] = Field(default_factory=list)


class ReasonRequest(BaseModel):
    detections: list[DetectionInput] = Field(min_length=1)
    frame_ref: str | None = None
    company_profile: dict[str, Any] = Field(default_factory=dict)
    zone_context: dict[str, Any] = Field(default_factory=dict)


class ApprovalDecisionRequest(BaseModel):
    thread_id: str = Field(min_length=1)
    decision: Literal["approve", "reject", "revise"]
    approver_id: str | None = None
    notes: str = ""
    revised_payload: dict[str, Any] | None = None

    @model_validator(mode="after")
    def _revision_requires_payload(self) -> "ApprovalDecisionRequest":
        if self.decision == "revise" and not self.revised_payload:
            raise ValueError("revised_payload is required when decision is revise")
        return self
