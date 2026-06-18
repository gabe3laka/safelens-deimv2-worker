"""
risk/reason_schema.py -- strict Pydantic schema for the event-driven /reason
VLM reasoning endpoint and the open-vocabulary (GroundingDINO) scanner.

Hard safety contract (enforced by these models + the engine):
  * Every VLM/scanner output is an AI DRAFT: produced_by is set, and
    requires_human_review is forced True; should_alert is forced False.
  * The deterministic risk engine remains the safety signal -- these drafts
    only ever populate `scene_risks` / candidate lists; they never create
    incidents, alerts, or official risk assessments on their own.

Import-light (pydantic only); no torch/transformers here.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from .risk_schema import Control

REASON_SCHEMA_VERSION = "reason.v1"
OPEN_VOCAB_SCHEMA_VERSION = "openvocab.v1"


# -- /reason request ----------------------------------------------------------

class ReasonRequest(BaseModel):
    """Input to POST /reason. The app sends it AFTER the deterministic engine
    produced a candidate risk -- the VLM explains/verifies, it does not detect."""

    request_id: Optional[str] = None
    session_id: Optional[str] = None
    frame_id: Optional[str] = None
    frame_b64: Optional[str] = None            # optional blurred frame/crop
    entities: List[Dict[str, Any]] = Field(default_factory=list)
    tracks: List[Dict[str, Any]] = Field(default_factory=list)
    scene_graph: Dict[str, Any] = Field(default_factory=dict)
    deterministic_risks: List[Dict[str, Any]] = Field(default_factory=list)
    risk_matrix: Dict[str, Any] = Field(default_factory=dict)
    known_hse_rules: List[str] = Field(default_factory=list)
    company_profile: Dict[str, Any] = Field(default_factory=dict)


# -- /reason response ---------------------------------------------------------

class VlmRisk(BaseModel):
    """A single AI-draft risk from the VLM. Always human-review, never alerts."""

    risk_id: str
    hazard_type: str = "unknown"
    risk_level: str = "GREEN"
    risk_score: int = 1
    severity: int = 1
    likelihood: int = 1
    risk_reason: Optional[str] = None       # primary linkable explanation
    reason: str = ""                        # alias / legacy (same content as risk_reason)
    visual_evidence: List[str] = Field(default_factory=list)
    evidence: List[str] = Field(default_factory=list)   # structured evidence list
    recommended_action: Optional[str] = None
    recommended_controls: List[Control] = Field(default_factory=list)
    # Linkability: every active scene_risk must have at least one of these.
    involved_track_ids: List[str] = Field(default_factory=list)
    involved_detection_ids: List[int] = Field(default_factory=list)
    linked_entity_id: Optional[str] = None
    bbox: Optional[Dict[str, float]] = None          # normalized x/y/w/h
    approximate_region: Optional[str] = None         # text description when no bbox
    # Provenance fields (set by engine; not trusted from VLM output)
    produced_by: str = "vlm_reasoner"
    reasoner_model: Optional[str] = None
    reasoner_status: Optional[str] = None
    # Context
    risk_state: str = "latent"                 # latent | active
    trigger_condition: Optional[str] = None
    confidence: float = 0.5
    # Forced by validators below -- a VLM draft can never self-authorize.
    requires_human_review: bool = True
    should_alert: bool = False


class ReasonResponse(BaseModel):
    """Output of POST /reason. Strict JSON; AI draft only."""

    schema_version: str = REASON_SCHEMA_VERSION
    produced_by: str = "vlm_reasoner"
    reasoner_model: Optional[str] = None
    # ok | disabled | unavailable | timeout | error | not_triggered
    reasoner_status: str = "ok"
    scene_summary: str = ""
    risks: List[VlmRisk] = Field(default_factory=list)
    uncertain_items: List[str] = Field(default_factory=list)
    requires_human_review: bool = True
    should_alert: bool = False
    request_id: Optional[str] = None
    session_id: Optional[str] = None
    frame_id: Optional[str] = None
    latency_ms: Optional[float] = None
    error: Optional[str] = None

    def enforce_draft_contract(self) -> "ReasonResponse":
        """Force the safety contract regardless of what the model emitted."""
        self.produced_by = "vlm_reasoner"
        self.requires_human_review = True
        self.should_alert = False
        for r in self.risks:
            r.requires_human_review = True
            r.should_alert = False
        return self


# -- open-vocabulary (GroundingDINO) scanner ----------------------------------

class OpenVocabCandidate(BaseModel):
    """A low-authority open-vocab detection. Candidate only; never an alert."""

    label: str
    phrase: Optional[str] = None
    confidence: float = 0.0
    bbox: Dict[str, float] = Field(default_factory=dict)   # normalized x/y/w/h


class OpenVocabResult(BaseModel):
    """Output of the GroundingDINO scanner. Always candidate_only + human-review."""

    schema_version: str = OPEN_VOCAB_SCHEMA_VERSION
    produced_by: str = "open_vocab_scanner"
    source_model: str = "GroundingDINO"
    # ok | disabled | unavailable | throttled | error
    status: str = "ok"
    candidate_only: bool = True
    requires_human_review: bool = True
    candidates: List[OpenVocabCandidate] = Field(default_factory=list)
    prompt: Optional[str] = None
    session_id: Optional[str] = None
    frame_id: Optional[str] = None
    latency_ms: Optional[float] = None
    error: Optional[str] = None

    def enforce_candidate_contract(self) -> "OpenVocabResult":
        self.produced_by = "open_vocab_scanner"
        self.candidate_only = True
        self.requires_human_review = True
        return self
