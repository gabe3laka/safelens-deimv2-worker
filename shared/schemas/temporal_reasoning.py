"""
shared/schemas/temporal_reasoning.py -- additive /detect response blocks for the
event-triggered temporal VLM perception layer.

All blocks are ADDITIVE: when TEMPORAL_REASONING_ENABLED is false they are
absent and /detect is byte-for-byte the legacy shape. Perception corrections are
advisory (requires_human_review=false); safety/compliance drafts are not emitted
here -- those flow through `scene_risks` as SceneRiskDraft (human review).
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field

from .risk import Control

TEMPORAL_SCHEMA_VERSION = "temporal.v1"


class TemporalReasoningBlock(BaseModel):
    """Top-level `temporal_reasoning` block summarising memory + trigger state."""

    enabled: bool = True
    session_id: Optional[str] = None
    memory_frames: int = 0
    active_tracks: int = 0
    triggered: bool = False
    trigger_reasons: List[str] = Field(default_factory=list)


class SceneContext(BaseModel):
    """Latest scene understanding (refreshed periodically / on mismatch)."""

    scene_type: Optional[str] = None
    environment_type: Optional[str] = None
    confidence: float = 0.0
    source: str = "vlm_reasoner"
    reason: str = ""
    last_checked_ms: int = 0


class SemanticCorrection(BaseModel):
    """A perception correction: a detector label the VLM judged wrong/contextual.

    PERCEPTION CORRECTION authority -- requires_human_review=false. It only fixes
    what the camera saw; it never creates or escalates a safety/compliance action.
    Raw detector output is preserved (raw_label) for provenance.
    """

    track_id: Optional[str] = None
    raw_label: str = ""
    corrected_label: str = ""
    correction_type: str = "false_positive"   # false_positive | relabel | suppress
    action: str = "suppress_from_hse_alerts"
    confidence: float = 0.0
    reason: str = ""
    produced_by: str = "vlm_reasoner"
    purpose: str = "perception_correction"
    authority: str = "advisory_perception"
    requires_human_review: bool = False


class EdgeRisk(BaseModel):
    """Object-near-edge temporal risk (deterministic; optional VLM validation)."""

    risk_id: str
    hazard_type: str = "object_near_edge"
    risk_state: str = "latent"
    trigger_condition: str = ""
    risk_level: str = "YELLOW"
    severity: int = 2
    likelihood: int = 2
    risk_score: int = 4
    edge_reference: str = "frame_fallback"     # surface | frame_fallback
    involved_track_ids: List[str] = Field(default_factory=list)
    visual_evidence: List[str] = Field(default_factory=list)
    recommended_controls: List[Control] = Field(default_factory=list)
    produced_by: str = "deterministic_risk_engine"
    requires_human_review: bool = False


class ReasonerStatus(BaseModel):
    """Non-blocking reasoner state attached to /detect (never makes it wait)."""

    enabled: bool = False
    mode: str = "qwen_vl"
    # idle | queued | running | ready | timeout | error | disabled
    state: str = "idle"
    last_trigger: Optional[str] = None
    result_age_ms: Optional[int] = None
    stale: bool = False
