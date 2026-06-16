"""
risk/risk_schema.py -- Pydantic models for the ADDITIVE risk-aware fields.

These models define the shape of the deterministic risk block that the engine
merges into the existing /detect + /ws/vision responses. Every field is additive:
when RISK_ENGINE_ENABLED is false the block is absent and the legacy contract is
byte-for-byte unchanged.

Import-light on purpose (pydantic only; no torch/cv2) so importing risk.* can
never break server boot -- mirroring schema.py / vision_backend.py.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

SCHEMA_VERSION = "risk.v1"


class Control(BaseModel):
    """One recommended control, ordered by the hierarchy of controls."""

    level: str = Field(..., description="elimination|substitution|engineering|administrative|ppe")
    action: str


class Track(BaseModel):
    """A tracked object across frames within a single session."""

    track_id: str
    label: str
    class_id: int = -1
    confidence: float = 0.0
    bbox: Dict[str, float] = Field(default_factory=dict)  # {x,y,w,h} normalized 0..1
    centroid: Dict[str, float] = Field(default_factory=dict)  # {x,y}
    velocity: Dict[str, float] = Field(default_factory=dict)  # {vx,vy} per second
    age_frames: int = 0
    hits: int = 0
    first_seen_ms: int = 0
    last_seen_ms: int = 0


class SceneRelation(BaseModel):
    """A deterministic geometric relation between two scene nodes."""

    subject: int = Field(..., description="entity index")
    relation: str = Field(..., description="near|overlaps|above|below|left_of|right_of")
    object: int = Field(..., description="entity index")
    distance: Optional[float] = None  # normalized centroid distance for 'near'
    iou: Optional[float] = None


class RiskItem(BaseModel):
    """A single deterministic risk produced by the rule engine."""

    risk_id: str
    rule_id: str
    hazard_type: str
    risk_state: str = "active"          # active | latent
    involved_track_ids: List[str] = Field(default_factory=list)
    involved_entities: List[int] = Field(default_factory=list)
    severity: int = 1
    likelihood: int = 1
    risk_score: int = 1
    risk_level: str = "GREEN"           # GREEN | YELLOW | ORANGE | RED
    reason: str = ""
    bbox: Optional[Dict[str, float]] = None  # location for overlay (normalized)
    recommended_controls: List[Control] = Field(default_factory=list)
    recommended_action: Optional[str] = None
    standard_reference: Optional[str] = None
    confidence: float = 1.0
    should_alert: bool = False
    # Provenance: deterministic engine is the safety signal, so it does NOT
    # require human review the way VLM drafts do.
    produced_by: str = "risk_engine"
    model_version: str = "risk_engine.v1"
    requires_human_review: bool = False
    timestamp_ms: int = 0


class RiskEngineMeta(BaseModel):
    """Per-response risk-engine metadata (also surfaced in /debug/state)."""

    enabled: bool = True
    degraded: bool = False
    error: Optional[str] = None
    matrix_profile: Optional[str] = None
    matrix_version: Optional[str] = None
    tracking_enabled: bool = True
    scene_graph_enabled: bool = True
    provenance_enabled: bool = True
    privacy_blur_enabled: bool = False
    model_version: str = "risk_engine.v1"
    session_id: Optional[str] = None
    active_tracks: int = 0
    risk_count: int = 0
    highest_level: str = "GREEN"
    alerting_count: int = 0
    stage_timings_ms: Dict[str, float] = Field(default_factory=dict)


class RiskResult(BaseModel):
    """The additive risk block merged into the detection response."""

    schema_version: str = SCHEMA_VERSION
    risk_engine: RiskEngineMeta = Field(default_factory=RiskEngineMeta)
    tracks: List[Track] = Field(default_factory=list)
    scene_graph: Dict[str, Any] = Field(default_factory=dict)
    risks: List[RiskItem] = Field(default_factory=list)
    # Reserved for AI-draft items from the (later) VLM reasoner. Always present
    # so the app contract is stable; populated only by a future reasoning PR.
    scene_risks: List[Dict[str, Any]] = Field(default_factory=list)
    highest_risk_level: str = "GREEN"
