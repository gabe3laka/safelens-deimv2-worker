"""
shared/schemas/risk.py -- minimal risk primitives shared across layers.

`Control` mirrors risk.risk_schema.Control intentionally (same fields) so the
shared package stays self-contained and import-light -- neither the temporal
layer nor the CPU agent needs to import the full risk engine just to describe a
recommended control.
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class Control(BaseModel):
    """One recommended control, ordered by the hierarchy of controls."""

    level: str = Field(..., description="elimination|substitution|engineering|administrative|ppe")
    action: str


class SceneRiskDraft(BaseModel):
    """A VLM-produced safety/compliance draft attached under `scene_risks`.

    Always an AI draft: requires_human_review is forced True. This is distinct
    from a perception correction (see SemanticCorrection), which does NOT need
    human approval because it only fixes a detector mislabel.
    """

    risk_id: str
    hazard_type: str = "unknown"
    risk_state: str = "latent"             # latent | active
    trigger_condition: Optional[str] = None
    risk_level: str = "GREEN"
    severity: int = 1
    likelihood: int = 1
    risk_score: int = 1
    involved_track_ids: List[str] = Field(default_factory=list)
    visual_evidence: List[str] = Field(default_factory=list)
    recommended_controls: List[Control] = Field(default_factory=list)
    reason: str = ""
    # Provenance / authority -- enforced, never trusted from the model.
    produced_by: str = "vlm_reasoner"
    purpose: str = "safety_draft"
    authority: str = "advisory_safety"
    requires_human_review: bool = True
