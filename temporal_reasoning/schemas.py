"""
temporal_reasoning/schemas.py -- Pydantic models for the temporal layer's
additive /detect blocks. Re-exported from shared.schemas so the GPU and CPU
layers agree on one definition (no duplicate, drifting copies).
"""

from __future__ import annotations

from shared.schemas.temporal_reasoning import (
    TEMPORAL_SCHEMA_VERSION,
    EdgeRisk,
    ReasonerStatus,
    SceneContext,
    SemanticCorrection,
    TemporalReasoningBlock,
)

__all__ = [
    "TEMPORAL_SCHEMA_VERSION",
    "EdgeRisk",
    "ReasonerStatus",
    "SceneContext",
    "SemanticCorrection",
    "TemporalReasoningBlock",
]
