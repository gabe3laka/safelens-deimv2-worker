"""
shared/schemas/ -- Pydantic models shared by the GPU perception and CPU agent
layers. Pydantic-only (no torch/cv2/transformers), so both layers can validate
their inputs/outputs without pulling heavy deps.
"""

from __future__ import annotations

from .agent_actions import (
    ActionPreview,
    AgentAction,
    ActionProvenance,
    ActionSource,
    AGENT_ACTION_SCHEMA_VERSION,
    APPROVAL_REQUIRED_ACTIONS,
)
from .approvals import ApprovalExecuteRequest, ApprovalPreviewRequest, ApprovalResult
from .company_profile import CompanyProfile
from .detection import DetectionContext, DetectionEntity
from .risk import Control, SceneRiskDraft
from .temporal_reasoning import (
    EdgeRisk,
    ReasonerStatus,
    SceneContext,
    SemanticCorrection,
    TemporalReasoningBlock,
)

__all__ = [
    "AGENT_ACTION_SCHEMA_VERSION",
    "APPROVAL_REQUIRED_ACTIONS",
    "ActionPreview",
    "ActionProvenance",
    "ActionSource",
    "AgentAction",
    "ApprovalExecuteRequest",
    "ApprovalPreviewRequest",
    "ApprovalResult",
    "CompanyProfile",
    "Control",
    "DetectionContext",
    "DetectionEntity",
    "EdgeRisk",
    "ReasonerStatus",
    "SceneContext",
    "SceneRiskDraft",
    "SemanticCorrection",
    "TemporalReasoningBlock",
]
