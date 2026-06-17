"""
agentic_cpu/schemas.py -- request/response models for the /agent/* routes.

Re-exports the shared action/approval models (one source of truth) and adds the
thin request bodies the routes accept. Pydantic-only (no GPU deps).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field

from shared.schemas import (  # re-export
    AGENT_ACTION_SCHEMA_VERSION,
    APPROVAL_REQUIRED_ACTIONS,
    ActionPreview,
    ActionProvenance,
    ActionSource,
    AgentAction,
    ApprovalExecuteRequest,
    ApprovalPreviewRequest,
    ApprovalResult,
    CompanyProfile,
    DetectionContext,
)


class AgentDraftRequest(BaseModel):
    """Generic body for the /agent/*/draft + /candidate routes.

    The CPU agent consumes STRUCTURED JSON only -- never pixels. detection_context
    is what /detect already returned (entities + risk blocks); the agent reasons
    over it, it does not run inference.
    """

    company_profile: Dict[str, Any] = Field(default_factory=dict)
    detection_context: Dict[str, Any] = Field(default_factory=dict)
    notes: Optional[str] = None
    source: Dict[str, Any] = Field(default_factory=dict)
    # optional inline payload (e.g. an audit finding to turn into a CAPA)
    payload: Dict[str, Any] = Field(default_factory=dict)


class CompanyProfileExtractRequest(BaseModel):
    """Body for /agent/company/profile/extract."""

    text: Optional[str] = None
    fields: Dict[str, Any] = Field(default_factory=dict)
    source: Dict[str, Any] = Field(default_factory=dict)


__all__ = [
    "AGENT_ACTION_SCHEMA_VERSION",
    "APPROVAL_REQUIRED_ACTIONS",
    "ActionPreview",
    "ActionProvenance",
    "ActionSource",
    "AgentAction",
    "AgentDraftRequest",
    "ApprovalExecuteRequest",
    "ApprovalPreviewRequest",
    "ApprovalResult",
    "CompanyProfile",
    "CompanyProfileExtractRequest",
    "DetectionContext",
]
