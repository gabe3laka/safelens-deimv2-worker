"""
shared/schemas/approvals.py -- request/response models for the approval routes
(/agent/approvals/preview, /agent/approvals/execute).

Flow: Preview -> Human approval -> Execute -> Log. Execute rejects any action
that was not explicitly approved.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class ApprovalPreviewRequest(BaseModel):
    """Ask the agent to render a preview of an action before it is approved."""

    action_type: str
    payload: Dict[str, Any] = Field(default_factory=dict)
    source: Dict[str, Any] = Field(default_factory=dict)


class ApprovalExecuteRequest(BaseModel):
    """Execute a previously previewed action. `approved` must be True and the
    `approved_by` actor recorded, or execution is rejected."""

    action_id: str
    approved: bool = False
    approved_by: Optional[str] = None


class ApprovalResult(BaseModel):
    ok: bool = True
    action_id: Optional[str] = None
    status: Optional[str] = None
    error: Optional[str] = None
    action: Optional[Dict[str, Any]] = None
