"""
shared/schemas/detection.py -- the structured detection view the CPU agent
consumes.

The CPU agent NEVER imports GPU loaders or runs inference. It receives the JSON
that /detect already produced (entities + optional risk block) and reasons over
it. These models describe that consumed shape so the agent can validate input
without importing schema.py's full vision contract or any GPU dep.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class DetectionEntity(BaseModel):
    """One detected object as seen by the agent (subset of vision schema.Entity)."""

    label: str
    class_id: int = -1
    confidence: float = 0.0
    bbox: Dict[str, float] = Field(default_factory=dict)   # {x,y,w,h} normalized
    source: Optional[str] = None
    track_id: Optional[str] = None


class DetectionContext(BaseModel):
    """A detection result forwarded to the agent (structured JSON, not pixels).

    Everything is optional so the agent degrades gracefully on partial input.
    """

    session_id: Optional[str] = None
    frame_id: Optional[str] = None
    detect_response_id: Optional[str] = None
    backend: Optional[str] = None
    entities: List[DetectionEntity] = Field(default_factory=list)
    risks: List[Dict[str, Any]] = Field(default_factory=list)
    scene_risks: List[Dict[str, Any]] = Field(default_factory=list)
    scene_context: Dict[str, Any] = Field(default_factory=dict)
    highest_risk_level: Optional[str] = None
