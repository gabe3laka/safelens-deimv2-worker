"""
build_schema.py -- Pydantic schema + limits for Build Mode blueprint processing.

Build Mode is a LIGHTWEIGHT, CPU-only feature, fully separate from the
EdgeCrafter /detect pipeline. It never loads a model and never touches the GPU.
These models also define the camelCase output contract the app already expects
inside `blueprint_frame`.
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


# -- Limits / TTL (in-memory MVP) ---------------------------------------------

MAX_FRAMES_PER_SESSION = 240
MAX_IMAGE_B64_CHARS = 12_000_000     # ~9 MB of base64 text
SESSION_TTL_SECONDS = 45 * 60        # 45 minutes
MAX_SESSIONS = 200                   # hard cap on concurrent in-memory sessions


class BuildError(Exception):
    """Structured Build Mode error carrying a stable code + HTTP status."""

    def __init__(self, code: str, status: int = 400):
        super().__init__(code)
        self.code = code
        self.status = status


# -- Blueprint output models (camelCase -- matches the app contract) ----------

class Point(BaseModel):
    x: float
    y: float


class Anchor(BaseModel):
    id: str
    x: float
    y: float
    label: str
    confidence: float = 1.0


class StepMarker(BaseModel):
    id: str
    label: str
    x: float
    y: float
    timestampMs: int = 0


class GestureOut(BaseModel):
    type: Optional[str] = None
    active: bool = False
    strength: Optional[float] = None


class BlueprintFrame(BaseModel):
    """The replayable per-frame blueprint returned inside `blueprint_frame`.

    Building each frame through this model also sanitizes NumPy scalar types
    into plain JSON numbers.
    """

    sessionId: str
    frameId: str
    timestampMs: int = 0
    outline: List[Point] = Field(default_factory=list)
    anchors: List[Anchor] = Field(default_factory=list)
    sparsePoints: List[Point] = Field(default_factory=list)
    handLandmarks: List[Point] = Field(default_factory=list)
    stepMarkers: List[StepMarker] = Field(default_factory=list)
    gesture: GestureOut = Field(default_factory=GestureOut)
