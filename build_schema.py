"""
build_schema.py -- Pydantic schema + limits for Build Mode blueprint processing.

Build Mode is a LIGHTWEIGHT, CPU-only feature, fully separate from the
EdgeCrafter /detect pipeline. It never loads a model and never touches the GPU.
These models also define the camelCase output contract the app already expects
inside `blueprint_frame`.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

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


class BlueprintNote(BaseModel):
    """A single rule-based AI note placed at a normalized point on the crop."""

    id: str
    type: str  # instruction | safety | quality | observation | next-step | intent
    text: str
    x: float
    y: float
    timestampMs: int = 0
    confidence: Optional[float] = None


class PlanStep(BaseModel):
    """A single suggested step for workflowMode == 'plan'."""

    id: str
    title: str
    instruction: str
    x: Optional[float] = None
    y: Optional[float] = None
    status: str = "pending"  # pending | active | completed | skipped
    safetyNote: Optional[str] = None
    qualityCheck: Optional[str] = None


class CropEntity(BaseModel):
    """A YOLO26 detection on the SELECTED CROP (normalized to crop 0..1)."""

    label: str
    class_id: int
    confidence: float
    bbox: Dict[str, float] = Field(default_factory=dict)  # {x, y, w, h}
    source: Optional[str] = None


class CropSegment(BaseModel):
    """A YOLO26 instance segment on the selected crop (normalized contour)."""

    label: str = ""
    class_id: int = -1
    confidence: float = 0.0
    maskContour: List[Point] = Field(default_factory=list)
    source: Optional[str] = None


class VirtualBlueprintPoint(BaseModel):
    """A rule-based (or reasoning-enriched) virtual blueprint point.

    role in: anchor | alignment-point | target-position | connection-point
             | inspection-point | warning-point
    x/y normalized 0..1 in selected-crop coords; z optional pseudo-depth 0..1.
    """

    id: str
    role: str
    x: float
    y: float
    z: Optional[float] = None
    label: Optional[str] = None
    instruction: Optional[str] = None
    linkedStepId: Optional[str] = None


class DepthPoint(BaseModel):
    """A sparse pseudo-depth sample (x/y crop 0..1, z relative depth 0..1)."""

    x: float
    y: float
    z: float
    confidence: Optional[float] = None


class PlanContext(BaseModel):
    """Compact context summary the app/DeepSeek can reason over."""

    selectedLabel: Optional[str] = None
    objectCount: Optional[int] = None
    hasMultipleParts: Optional[bool] = None
    likelyUse: Optional[str] = None  # identify|inspect|assemble|repair|troubleshoot|unknown
    contextSource: str = "rules"     # yolo26|rules|depth|open-vocab|known-part-pose
    warnings: List[str] = Field(default_factory=list)


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

    # -- v2 / Build-Plan fields (all optional -- old app contract still works) --
    version: Optional[int] = 2
    workflowMode: Optional[str] = "build"
    sourceAssetId: Optional[str] = None

    sourceMaskB64: Optional[str] = None
    maskSource: Optional[str] = None
    maskContour: List[Point] = Field(default_factory=list)

    instruction: Optional[str] = None
    aiNotes: List[BlueprintNote] = Field(default_factory=list)
    nextAction: Optional[str] = None
    safetyWarning: Optional[str] = None
    qualityCheck: Optional[str] = None
    activityLabel: Optional[str] = None
    detectedIntent: Optional[str] = None
    importance: Optional[str] = None

    planSteps: List[PlanStep] = Field(default_factory=list)
    currentPlanStepIndex: Optional[int] = None

    # Plan Mode visual guidance overlays. Kept as flexible dicts so the `arrow`
    # type's nested `from`/`to` points pass through verbatim. Each item is:
    #   {id, type, [x, y] | [from:{x,y}, to:{x,y}], label?, stepId?, confidence?}
    # type in: arrow | target | ghost-position | highlight | warning-zone | callout | step-marker
    planOverlays: List[Dict[str, Any]] = Field(default_factory=list)

    # -- Plan-context fields (all optional; geometry/context for the app + DeepSeek) --
    selectedLabel: Optional[str] = None
    cropEntities: List[CropEntity] = Field(default_factory=list)
    cropSegments: List[CropSegment] = Field(default_factory=list)
    suggestedGoals: List[str] = Field(default_factory=list)
    virtualBlueprintPoints: List[VirtualBlueprintPoint] = Field(default_factory=list)
    reasoningSource: Optional[str] = None   # "rules" here; app may set "deepseek"

    depthPoints: List[DepthPoint] = Field(default_factory=list)
    depthSource: Optional[str] = None
    depthConfidence: Optional[float] = None
    depthWarning: Optional[str] = None

    planContext: Optional[PlanContext] = None

    # Optional future adapters -- disabled stubs return None (never required).
    knownPartPose: Optional[Dict[str, Any]] = None
    assemblyState: Optional[Dict[str, Any]] = None
