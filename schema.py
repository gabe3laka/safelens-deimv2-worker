"""
schema.py - Pydantic models for the SafeLens vision worker request / response.

Supports two backends behind a single response contract:
  - edgecrafter (default): ECDet-S boxes + optional ECPose-S keypoints
  - deimv2 (legacy fallback): DEIMv2 boxes only

Response (returned to Eagle Vision 2):
{
  "entities": [
    {"label": "person", "class_id": 0, "confidence": 0.88,
     "bbox": {"x": 0.12, "y": 0.10, "w": 0.32, "h": 0.72},
     "source": "edgecrafter-det"}
  ],
  "poses": [
    {"label": "person", "confidence": 0.84,
     "keypoints": [{"name": "nose", "x": 0.31, "y": 0.18, "score": 0.91}],
     "skeleton": [[5, 7], [7, 9], [6, 8]],
     "source": "edgecrafter-pose"}
  ],
  "model": "EdgeCrafter",
  "backend": "edgecrafter",
  "tasks": ["det", "pose"],
  "inference_ms": 123,
  "img_w": 640,
  "img_h": 480,
  "error": null,
  "warning": null
}
"""

from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field


# -- Request ------------------------------------------------------------------

class InferRequest(BaseModel):
    """Payload forwarded from the Supabase Edge Function proxy."""

    image_b64: str = Field(..., description="Base-64 encoded image (JPEG or PNG)")
    conf: float = Field(0.25, ge=0.0, le=1.0, description="Confidence threshold")
    img_size: int = Field(
        640, ge=32, le=1280, description="Inference resolution (square)"
    )
    classes: Optional[List[int]] = Field(
        None,
        description="Optional COCO class-id filter. None returns all classes.",
    )


# -- Response -----------------------------------------------------------------

class BBox(BaseModel):
    """Bounding box normalised to 0..1 relative to the *original* image."""

    x: float = Field(..., ge=0.0, le=1.0, description="Left edge (normalised)")
    y: float = Field(..., ge=0.0, le=1.0, description="Top edge (normalised)")
    w: float = Field(..., ge=0.0, le=1.0, description="Width (normalised)")
    h: float = Field(..., ge=0.0, le=1.0, description="Height (normalised)")


class Entity(BaseModel):
    """A single detected object / person box.

    Backward compatible with the existing Eagle Vision 2 teal-box overlay.
    The optional 'source' field tags which model produced the box
    (e.g. 'edgecrafter-det' or 'deimv2').
    """

    label: str = Field(..., description="Human-readable COCO class name")
    class_id: int = Field(..., description="COCO class index (0=person, 2=car, ...)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence")
    bbox: BBox
    source: Optional[str] = Field(
        None, description="Producer tag, e.g. 'edgecrafter-det' or 'deimv2'"
    )


class Keypoint(BaseModel):
    """A single COCO keypoint, normalised to 0..1 on the original image."""

    name: str = Field(..., description="COCO keypoint name, e.g. 'nose'")
    x: float = Field(..., description="Normalised x (0..1)")
    y: float = Field(..., description="Normalised y (0..1)")
    score: float = Field(..., description="Keypoint visibility / confidence")


class Pose(BaseModel):
    """A single person pose: COCO-17 keypoints + skeleton edges.

    Optional output. The frontend is NOT required to draw poses until a
    later app PR; entities remain the primary backward-compatible payload.
    """

    label: str = Field("person", description="Pose class label")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Pose confidence")
    keypoints: List[Keypoint] = Field(default_factory=list)
    skeleton: List[List[int]] = Field(
        default_factory=list, description="COCO skeleton edges (0-based index pairs)"
    )
    source: Optional[str] = Field(
        None, description="Producer tag, e.g. 'edgecrafter-pose'"
    )


class InferResponse(BaseModel):
    """Payload returned to Eagle Vision 2.

    On success: entities (and optionally poses) populated; error/warning None.
    On error: entities is [] and poses is [], error holds a structured code.
    """

    entities: List[Entity] = Field(default_factory=list)
    poses: List[Pose] = Field(default_factory=list)
    inference_ms: float = Field(0.0, description="Wall-clock inference time in ms")
    model: str = Field("", description="Model identifier, e.g. 'EdgeCrafter'")
    backend: str = Field("", description="Active backend: 'edgecrafter' | 'deimv2'")
    tasks: List[str] = Field(
        default_factory=list, description="Active tasks, e.g. ['det', 'pose']"
    )
    img_w: int = Field(0, description="Original image width in pixels")
    img_h: int = Field(0, description="Original image height in pixels")

    error: Optional[str] = Field(
        None,
        description=(
            "Structured error code/message. Known values: 'missing_image_b64', "
            "'invalid_base64', 'model_not_ready', 'model_load_failed: <msg>'."
        ),
    )
    warning: Optional[str] = Field(
        None, description="Non-fatal warning (e.g. class filter matched nothing)."
    )
