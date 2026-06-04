"""
schema.py — Pydantic models for the DEIMv2 RunPod worker request / response.

Request  (sent by Eagle Vision 2 Supabase Edge Function proxy):
  {
      "image_b64": "<base64-encoded JPEG/PNG>",   # required
          "conf":      0.35,                           # optional, default 0.35
              "img_size":  640,                            # optional, default 640
                  "classes":   [0, 2, 7]                       # optional COCO class filter
                    }

                    Response (returned to Eagle Vision 2 BackendVisionDetector):
                      {
                          "entities": [
                                {
                                        "label":      "person",   # COCO class name
                                                "class_id":   0,          # COCO class index
                                                        "confidence": 0.87,       # 0..1
                                                                "bbox": {
                                                                          "x": 0.12, "y": 0.08,  # normalised 0..1 (top-left)
                                                                                    "w": 0.18, "h": 0.52   # normalised 0..1
                                                                                            }
                                                                                                  }
                                                                                                      ],
                                                                                                          "inference_ms": 45.2,        # wall-clock inference time
                                                                                                              "model":        "deimv2-s",  # which model variant was used
                                                                                                                  "img_w": 640,
                                                                                                                      "img_h": 480
                                                                                                                        }
                                                                                                                        """

from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field


# ── Request ──────────────────────────────────────────────────────────────────

class InferRequest(BaseModel):
      """Payload forwarded from the Supabase Edge Function proxy."""

    image_b64: str = Field(..., description="Base-64 encoded image (JPEG or PNG)")
    conf: float = Field(0.35, ge=0.0, le=1.0, description="Confidence threshold")
    img_size: int = Field(640, ge=32, le=1280, description="Inference resolution (shorter side)")
    classes: Optional[List[int]] = Field(
              None,
              description="Optional COCO class-id filter. None → return all classes.",
    )


# ── Response ─────────────────────────────────────────────────────────────────

class BBox(BaseModel):
      """Bounding box normalised to 0..1 relative to the *original* image."""

    x: float = Field(..., ge=0.0, le=1.0, description="Left edge")
    y: float = Field(..., ge=0.0, le=1.0, description="Top edge")
    w: float = Field(..., ge=0.0, le=1.0, description="Width")
    h: float = Field(..., ge=0.0, le=1.0, description="Height")


class Entity(BaseModel):
      """A single detected object."""

    label: str = Field(..., description="Human-readable COCO class name")
    class_id: int = Field(..., description="COCO class index (0=person, 2=car, …)")
    confidence: float = Field(..., ge=0.0, le=1.0)
    bbox: BBox


class InferResponse(BaseModel):
      """Payload returned to Eagle Vision 2."""

    entities: List[Entity] = Field(default_factory=list)
    inference_ms: float = Field(..., description="Wall-clock inference time in ms")
    model: str = Field(..., description="Model variant identifier, e.g. 'deimv2-s'")
    img_w: int = Field(..., description="Original image width in pixels")
    img_h: int = Field(..., description="Original image height in pixels")
