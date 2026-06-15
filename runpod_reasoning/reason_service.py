"""RunPod Senior-QHSE-Manager reasoning service scaffold.

The deterministic implementation keeps the full typed contract executable while
the selected VLM adapter is still deferred. RunPod, not this bundle build,
provides the GPU runtime.
"""
from __future__ import annotations

from typing import Any, Literal

from fastapi import FastAPI
from pydantic import BaseModel, Field, model_validator

APPROVAL_THRESHOLD = 10


def _band(score: int) -> str:
    if score <= 4:
        return "low"
    if score <= 9:
        return "medium"
    if score <= 16:
        return "high"
    return "critical"


class Detection(BaseModel):
    label: str
    confidence: float = Field(ge=0, le=1)
    bbox: list[float] | None = None


class ReasonRequest(BaseModel):
    detections: list[Detection] = Field(min_length=1)
    frame_ref: str | None = None
    company_profile: dict[str, Any] = Field(default_factory=dict)
    zone_context: dict[str, Any] = Field(default_factory=dict)


class ControlRecommendation(BaseModel):
    control_type: Literal["elimination", "substitution", "engineering", "administrative", "ppe"]
    action: str = Field(min_length=1)


class ReasoningRecord(BaseModel):
    hazard: str
    object_or_condition: str
    location_context: str
    is_elevated: bool
    people_exposed: list[str]
    risk_state: Literal["latent", "active"]
    trigger_condition: str
    likelihood: int = Field(ge=1, le=5)
    severity: int = Field(ge=1, le=5)
    score: int = Field(ge=1, le=25)
    matrix_band: Literal["low", "medium", "high", "critical"]
    hierarchy_of_controls_recommendation: list[ControlRecommendation]
    reasoning: str
    standard_reference: str
    requires_human_approval: bool

    @model_validator(mode="after")
    def enforce_matrix(self) -> "ReasoningRecord":
        self.score = self.likelihood * self.severity
        self.matrix_band = _band(self.score)  # type: ignore[assignment]
        self.requires_human_approval = self.score >= APPROVAL_THRESHOLD
        return self


app = FastAPI(title="safelens-reasoning-service", version="0.2.0")


@app.get("/health")
async def health() -> dict[str, Any]:
    return {
        "ok": True,
        "service": "safelens-reasoning",
        "model": "deterministic-contract-scaffold",
        "vlm_wired": False,
    }


@app.post("/reason", response_model=ReasoningRecord)
async def reason(req: ReasonRequest) -> ReasoningRecord:
    return ReasoningRecord.model_validate(_reason_with_vlm(req))


def _reason_with_vlm(req: ReasonRequest) -> dict[str, Any]:
    """Replace this body with the reviewed VLM adapter without changing I/O."""
    det = req.detections[0]
    zone = req.zone_context
    elevated = bool(zone.get("is_elevated", False))
    people = zone.get("people_exposed") or (["worker below"] if elevated else [])
    likelihood = 3 if people else 2
    severity = 4 if elevated else 3
    score = likelihood * severity
    return {
        "hazard": f"{det.label} hazard",
        "object_or_condition": det.label,
        "location_context": zone.get("zone_type", "unspecified location"),
        "is_elevated": elevated,
        "people_exposed": people,
        "risk_state": "active" if people else "latent",
        "trigger_condition": (
            "people are currently inside the exposure path"
            if people
            else "displacement, vibration, or entry into the exposure zone"
        ),
        "likelihood": likelihood,
        "severity": severity,
        "score": score,
        "matrix_band": _band(score),
        "hierarchy_of_controls_recommendation": [
            {"control_type": "engineering", "action": "install a guard, barrier, or edge-protection control"},
            {"control_type": "administrative", "action": "establish an exclusion zone and authorized supervision"},
            {"control_type": "ppe", "action": "apply task-appropriate PPE as the final control layer"},
        ],
        "reasoning": (
            "Deterministic relational scaffold. Exposure and elevation alter the "
            "risk state; a reviewed VLM adapter must replace this implementation."
        ),
        "standard_reference": "OSHA 29 CFR 1926 / ISO 45001 6.1.2 (confirm per hazard)",
        "requires_human_approval": score >= APPROVAL_THRESHOLD,
    }
