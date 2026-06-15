"""Pure HSE decision logic shared by the LangGraph nodes and the worker routes.

No third-party imports on purpose: this module must stay importable in every
environment so the approval threshold and 5x5 risk-matrix banding can be
unit-tested without pydantic/langgraph/torch. Grounding: HSE Study Guide / NEBOSH
5x5 matrix bands and the score>=10 human-approval rule from the build prompt.
"""
from __future__ import annotations

from typing import Any

# 5x5 likelihood x severity bands: 1-4 low, 5-9 medium, 10-16 high, 17-25 critical.
APPROVAL_THRESHOLD = 10   # any score >= 10 must pass a human-approval interrupt gate
HALT_THRESHOLD = 17       # critical band: recommend stop-work pending approval

# Detection labels (draft-branch/schemas/model_classes.json) whose presence warrants deep
# contextual reasoning rather than a flat PPE pass/fail check.
RISK_SENSITIVE_CLASSES = frozenset({
    "open_hole", "open_panel", "suspended_load", "ladder", "scaffold",
    "forklift", "fire", "smoke", "spill", "trailing_cable", "blocked_exit",
    "gas_cylinder", "no_hardhat", "no_safety_vest", "no_gloves", "no_goggles",
})


def requires_approval(score: int) -> bool:
    """Any risk score >= 10 must route through a human-approval interrupt gate."""
    return int(score) >= APPROVAL_THRESHOLD


def should_halt(score: int) -> bool:
    """Critical band (>=17): recommend immediate stop-work pending approval."""
    return int(score) >= HALT_THRESHOLD


def band_for_score(score: int) -> str:
    s = int(score)
    if not 1 <= s <= 25:
        raise ValueError("risk score must be between 1 and 25")
    if s <= 4:
        return "low"
    if s <= 9:
        return "medium"
    if s <= 16:
        return "high"
    return "critical"


def is_risk_sensitive(label: str) -> bool:
    return str(label).lower() in RISK_SENSITIVE_CLASSES


def risk_sensitive_detections(detections: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """Filter detections to the classes that justify a reasoning-engine call."""
    out: list[dict[str, Any]] = []
    for det in detections or []:
        label = det.get("label") or det.get("class") or det.get("name") or ""
        if is_risk_sensitive(label):
            out.append(det)
    return out


def build_approval_request(action_type: str, payload: dict[str, Any], score: int) -> dict[str, Any]:
    """Shape of the interrupt() payload surfaced to the human approver."""
    return {
        "action_type": action_type,
        "score": int(score),
        "matrix_band": band_for_score(score),
        "halt_recommended": should_halt(score),
        "payload": payload,
        "decision_required": ["approve", "reject", "revise"],
    }
