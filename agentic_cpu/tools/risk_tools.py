"""
agentic_cpu/tools/risk_tools.py -- risk scoring + hierarchy-of-controls helpers.

Self-contained (a small local controls map) so the CPU agent does not need to
import the risk engine. 5x5 risk matrix bands match the deterministic engine's
defaults.
"""

from __future__ import annotations

from typing import Any, Dict, List

_HIERARCHY = ["elimination", "substitution", "engineering", "administrative", "ppe"]

# hazard_type -> ordered controls (hierarchy of controls). Generic, advisory.
_CONTROLS: Dict[str, List[Dict[str, str]]] = {
    "object_near_edge": [
        {"level": "elimination", "action": "Move the object away from the edge."},
        {"level": "engineering", "action": "Add a raised lip / tray / edge guard."},
        {"level": "administrative", "action": "Brief staff not to place items at edges."},
    ],
    "person_forklift_proximity": [
        {"level": "elimination", "action": "Separate pedestrians and forklifts (segregated routes)."},
        {"level": "engineering", "action": "Install barriers and proximity sensors."},
        {"level": "administrative", "action": "Enforce pedestrian exclusion zones and speed limits."},
        {"level": "ppe", "action": "High-visibility clothing for all on-foot staff."},
    ],
    "spill": [
        {"level": "elimination", "action": "Clean the spill immediately and remove the source."},
        {"level": "engineering", "action": "Install drainage / non-slip flooring."},
        {"level": "administrative", "action": "Place wet-floor signage and schedule inspections."},
    ],
    "fall_from_height": [
        {"level": "elimination", "action": "Do the work at ground level where possible."},
        {"level": "engineering", "action": "Use guardrails / scaffolding / MEWP."},
        {"level": "ppe", "action": "Use a fall-arrest harness anchored correctly."},
    ],
}
_DEFAULT_CONTROLS = [
    {"level": "engineering", "action": "Apply an engineering control appropriate to the hazard."},
    {"level": "administrative", "action": "Add a safe-work procedure and briefing."},
    {"level": "ppe", "action": "Provide suitable PPE as a last line of defence."},
]


def score(severity: int, likelihood: int) -> Dict[str, Any]:
    """5x5 risk score + band. severity/likelihood clamped to 1..5."""
    s = max(1, min(5, int(severity or 1)))
    likelihood_val = max(1, min(5, int(likelihood or 1)))
    rs = s * likelihood_val
    if rs <= 4:
        level = "GREEN"
    elif rs <= 9:
        level = "YELLOW"
    elif rs <= 15:
        level = "ORANGE"
    else:
        level = "RED"
    return {"severity": s, "likelihood": likelihood_val, "risk_score": rs, "risk_level": level}


def controls_for(hazard_type: str) -> List[Dict[str, str]]:
    return list(_CONTROLS.get(str(hazard_type or "").lower(), _DEFAULT_CONTROLS))


def residual_after_controls(severity: int, likelihood: int) -> Dict[str, Any]:
    """Naive residual: controls reduce likelihood by one band (never below 1)."""
    return score(severity, max(1, int(likelihood or 1) - 1))


def hierarchy() -> List[str]:
    return list(_HIERARCHY)
