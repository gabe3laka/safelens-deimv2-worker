"""
risk/controls.py -- hazard_type -> hierarchy-ordered recommended controls.

Follows the OSHA/NIOSH/ISO-45001 hierarchy of controls, most-effective first:
    elimination -> substitution -> engineering -> administrative -> ppe

The engine never recommends "wear PPE" as the only control; each hazard maps to
controls spanning the hierarchy so the response nudges up it. Returned as plain
dicts ({level, action}) that validate against risk_schema.Control.
"""

from __future__ import annotations

from typing import Dict, List

# Canonical ordering used to sort/validate any control list.
HIERARCHY = ("elimination", "substitution", "engineering", "administrative", "ppe")
_ORDER = {level: i for i, level in enumerate(HIERARCHY)}

# hazard_type -> ordered controls. Keep wording control-oriented, never
# "operate live equipment" style instructions.
_CONTROLS: Dict[str, List[Dict[str, str]]] = {
    "ppe_missing_hardhat": [
        {"level": "engineering", "action": "Restrict the area to remove overhead/strike exposure."},
        {"level": "administrative", "action": "Enforce hard-hat policy at the access point; brief the worker."},
        {"level": "ppe", "action": "Require a compliant hard hat before entry."},
    ],
    "ppe_missing_vest": [
        {"level": "administrative", "action": "Enforce hi-vis policy in the traffic/active zone."},
        {"level": "ppe", "action": "Require a compliant high-visibility vest before entry."},
    ],
    "ppe_missing_gloves": [
        {"level": "administrative", "action": "Confirm the task's glove requirement and brief the worker."},
        {"level": "ppe", "action": "Require task-appropriate gloves."},
    ],
    "person_vehicle_proximity": [
        {"level": "elimination", "action": "Separate pedestrians and vehicles with segregated routes."},
        {"level": "engineering", "action": "Add barriers, exclusion zones, or proximity warning systems."},
        {"level": "administrative", "action": "Enforce right-of-way rules, banksman, and speed limits."},
        {"level": "ppe", "action": "Ensure high-visibility clothing for all pedestrians."},
    ],
    "person_forklift_proximity": [
        {"level": "elimination", "action": "Keep pedestrians out of the forklift operating zone."},
        {"level": "engineering", "action": "Install physical barriers / proximity detection on the MHE."},
        {"level": "administrative", "action": "Apply keep-clear rules and operator/pedestrian briefings."},
        {"level": "ppe", "action": "Require high-visibility clothing in the zone."},
    ],
    "fire": [
        {"level": "elimination", "action": "Remove ignition sources and combustibles from the area."},
        {"level": "engineering", "action": "Activate suppression; verify detection and alarms."},
        {"level": "administrative", "action": "Initiate emergency response and evacuation per plan."},
    ],
    "smoke": [
        {"level": "engineering", "action": "Verify detection/ventilation; locate the source."},
        {"level": "administrative", "action": "Investigate and prepare to evacuate if it escalates."},
    ],
    "blocked_exit": [
        {"level": "elimination", "action": "Clear the obstruction from the escape route immediately."},
        {"level": "administrative", "action": "Re-brief housekeeping and keep-clear rules for exits."},
    ],
    "object_near_edge": [
        {"level": "elimination", "action": "Move the object away from the edge to a stable position."},
        {"level": "engineering", "action": "Add a raised lip / edge guard to the surface."},
        {"level": "administrative", "action": "Mark the surface keep-clear and check during inspections."},
    ],
    "working_at_height_ladder": [
        {"level": "elimination", "action": "Avoid work at height where the task allows."},
        {"level": "engineering", "action": "Use a stable platform/guardrails; secure and inspect the ladder."},
        {"level": "administrative", "action": "Apply ladder safety rules (angle, footing, three points)."},
        {"level": "ppe", "action": "Use fall-arrest where collective protection is insufficient."},
    ],
    "scaffold_proximity": [
        {"level": "engineering", "action": "Verify guardrails, toe boards, and scaffold inspection tag."},
        {"level": "administrative", "action": "Restrict the drop zone and enforce inspection intervals."},
        {"level": "ppe", "action": "Use fall-arrest at unprotected edges."},
    ],
    "spill_slip": [
        {"level": "elimination", "action": "Clean up the spill and remove the slip source."},
        {"level": "engineering", "action": "Contain/bund the substance; improve drainage."},
        {"level": "administrative", "action": "Barricade and sign the area until dry; brief housekeeping."},
    ],
    "electrical_exposed": [
        {"level": "elimination", "action": "De-energise and isolate (LOTO) before any work."},
        {"level": "engineering", "action": "Guard or enclose the exposed conductor; verify barriers."},
        {"level": "administrative", "action": "Restrict access to qualified persons under permit-to-work."},
    ],
    "suspended_load_overhead": [
        {"level": "elimination", "action": "Keep personnel out from under suspended loads."},
        {"level": "engineering", "action": "Use exclusion zones and verify rigging/lifting gear."},
        {"level": "administrative", "action": "Apply lift plan, banksman, and keep-clear enforcement."},
    ],
    "open_hole_fall": [
        {"level": "elimination", "action": "Cover or fill the opening; eliminate the fall path."},
        {"level": "engineering", "action": "Install guardrails / a secured, load-rated cover."},
        {"level": "administrative", "action": "Barricade and sign the opening; brief nearby workers."},
    ],
}

_DEFAULT = [
    {"level": "administrative", "action": "Inspect the condition and apply the relevant safe system of work."},
    {"level": "ppe", "action": "Ensure task-appropriate PPE is worn."},
]


def controls_for(hazard_type: str) -> List[Dict[str, str]]:
    """Return hierarchy-ordered controls for a hazard_type (sorted, deduped)."""
    raw = _CONTROLS.get(hazard_type, _DEFAULT)
    ordered = sorted(raw, key=lambda c: _ORDER.get(c["level"], len(HIERARCHY)))
    return [dict(c) for c in ordered]


def primary_action(hazard_type: str) -> str:
    """The single highest-priority recommended action for a hazard_type."""
    ctrls = controls_for(hazard_type)
    return ctrls[0]["action"] if ctrls else _DEFAULT[0]["action"]
