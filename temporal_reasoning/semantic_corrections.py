"""
temporal_reasoning/semantic_corrections.py -- perception corrections.

A semantic correction fixes a detector mislabel using scene context (e.g. a
ceiling-mounted "bus" -> "ceiling panel", suppressed from HSE alerts). It is a
PERCEPTION CORRECTION: produced_by=vlm_reasoner, purpose=perception_correction,
authority=advisory_perception, requires_human_review=FALSE -- because it only
corrects what the camera saw, it never creates or escalates a safety action.

Hard rules:
  * Raw detector output is PRESERVED (raw_label) -- corrections never delete it.
  * Real hazards (person, spill, knife, fire, edge risks, ...) are NEVER
    suppressed by a scene hint; only out-of-place vehicle/construction
    false-positives are corrected in mock mode.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .session_memory import _bool_env, _float_env

# Out-of-place-indoors detections we may correct (NEVER real hazards).
_VEHICLE_FALSE_POSITIVES = {
    "bus": "ceiling panel / large indoor fixture",
    "truck": "indoor fixture",
    "train": "indoor fixture",
    "airplane": "ceiling fixture",
    "boat": "indoor object",
    "car": "indoor object",
}
# Hazard-relevant labels that must NEVER be auto-suppressed.
_PROTECTED = {"person", "knife", "scissors", "fire hydrant", "fork", "spill",
              "fire", "smoke", "forklift"}
_INDOOR_ENVS = {"indoor_public", "indoor_workplace", "indoor_private",
                "indoor_industrial", "cafe", "office", "restaurant", "shop",
                "store", "home", "classroom", "kitchen", "lab", "indoor"}


def enabled() -> bool:
    return _bool_env("SEMANTIC_CORRECTION_ENABLED", True)


def contextual_suppression_enabled() -> bool:
    return _bool_env("CONTEXTUAL_SUPPRESSION_ENABLED", True)


def _low_conf_threshold() -> float:
    return _float_env("SEMANTIC_CORRECTION_LOW_CONF_THRESHOLD", 0.35)


def _is_indoor(scene_context: Dict[str, Any]) -> bool:
    env = str((scene_context or {}).get("environment_type", "")).lower()
    st = str((scene_context or {}).get("scene_type", "")).lower()
    return any(h in env or h in st for h in _INDOOR_ENVS)


def mock_corrections(entities: List[Dict[str, Any]], scene_context: Dict[str, Any],
                     tracks: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
    """Deterministic perception corrections for an indoor scene. Never raises."""
    if not enabled() or not contextual_suppression_enabled():
        return []
    if not _is_indoor(scene_context):
        return []
    out: List[Dict[str, Any]] = []
    # map entity index -> track id (best effort)
    rows = tracks or []
    by_idx: Dict[int, str] = {}
    for i, t in enumerate(rows):
        if t.get("track_id"):
            by_idx[i] = str(t["track_id"])
    for i, e in enumerate(entities or []):
        label = str(e.get("label", "")).lower()
        if label in _PROTECTED:
            continue
        if label in _VEHICLE_FALSE_POSITIVES:
            out.append({
                "track_id": e.get("track_id") or by_idx.get(i),
                "raw_label": label,
                "corrected_label": _VEHICLE_FALSE_POSITIVES[label],
                "correction_type": "false_positive",
                "action": "suppress_from_hse_alerts",
                "confidence": round(float(scene_context.get("confidence", 0.7) or 0.7), 3),
                "reason": (f"A '{label}' is implausible in an indoor "
                           f"{scene_context.get('scene_type', 'scene')}; it is most "
                           f"likely an indoor fixture, not a vehicle."),
                "produced_by": "vlm_reasoner",
                "purpose": "perception_correction",
                "authority": "advisory_perception",
                "requires_human_review": False,
            })
    return out


def from_vlm_json(data: Dict[str, Any], entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Parse perception corrections from VLM JSON; force the advisory contract."""
    if not isinstance(data, dict):
        return []
    raw = data.get("semantic_corrections")
    if not isinstance(raw, list):
        return []
    out: List[Dict[str, Any]] = []
    for c in raw:
        if not isinstance(c, dict):
            continue
        if str(c.get("raw_label", "")).lower() in _PROTECTED:
            continue  # never suppress a real hazard, whatever the model says
        out.append({
            "track_id": c.get("track_id"),
            "raw_label": str(c.get("raw_label", "")),
            "corrected_label": str(c.get("corrected_label", "")),
            "correction_type": str(c.get("correction_type", "false_positive")),
            "action": str(c.get("action", "suppress_from_hse_alerts")),
            "confidence": float(c.get("confidence", 0.5) or 0.5),
            "reason": str(c.get("reason", ""))[:500],
            "produced_by": "vlm_reasoner",
            "purpose": "perception_correction",
            "authority": "advisory_perception",
            "requires_human_review": False,   # enforced, never trusted from model
        })
    return out
