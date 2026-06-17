"""
temporal_reasoning/edge_risk.py -- deterministic object-near-edge temporal risk.

Detects when a tracked object sits near / is moving toward a support edge and
could fall. Uses bbox motion across recent frames + (if available) a detected
support surface (table/desk/bench/counter). When no surface is detected it falls
back to the FRAME edge and clearly marks edge_reference="frame_fallback" -- it
never invents surface geometry.

Pure Python (no torch/numpy); CPU-only; never raises.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from . import session_memory as mem
from .session_memory import _bool_env, _float_env, _int_env

# COCO-ish support surfaces an object might rest on / fall from.
_SURFACE_LABELS = {"dining table", "table", "desk", "bench", "counter", "couch",
                   "chair", "tv", "refrigerator", "oven", "shelf"}
# Objects we care about falling (small/portable). Persons/vehicles are excluded.
_PORTABLE_HINT = {"cup", "bottle", "wine glass", "bowl", "cell phone", "laptop",
                  "book", "vase", "remote", "mouse", "keyboard", "scissors",
                  "knife", "fork", "spoon", "potted plant", "clock", "box",
                  "handbag", "backpack", "tool"}
_NON_PORTABLE = {"person", "car", "bus", "truck", "train", "motorcycle",
                 "bicycle", "airplane", "boat"}


def enabled() -> bool:
    return _bool_env("OBJECT_EDGE_RISK_ENABLED", True)


def _distance_threshold() -> float:
    return _float_env("OBJECT_EDGE_DISTANCE_THRESHOLD", 0.10)


def _history_frames() -> int:
    return max(2, _int_env("OBJECT_EDGE_HISTORY_FRAMES", 6))


def _is_portable(label: Optional[str]) -> bool:
    if not label:
        return False
    lab = str(label).lower()
    if lab in _NON_PORTABLE:
        return False
    return lab in _PORTABLE_HINT or True  # default: treat unknown small objects as portable


def _surfaces(entities: List[Dict[str, Any]]) -> List[Dict[str, float]]:
    out = []
    for e in entities or []:
        if str(e.get("label", "")).lower() in _SURFACE_LABELS:
            b = e.get("bbox") or {}
            if b:
                out.append(b)
    return out


def _nearest_surface_edge_gap(cx: float, cy: float, bbox: Dict[str, float],
                              surfaces: List[Dict[str, float]]) -> Optional[float]:
    """Vertical gap from the object's bottom to the nearest surface's TOP edge.

    A small positive gap = object resting right at a surface lip. Returns None if
    the object does not horizontally overlap any surface (so we use frame edge).
    """
    obj_bottom = float(bbox.get("y", 0.0)) + float(bbox.get("h", 0.0))
    best: Optional[float] = None
    for s in surfaces:
        sx, sw = float(s.get("x", 0.0)), float(s.get("w", 0.0))
        sy = float(s.get("y", 0.0))
        if sx <= cx <= sx + sw:          # horizontally over this surface
            gap = abs(sy - obj_bottom)
            if best is None or gap < best:
                best = gap
    return best


def _frame_edge_gap(bbox: Dict[str, float]) -> float:
    """Smallest normalized gap from the bbox to any frame edge (0 = touching)."""
    x, y = float(bbox.get("x", 0.0)), float(bbox.get("y", 0.0))
    w, h = float(bbox.get("w", 0.0)), float(bbox.get("h", 0.0))
    return max(0.0, min(x, y, 1.0 - (x + w), 1.0 - (y + h)))


def evaluate(session_id: Optional[str], *, entities: List[Dict[str, Any]],
             tracks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return a list of object_near_edge EdgeRisk dicts (latent). Never raises."""
    if not enabled():
        return []
    try:
        surfaces = _surfaces(entities)
        risks: List[Dict[str, Any]] = []
        thr = _distance_threshold()
        rows = tracks or []
        if not rows:
            rows = [{"track_id": f"e{i}", "label": e.get("label"), "bbox": e.get("bbox", {})}
                    for i, e in enumerate(entities or [])]
        for t in rows:
            tid = str(t.get("track_id") or t.get("id") or "")
            label = t.get("label")
            bbox = t.get("bbox") or {}
            if not tid or not bbox or not _is_portable(label):
                continue
            cx = float(bbox.get("x", 0.0)) + float(bbox.get("w", 0.0)) / 2.0
            cy = float(bbox.get("y", 0.0)) + float(bbox.get("h", 0.0)) / 2.0
            hist = mem.track_history(session_id, tid)[-_history_frames():]

            surf_gap = _nearest_surface_edge_gap(cx, cy, bbox, surfaces)
            if surf_gap is not None:
                edge_reference = "surface"
                gap = surf_gap
            else:
                edge_reference = "frame_fallback"
                gap = _frame_edge_gap(bbox)

            near = gap <= thr
            # Motion: did the object move CLOSER to its edge over recent frames?
            approaching = False
            if len(hist) >= 2:
                if edge_reference == "frame_fallback":
                    prev_gap = _frame_edge_gap(hist[0]["bbox"])
                    approaching = (prev_gap - gap) > 0.01
                else:
                    prev_b = hist[0]["bbox"]
                    prev_gap = _nearest_surface_edge_gap(
                        float(prev_b.get("x", 0.0)) + float(prev_b.get("w", 0.0)) / 2.0,
                        cy, prev_b, surfaces)
                    if prev_gap is not None:
                        approaching = (prev_gap - gap) > 0.01
            if not (near or approaching):
                continue

            evidence = []
            if approaching:
                evidence.append("Tracked object moved closer to the edge over recent frames")
            if near:
                evidence.append("Object appears near a support boundary")
            likelihood = 3 if (near and approaching) else 2
            severity = 2
            risks.append({
                "risk_id": f"edge_risk_{tid}",
                "hazard_type": "object_near_edge",
                "risk_state": "latent",
                "trigger_condition": ("Object is close to the edge and may fall if "
                                      "bumped or moved further."),
                "risk_level": "YELLOW",
                "severity": severity,
                "likelihood": likelihood,
                "risk_score": severity * likelihood,
                "edge_reference": edge_reference,
                "involved_track_ids": [tid],
                "visual_evidence": evidence,
                "recommended_controls": [
                    {"level": "elimination", "action": "Move the object away from the edge."},
                    {"level": "engineering",
                     "action": "Use a raised lip, tray, or edge guard if objects are repeatedly placed there."},
                ],
                "produced_by": "deterministic_risk_engine",
                "requires_human_review": False,
            })
        return risks
    except Exception:  # noqa: BLE001 -- deterministic layer must never break /detect
        return []
