"""
risk/scene_graph.py -- deterministic geometric relationships between entities.

Pure geometry over normalized 0..1 bboxes (no ML, no GPU). Produces the
relations the risk rules reason over: near / overlaps / above / below /
left_of / right_of, plus per-entity edge-proximity flags. Deterministic and
cheap so it can run on the live /detect + /ws/vision path.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Tuple


def _near_threshold() -> float:
    try:
        return float(os.getenv("RISK_NEAR_THRESHOLD", "0.12"))
    except (TypeError, ValueError):
        return 0.12


def _edge_threshold() -> float:
    try:
        return float(os.getenv("RISK_EDGE_THRESHOLD", "0.04"))
    except (TypeError, ValueError):
        return 0.04


def centroid(bbox: Dict[str, float]) -> Dict[str, float]:
    return {"x": bbox.get("x", 0.0) + bbox.get("w", 0.0) / 2.0,
            "y": bbox.get("y", 0.0) + bbox.get("h", 0.0) / 2.0}


def centroid_distance(a: Dict[str, float], b: Dict[str, float]) -> float:
    ca, cb = centroid(a), centroid(b)
    dx, dy = ca["x"] - cb["x"], ca["y"] - cb["y"]
    return (dx * dx + dy * dy) ** 0.5


def iou(a: Dict[str, float], b: Dict[str, float]) -> float:
    ax0, ay0 = a.get("x", 0.0), a.get("y", 0.0)
    ax1, ay1 = ax0 + a.get("w", 0.0), ay0 + a.get("h", 0.0)
    bx0, by0 = b.get("x", 0.0), b.get("y", 0.0)
    bx1, by1 = bx0 + b.get("w", 0.0), by0 + b.get("h", 0.0)
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def edge_proximity(bbox: Dict[str, float], thresh: float | None = None) -> Dict[str, bool]:
    """Which image edges the bbox is close to (proxy for unsupported/near-fall)."""
    t = _edge_threshold() if thresh is None else thresh
    x0, y0 = bbox.get("x", 0.0), bbox.get("y", 0.0)
    x1, y1 = x0 + bbox.get("w", 0.0), y0 + bbox.get("h", 0.0)
    return {
        "left": x0 <= t,
        "right": x1 >= 1.0 - t,
        "top": y0 <= t,
        "bottom": y1 >= 1.0 - t,
    }


def build(entities: List[Dict[str, Any]], img_w: int = 0, img_h: int = 0) -> Dict[str, Any]:
    """Return {nodes, relations, near_threshold} for a list of entity dicts."""
    near_t = _near_threshold()
    nodes = []
    for i, e in enumerate(entities or []):
        bb = e.get("bbox") or {}
        nodes.append({
            "index": i,
            "label": str(e.get("label", "")),
            "class_id": int(e.get("class_id", -1)),
            "confidence": float(e.get("confidence", 0.0)),
            "centroid": centroid(bb),
            "edges": edge_proximity(bb),
        })

    relations: List[Dict[str, Any]] = []
    n = len(entities or [])
    for i in range(n):
        bi = entities[i].get("bbox") or {}
        ci = centroid(bi)
        for j in range(i + 1, n):
            bj = entities[j].get("bbox") or {}
            cj = centroid(bj)
            dist = centroid_distance(bi, bj)
            overlap = iou(bi, bj)
            if overlap > 0.0:
                relations.append({"subject": i, "relation": "overlaps", "object": j,
                                  "iou": round(overlap, 4), "distance": round(dist, 4)})
            elif dist <= near_t:
                relations.append({"subject": i, "relation": "near", "object": j,
                                  "iou": 0.0, "distance": round(dist, 4)})
            # vertical relation (subject above object) for overhead hazards
            if ci["y"] < cj["y"] - 0.02:
                relations.append({"subject": i, "relation": "above", "object": j,
                                  "distance": round(dist, 4)})
            elif cj["y"] < ci["y"] - 0.02:
                relations.append({"subject": j, "relation": "above", "object": i,
                                  "distance": round(dist, 4)})

    return {"nodes": nodes, "relations": relations,
            "near_threshold": near_t, "object_count": n}


def pairs_with_relation(scene: Dict[str, Any], relation: str) -> List[Tuple[int, int]]:
    """All (subject, object) index pairs with a given relation."""
    return [(r["subject"], r["object"]) for r in scene.get("relations", [])
            if r.get("relation") == relation]
