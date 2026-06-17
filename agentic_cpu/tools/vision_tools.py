"""
agentic_cpu/tools/vision_tools.py -- summarise detection JSON for the agents.

Consumes the /detect response shape (entities + risk blocks). Does NOT run
inference and does NOT import any vision/GPU module.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List


def summarize_detection(detection_context: Dict[str, Any]) -> Dict[str, Any]:
    """Counts + top labels + risk summary from a detection response dict."""
    dc = detection_context or {}
    entities = dc.get("entities", []) or []
    labels = Counter(str(e.get("label", "unknown")) for e in entities)
    risks = dc.get("risks", []) or []
    scene_risks = dc.get("scene_risks", []) or []
    return {
        "entity_count": len(entities),
        "label_counts": dict(labels),
        "top_labels": [lab for lab, _ in labels.most_common(5)],
        "risk_count": len(risks),
        "scene_risk_count": len(scene_risks),
        "highest_risk_level": dc.get("highest_risk_level", "GREEN"),
        "scene_context": dc.get("scene_context", {}),
        "backend": dc.get("backend"),
    }


def extract_hazards(detection_context: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Flatten deterministic + scene risks into a hazard list for drafting."""
    dc = detection_context or {}
    out: List[Dict[str, Any]] = []
    for r in (dc.get("risks", []) or []):
        out.append({
            "hazard_type": r.get("hazard_type", "unknown"),
            "risk_level": r.get("risk_level", "GREEN"),
            "severity": r.get("severity", 1),
            "likelihood": r.get("likelihood", 1),
            "risk_score": r.get("risk_score", 1),
            "involved_track_ids": r.get("involved_track_ids", []),
            "recommended_controls": r.get("recommended_controls", []),
            "source": r.get("produced_by", "risk_engine"),
        })
    for r in (dc.get("scene_risks", []) or []):
        out.append({
            "hazard_type": r.get("hazard_type", "unknown"),
            "risk_level": r.get("risk_level", "GREEN"),
            "severity": r.get("severity", 1),
            "likelihood": r.get("likelihood", 1),
            "risk_score": r.get("risk_score", 1),
            "involved_track_ids": r.get("involved_track_ids", []),
            "recommended_controls": r.get("recommended_controls", []),
            "source": r.get("produced_by", "vlm_reasoner"),
            "requires_human_review": True,
        })
    return out


def low_confidence_detections(detection_context: Dict[str, Any],
                              threshold: float = 0.35) -> List[Dict[str, Any]]:
    """Detections below `threshold` -- candidates for the vision-improvement agent."""
    dc = detection_context or {}
    out = []
    for e in (dc.get("entities", []) or []):
        try:
            if float(e.get("confidence", 1.0)) < threshold:
                out.append({"label": e.get("label"), "confidence": e.get("confidence"),
                            "track_id": e.get("track_id"), "bbox": e.get("bbox", {})})
        except (TypeError, ValueError):
            continue
    return out
