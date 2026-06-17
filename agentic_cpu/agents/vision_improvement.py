"""
agentic_cpu/agents/vision_improvement.py -- propose dataset candidates to improve
the detector. Approving a candidate for training is approval-required
(dataset_candidate_approve); this agent NEVER auto-approves and NEVER triggers
training. It only consumes detection JSON (low-confidence + corrected labels).
"""

from __future__ import annotations

from typing import Any, Dict

from ..tools import vision_tools
from . import new_action


def candidate(req: Dict[str, Any]) -> Dict[str, Any]:
    dc = req.get("detection_context") or {}
    low_conf = vision_tools.low_confidence_detections(dc)
    corrections = dc.get("semantic_corrections", []) or []
    candidates = []
    for lc in low_conf:
        candidates.append({"reason": "low_confidence", "label": lc.get("label"),
                           "confidence": lc.get("confidence"), "track_id": lc.get("track_id")})
    for c in corrections:
        candidates.append({"reason": "perception_correction",
                           "raw_label": c.get("raw_label"),
                           "corrected_label": c.get("corrected_label"),
                           "track_id": c.get("track_id")})
    payload = {
        "candidate_type": "dataset_candidate",
        "candidates": candidates,
        "session_id": dc.get("session_id"),
        "privacy_note": ("No raw frames are included. Frames for labelling must be "
                         "exported through the privacy-blur pipeline on approval."),
        "disclaimer": ("AI-proposed candidates. Approval is required before any image "
                       "is added to a training set or any training/deployment runs."),
    }
    return new_action(
        "dataset_candidate_approve", req,
        title="Vision-improvement dataset candidate",
        summary=(f"{len(candidates)} candidate(s) proposed for review "
                 f"({len(low_conf)} low-confidence, {len(corrections)} corrected)."),
        payload=payload,
    )
