"""Agent 5 - Vision Improvement.

Proposes (NEVER auto-deploys) uncertain frames as privacy-flagged dataset
candidates for the human-approved training loop. Every candidate is flagged for
blurring and marked ``auto_deploy=False`` and ``review_status=pending``.
"""
from __future__ import annotations

from typing import Any

UNCERTAIN_CONF = 0.55  # detections below this confidence are worth re-labelling


def run_vision_improvement_agent(state: dict[str, Any]) -> dict[str, Any]:
    detections = state.get("detections") or []
    frame_ref = (state.get("frame_context") or {}).get("frame_ref", "")

    candidates: list[dict[str, Any]] = []
    for det in detections:
        raw_conf = det.get("confidence")
        if raw_conf is None:
            raw_conf = det.get("conf")
        conf = 1.0 if raw_conf is None else float(raw_conf)
        if conf < UNCERTAIN_CONF:
            candidates.append({
                "frame_ref": frame_ref,
                "hazard_tags": [det.get("label", "unknown")],
                "confidence": conf,
                "privacy_flags": {"contains_people": True, "blur_required": True},
                "review_status": "pending",
                "auto_deploy": False,
                "pipeline_stages": [
                    "privacy_blur",
                    "group_by_hazard",
                    "auto_label",
                    "human_review",
                    "export_yolo_coco",
                    "prepare_runpod_job",
                    "evaluate",
                    "compare_incumbent",
                    "human_promotion_approval",
                    "monitor",
                ],
            })

    return {
        "dataset_candidates": (state.get("dataset_candidates") or []) + candidates,
        "vision_improvement_plan": {
            "candidate_count": len(candidates),
            "export_formats": ["yolo", "coco"],
            "training_target": "runpod",
            "compare_to_incumbent": True,
            "promotion_requires_human_approval": True,
            "auto_deploy": False,
        },
        "action_log": (state.get("action_log") or [])
        + [{"agent": "vision_improvement", "status": "ok",
            "summary": f"{len(candidates)} training candidate(s) queued (no auto-deploy)"}],
    }
