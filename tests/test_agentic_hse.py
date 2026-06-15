"""Focused safety and integration tests for the agentic HSE layer."""
from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
from pathlib import Path

import pytest

from agentic_hse.approval import band_for_score, requires_approval, should_halt
from agentic_hse.graph import build_graph, route_after_assessment, route_after_approval
from agentic_hse.models import ReasoningRecord, RiskAssessmentDraft
from agentic_hse.nodes.observation_agent import _fallback
from agentic_hse.nodes.risk_assessment_agent import run_risk_assessment_agent
from agentic_hse.nodes.vision_improvement_agent import run_vision_improvement_agent

ROOT = Path(__file__).resolve().parents[2]


def test_score_boundaries_and_invalid_values():
    assert band_for_score(1) == "low"
    assert band_for_score(4) == "low"
    assert band_for_score(5) == "medium"
    assert band_for_score(9) == "medium"
    assert band_for_score(10) == "high"
    assert band_for_score(16) == "high"
    assert band_for_score(17) == "critical"
    assert band_for_score(25) == "critical"
    assert requires_approval(10) is True
    assert should_halt(17) is True
    with pytest.raises(ValueError):
        band_for_score(0)
    with pytest.raises(ValueError):
        band_for_score(26)


def test_reasoning_fallback_is_fail_safe():
    record = _fallback({"label": "open_hole", "confidence": 0.9}, "test")
    assert record["score"] >= 10
    assert record["requires_human_approval"] is True
    assert record["risk_state"] == "active"


def test_reasoning_and_risk_models_recompute_scores():
    record = ReasoningRecord(
        hazard="open hole",
        object_or_condition="open_hole",
        location_context="walkway",
        is_elevated=False,
        people_exposed=["worker"],
        risk_state="active",
        trigger_condition="worker enters opening",
        likelihood=3,
        severity=5,
        score=1,
        matrix_band="low",
        hierarchy_of_controls_recommendation=[],
        reasoning="contextual",
        standard_reference="site rule",
        requires_human_approval=False,
    )
    assert record.score == 15
    assert record.matrix_band == "high"
    assert record.requires_human_approval is True

    draft = RiskAssessmentDraft(
        hazard="open hole",
        likelihood=2,
        severity=4,
        score=1,
        matrix_band="low",
        residual_likelihood=1,
        residual_severity=2,
    )
    assert draft.score == 8
    assert draft.residual_score == 2
    assert draft.requires_human_approval is True


def test_multi_hazard_risk_uses_matching_record_only():
    state = {
        "events": [
            {"hazard": "spill hazard", "object_or_condition": "spill", "likelihood": 2, "severity": 2, "score": 4},
            {"hazard": "open hole hazard", "object_or_condition": "open_hole", "likelihood": 3, "severity": 5, "score": 15},
        ],
        "reasoning": {
            "records": [
                {
                    "hazard": "spill hazard",
                    "object_or_condition": "spill",
                    "people_exposed": ["cleaner"],
                    "hierarchy_of_controls_recommendation": [{"control_type": "administrative", "action": "clean spill"}],
                    "standard_reference": "spill rule",
                },
                {
                    "hazard": "open hole hazard",
                    "object_or_condition": "open_hole",
                    "people_exposed": ["worker"],
                    "hierarchy_of_controls_recommendation": [{"control_type": "engineering", "action": "install rated cover"}],
                    "standard_reference": "opening rule",
                },
            ]
        },
    }
    result = run_risk_assessment_agent(state)["risk_assessment"]
    assert result["persons_at_risk"] == ["worker"]
    assert result["recommended_controls"] == [{"control_type": "engineering", "action": "install rated cover"}]
    assert result["standard_reference"] == "opening rule"


def test_zero_confidence_detection_is_queued():
    result = run_vision_improvement_agent(
        {"detections": [{"label": "open_hole", "confidence": 0.0}], "frame_context": {"frame_ref": "frame://1"}}
    )
    assert result["dataset_candidates"][0]["confidence"] == 0.0
    assert result["dataset_candidates"][0]["auto_deploy"] is False


def test_graph_interrupt_approve_and_revise():
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.types import Command

    async def scenario():
        graph = build_graph(checkpointer=InMemorySaver())
        config = {"configurable": {"thread_id": "focused-graph-test"}}
        initial = {
            "detections": [{"label": "open_hole", "confidence": 0.9}],
            "frame_context": {"frame_ref": "frame://1"},
        }
        interrupted = await graph.ainvoke(initial, config=config)
        assert interrupted["__interrupt__"]

        revised = await graph.ainvoke(
            Command(resume={
                "decision": "revise",
                "notes": "add a rated cover",
                "revised_payload": {"likelihood": 3, "severity": 4},
            }),
            config=config,
        )
        assert revised["__interrupt__"]

        completed = await graph.ainvoke(
            Command(resume={"decision": "approve", "notes": "approved"}),
            config=config,
        )
        assert any(entry["status"] == "executed" for entry in completed["action_log"])

    asyncio.run(scenario())


def test_approval_router_rejects_execution():
    assert route_after_assessment({"pending_approval": {"score": 4}}) == "approval"
    assert route_after_approval({"approvals": [{"decision": {"decision": "reject"}}]}) == "log"
    assert route_after_approval({"approvals": [{"decision": {"decision": "revise"}}]}) == "revise"


def _load_prepare_dataset_module():
    path = ROOT / "runpod_training" / "prepare_dataset.py"
    spec = importlib.util.spec_from_file_location("prepare_dataset", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_dataset_split_keeps_groups_together(tmp_path):
    module = _load_prepare_dataset_module()
    samples = []
    for group in ("site-a", "site-b", "video-c", "video-d"):
        for index in range(3):
            samples.append(module.Sample(
                source="test",
                group_id=group,
                image=tmp_path / f"{group}-{index}.jpg",
                width=100,
                height=100,
                annotations=[],
                digest=f"{group}-{index}",
                provenance={"dataset": "test"},
            ))
    splits = module.split_samples(samples, seed=7)
    locations = {}
    for split, values in splits.items():
        for sample in values:
            locations.setdefault(sample.group_id, set()).add(split)
    assert all(len(value) == 1 for value in locations.values())


def test_rag_export_and_loader_contract_match():
    builder = (ROOT / "rag" / "build_local_index.py").read_text(encoding="utf-8")
    loader = (ROOT / "draft-branch" / "db" / "pgvector_loader.sql").read_text(encoding="utf-8")
    assert '"embedding_json"' in builder
    assert "documents_export.csv" in builder
    assert "chunks_export.csv" in builder
    assert "vector(384)" in loader
    assert "insert into documents" in loader.lower()


def test_json_artifacts_and_openapi_contract():
    for path in ROOT.rglob("*.json"):
        if "__pycache__" not in path.parts:
            json.loads(path.read_text(encoding="utf-8"))
    openapi = (ROOT / "draft-branch" / "integration" / "openapi-agentic-hse.yaml").read_text(encoding="utf-8")
    assert "reasoning_url" not in openapi
    assert "thread_id" in openapi
    assert "revised_payload" in openapi
