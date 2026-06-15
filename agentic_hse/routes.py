"""Typed HTTP surface for the SafeLens agentic HSE graph."""
from __future__ import annotations

import os
import uuid
from typing import Any

from fastapi import APIRouter, HTTPException

from .approval import APPROVAL_THRESHOLD, HALT_THRESHOLD
from .models import (
    ApprovalDecisionRequest,
    AuditObservationDraft,
    ReasonRequest,
    ReasoningRecord,
    RiskAssessmentDraft,
    SessionStartRequest,
    TrainingModuleDraft,
)
from .runtime import get_runtime, graph_config, summarize_result

router = APIRouter(prefix="/agentic", tags=["agentic-hse"])


async def _runtime_or_503():
    try:
        return await get_runtime()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=503, detail=f"agentic_runtime_unavailable: {exc}") from exc


@router.get("/health")
async def agentic_health() -> dict[str, Any]:
    runtime = await _runtime_or_503()
    return {
        "ok": True,
        "service": "agentic-hse",
        "approval_threshold": APPROVAL_THRESHOLD,
        "halt_threshold": HALT_THRESHOLD,
        "checkpoint_backend": runtime.backend,
        "durable": runtime.durable,
    }


@router.post("/session/start")
async def session_start(payload: SessionStartRequest) -> dict[str, Any]:
    runtime = await _runtime_or_503()
    thread_id = payload.thread_id or str(uuid.uuid4())
    initial_state = payload.model_dump(mode="json", exclude={"thread_id"})
    initial_state["thread_id"] = thread_id
    initial_state["reasoning_url"] = os.getenv("SAFELENS_REASONING_URL", "").strip()
    try:
        result = await runtime.graph.ainvoke(initial_state, config=graph_config(thread_id))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=503, detail=f"graph_start_failed: {exc}") from exc
    summary = summarize_result(result)
    return {
        "ok": True,
        "thread_id": thread_id,
        "checkpoint_backend": runtime.backend,
        "durable": runtime.durable,
        **summary,
    }


@router.post("/reason", response_model=ReasoningRecord)
async def reason(payload: ReasonRequest) -> ReasoningRecord:
    reasoning_url = os.getenv("SAFELENS_REASONING_URL", "").strip()
    request_payload = payload.model_dump(mode="json")
    if reasoning_url:
        try:
            from .reasoning_client import reason_over_hazard

            record = await reason_over_hazard(reasoning_url, request_payload)
            return ReasoningRecord.model_validate(record)
        except Exception:
            pass

    from .nodes.observation_agent import _fallback

    fallback = _fallback(request_payload["detections"][0], note="reasoning_service_unavailable")
    return ReasoningRecord.model_validate(fallback)


@router.post("/risk-assessment/draft", response_model=RiskAssessmentDraft)
async def risk_assessment_draft(payload: RiskAssessmentDraft) -> RiskAssessmentDraft:
    return RiskAssessmentDraft.model_validate(payload.model_dump())


@router.post("/risk-assessment/approve")
async def risk_assessment_approve(payload: ApprovalDecisionRequest) -> dict[str, Any]:
    runtime = await _runtime_or_503()
    try:
        from langgraph.types import Command

        result = await runtime.graph.ainvoke(
            Command(resume=payload.model_dump(mode="json", exclude_none=True)),
            config=graph_config(payload.thread_id),
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=409, detail=f"graph_resume_failed: {exc}") from exc
    summary = summarize_result(result)
    action_log = summary["state"].get("action_log") or []
    executed = any(item.get("status") == "executed" for item in action_log)
    return {
        "ok": True,
        "thread_id": payload.thread_id,
        "decision": payload.decision,
        "executed": executed,
        **summary,
    }


@router.post("/audit/draft", response_model=AuditObservationDraft)
async def audit_draft(payload: AuditObservationDraft) -> AuditObservationDraft:
    return payload


@router.post("/training/draft", response_model=TrainingModuleDraft)
async def training_draft(payload: TrainingModuleDraft) -> TrainingModuleDraft:
    payload.anonymized = True
    return payload
