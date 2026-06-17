"""
agentic_cpu/router.py -- FastAPI APIRouter for the CPU agentic layer.

Mounted into the existing app under /agent/* (server.py). Every route is bounded
and CPU-only; none can block /detect. Serious drafts are pending_approval and
only finalize via /agent/approvals/execute with approved=true.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from . import agents, approvals, config, graph, jobs, llm
from .agents import company_setup

log = logging.getLogger("safelens-vision-worker.agentic.router")

router = APIRouter()


# -- guards -------------------------------------------------------------------

def _disabled_response() -> JSONResponse:
    return JSONResponse({"ok": False, "error": "agentic_cpu_disabled",
                         "enabled": False}, status_code=503)


def _pressure_response() -> JSONResponse:
    return JSONResponse({"ok": False, "error": "degraded_gpu_pressure",
                         "degraded_mode": "gpu_pressure"}, status_code=503)


def _new_work_guard():
    """Block new agent work when disabled or under GPU pressure (degrade CPU
    first, never /detect). Returns a JSONResponse to short-circuit, or None."""
    if not config.agentic_enabled():
        return _disabled_response()
    if config.under_gpu_pressure():
        return _pressure_response()
    return None


# -- health / ready -----------------------------------------------------------

@router.get("/health")
async def agent_health():
    return JSONResponse({
        "ok": True, "layer": "agentic_cpu",
        "enabled": config.agentic_enabled(), "mode": config.mode(),
        "jobs_inflight": jobs.queue_depth(), "queue_depth": jobs.queue_depth(),
    })


@router.get("/ready")
async def agent_ready():
    enabled = config.agentic_enabled()
    ready = enabled and llm.available() and not config.under_gpu_pressure()
    return JSONResponse({
        "ok": ready,
        "agentic_cpu_ready": ready,
        "agentic_cpu_enabled": enabled,
        "mode": config.mode(),
        "llm_available": llm.available(),
        "under_gpu_pressure": config.under_gpu_pressure(),
    })


# -- company profile (informational; no approval) -----------------------------

@router.post("/company/profile/extract")
async def company_profile_extract(payload: Dict[str, Any]):
    guard = _new_work_guard()
    if guard:
        return guard
    try:
        profile = company_setup.extract_profile(payload or {})
        return JSONResponse({"ok": True, "profile": profile})
    except Exception as exc:  # noqa: BLE001
        log.warning("company profile extract failed: %s", exc)
        return JSONResponse({"ok": False, "error": f"{type(exc).__name__}: {exc}"},
                            status_code=500)


# -- drafting routes (bounded background job; pending_approval) ----------------

def _run_draft(action_type: str, body: Dict[str, Any]):
    guard = _new_work_guard()
    if guard:
        return guard
    try:
        job_id = jobs.submit_and_wait(action_type, agents.dispatch, action_type, body or {})
    except jobs.QueueFull as exc:
        return JSONResponse({"ok": False, "error": "queue_full", "detail": str(exc)},
                            status_code=429)
    job = jobs.get(job_id)
    if job and job.get("status") == "done":
        action = job.get("result") or {}
        approvals.register(action)
        return JSONResponse({**action, "job_id": job_id})
    if job and job.get("status") == "error":
        return JSONResponse({"ok": False, "error": job.get("error", "job_failed"),
                             "job_id": job_id}, status_code=500)
    # long job still running -> 202 accepted; poll GET /agent/jobs/{job_id}
    return JSONResponse({"ok": True, "status": "accepted", "job_id": job_id,
                         "requires_human_approval": True}, status_code=202)


@router.post("/safety-observation/draft")
async def safety_observation_draft(payload: Dict[str, Any]):
    return _run_draft("incident_create", payload)


@router.post("/risk-assessment/draft")
async def risk_assessment_draft(payload: Dict[str, Any]):
    return _run_draft("risk_assessment_approve", payload)


@router.post("/audit/draft")
async def audit_draft(payload: Dict[str, Any]):
    return _run_draft("audit_finding_send", payload)


@router.post("/capa/draft")
async def capa_draft(payload: Dict[str, Any]):
    return _run_draft("capa_create", payload)


@router.post("/training/draft")
async def training_draft(payload: Dict[str, Any]):
    return _run_draft("training_record_create", payload)


@router.post("/vision-improvement/candidate")
async def vision_improvement_candidate(payload: Dict[str, Any]):
    return _run_draft("dataset_candidate_approve", payload)


# -- approvals ----------------------------------------------------------------

@router.post("/approvals/preview")
async def approvals_preview(payload: Dict[str, Any]):
    guard = _new_work_guard()
    if guard:
        return guard
    action_type = (payload or {}).get("action_type")
    if not action_type or not graph.is_registered(action_type):
        return JSONResponse({"ok": False, "error": "unknown_action_type",
                             "action_type": action_type}, status_code=400)
    try:
        action = approvals.preview(action_type, (payload or {}).get("payload", {}),
                                   (payload or {}).get("source", {}))
        return JSONResponse(action)
    except Exception as exc:  # noqa: BLE001
        return JSONResponse({"ok": False, "error": f"{type(exc).__name__}: {exc}"},
                            status_code=500)


@router.post("/approvals/execute")
async def approvals_execute(payload: Dict[str, Any]):
    if not config.agentic_enabled():
        return _disabled_response()
    action_id = (payload or {}).get("action_id")
    if not action_id:
        return JSONResponse({"ok": False, "error": "action_id_required"}, status_code=400)
    result = approvals.execute(action_id, bool((payload or {}).get("approved", False)),
                               (payload or {}).get("approved_by"))
    status = 200 if result.get("ok") else (
        404 if result.get("error") == "action_not_found" else 403)
    return JSONResponse(result, status_code=status)


# -- jobs ---------------------------------------------------------------------

@router.get("/jobs/{job_id}")
async def get_job(job_id: str):
    job = jobs.get(job_id)
    if not job:
        return JSONResponse({"ok": False, "error": "job_not_found", "job_id": job_id},
                            status_code=404)
    return JSONResponse({"ok": True, **job})
