"""
tests/test_agentic_cpu.py -- CPU agentic layer (/agent/*).

CPU-only, mock LLM, no DB. Covers health/ready, drafts return pending_approval,
the approval gate (execute rejects unapproved, accepts approved), job status,
queue saturation -> 429, job timeout -> structured error, disabled mode does not
break vision, and GPU-pressure degradation.
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("pydantic")

import agentic_cpu
from agentic_cpu import jobs


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("AGENTIC_CPU_ENABLED", "true")
    monkeypatch.setenv("CPU_AGENT_MODE", "mock")
    monkeypatch.setenv("CPU_AGENT_LLM_PROVIDER", "mock")
    monkeypatch.setenv("CPU_AGENT_REQUIRE_APPROVAL", "true")
    monkeypatch.delenv("CPU_AGENT_MAX_INFLIGHT", raising=False)
    monkeypatch.delenv("CPU_AGENT_QUEUE_MAX", raising=False)
    monkeypatch.delenv("CPU_AGENT_JOB_TIMEOUT_MS", raising=False)
    agentic_cpu.reset_all()
    yield
    agentic_cpu.reset_all()


@pytest.fixture()
def server_mod(monkeypatch):
    import importlib
    monkeypatch.setenv("SKIP_WARMUP", "true")
    monkeypatch.setenv("AUTO_WARMUP", "false")
    if "server" in sys.modules:
        del sys.modules["server"]
    return importlib.import_module("server")


@pytest.fixture()
def client(server_mod):
    from fastapi.testclient import TestClient
    with TestClient(server_mod.app) as c:
        yield c


_DRAFT_BODY = {
    "detection_context": {
        "session_id": "cam_1",
        "entities": [{"label": "cup", "confidence": 0.2, "track_id": "t1"}],
        "risks": [{"hazard_type": "object_near_edge", "risk_level": "YELLOW",
                   "severity": 2, "likelihood": 2, "risk_score": 4,
                   "recommended_controls": [{"level": "elimination", "action": "move it"}]}],
        "semantic_corrections": [{"raw_label": "bus", "corrected_label": "panel", "track_id": "t9"}],
    },
    "company_profile": {"company_name": "Acme", "industry": "hospitality"},
    "source": {"vision_session_id": "cam_1", "frame_id": "f1"},
}


# -- health / ready -----------------------------------------------------------

def test_agent_health(client):
    body = client.get("/agent/health").json()
    assert body["ok"] is True and body["layer"] == "agentic_cpu"
    assert body["enabled"] is True and body["mode"] == "mock"


def test_agent_ready(client):
    body = client.get("/agent/ready").json()
    assert body["agentic_cpu_enabled"] is True
    assert body["agentic_cpu_ready"] is True and body["llm_available"] is True


# -- drafts return pending_approval -------------------------------------------

@pytest.mark.parametrize("route,expected_type", [
    ("/agent/risk-assessment/draft", "risk_assessment_approve"),
    ("/agent/audit/draft", "audit_finding_send"),
    ("/agent/capa/draft", "capa_create"),
    ("/agent/training/draft", "training_record_create"),
    ("/agent/safety-observation/draft", "incident_create"),
    ("/agent/vision-improvement/candidate", "dataset_candidate_approve"),
])
def test_draft_routes_return_pending_approval(client, route, expected_type):
    r = client.post(route, json=_DRAFT_BODY)
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "pending_approval"
    assert body["requires_human_approval"] is True
    assert body["action_type"] == expected_type
    assert body["created_by"] == "agentic_cpu"
    assert "job_id" in body and body["action_id"].startswith("act_")


def test_company_profile_extract(client):
    r = client.post("/agent/company/profile/extract",
                    json={"text": "We run a busy cafe kitchen and follow OSHA."})
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["profile"]["industry"] == "hospitality"
    assert "OSHA" in body["profile"]["regulatory_frameworks"]


# -- approval gate ------------------------------------------------------------

def test_execute_rejects_unapproved(client):
    aid = client.post("/agent/risk-assessment/draft", json=_DRAFT_BODY).json()["action_id"]
    r = client.post("/agent/approvals/execute", json={"action_id": aid, "approved": False})
    assert r.status_code == 403
    assert r.json()["error"] == "approval_required"


def test_execute_accepts_approved(client):
    aid = client.post("/agent/risk-assessment/draft", json=_DRAFT_BODY).json()["action_id"]
    r = client.post("/agent/approvals/execute",
                    json={"action_id": aid, "approved": True, "approved_by": "alice"})
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True and body["status"] == "executed"
    assert body["action"]["approved_by"] == "alice"


def test_execute_unknown_action_404(client):
    r = client.post("/agent/approvals/execute",
                    json={"action_id": "act_nope", "approved": True, "approved_by": "x"})
    assert r.status_code == 404
    assert r.json()["error"] == "action_not_found"


def test_approvals_preview_then_execute(client):
    pv = client.post("/agent/approvals/preview",
                     json={"action_type": "capa_create", "payload": _DRAFT_BODY["detection_context"],
                           "source": {"frame_id": "f1"}})
    assert pv.status_code == 200
    aid = pv.json()["action_id"]
    assert pv.json()["status"] == "pending_approval"
    ex = client.post("/agent/approvals/execute",
                     json={"action_id": aid, "approved": True, "approved_by": "bob"})
    assert ex.status_code == 200 and ex.json()["status"] == "executed"


# -- jobs ---------------------------------------------------------------------

def test_jobs_status(client):
    body = client.post("/agent/risk-assessment/draft", json=_DRAFT_BODY).json()
    job_id = body["job_id"]
    jr = client.get(f"/agent/jobs/{job_id}")
    assert jr.status_code == 200
    assert jr.json()["status"] == "done"
    assert jr.json()["result"]["action_type"] == "risk_assessment_approve"


def test_jobs_unknown_404(client):
    assert client.get("/agent/jobs/job_nope").status_code == 404


def test_job_timeout_structured_error(monkeypatch):
    monkeypatch.setenv("CPU_AGENT_JOB_TIMEOUT_MS", "50")
    jobs.reset()
    jid = jobs.submit("slow", lambda: time.sleep(0.5))
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        job = jobs.get(jid)
        if job and job["status"] in ("error", "done"):
            break
        time.sleep(0.02)
    job = jobs.get(jid)
    assert job["status"] == "error" and job["error"] == "job_timeout"


def test_queue_saturation_returns_429(client, monkeypatch):
    monkeypatch.setenv("CPU_AGENT_MAX_INFLIGHT", "1")
    monkeypatch.setenv("CPU_AGENT_QUEUE_MAX", "1")
    jobs.reset()
    block = threading.Event()
    jobs.submit("blocker", lambda: block.wait(3.0))   # occupies the only slot
    try:
        r = client.post("/agent/risk-assessment/draft", json=_DRAFT_BODY)
        assert r.status_code == 429
        assert r.json()["error"] == "queue_full"
    finally:
        block.set()


# -- disabled mode + GPU pressure ---------------------------------------------

def test_disabled_blocks_agent_but_health_works(client, monkeypatch):
    monkeypatch.setenv("AGENTIC_CPU_ENABLED", "false")
    assert client.get("/agent/health").json()["enabled"] is False
    r = client.post("/agent/risk-assessment/draft", json=_DRAFT_BODY)
    assert r.status_code == 503 and r.json()["error"] == "agentic_cpu_disabled"


def test_disabled_agent_does_not_break_detect(server_mod, monkeypatch):
    """With the CPU agent disabled, /detect is fully unaffected."""
    import base64 as _b64
    import io as _io
    from fastapi.testclient import TestClient
    from PIL import Image
    import vision_backend
    from schema import BBox, Entity, InferResponse
    monkeypatch.setenv("AGENTIC_CPU_ENABLED", "false")
    monkeypatch.setattr(vision_backend, "run_inference", lambda **kw: InferResponse(
        entities=[Entity(label="person", class_id=0, confidence=0.9,
                         bbox=BBox(x=0.3, y=0.4, w=0.1, h=0.4), source="yolo26")],
        inference_ms=5, model="YOLO26", backend="yolo26", tasks=["det"], img_w=640, img_h=480))
    buf = _io.BytesIO(); Image.new("RGB", (8, 8), (200, 200, 200)).save(buf, "JPEG")
    img = _b64.b64encode(buf.getvalue()).decode()
    with TestClient(server_mod.app) as c:
        with server_mod._STATE_LOCK:
            server_mod._STATE["status"] = "ready"
        try:
            r = c.post("/detect", json={"image_b64": img, "session_id": "cam_1"})
            assert r.status_code == 200
            assert r.json()["entities"][0]["label"] == "person"
        finally:
            with server_mod._STATE_LOCK:
                server_mod._STATE["status"] = "cold"


def test_gpu_pressure_degrades_agent_first(client, monkeypatch):
    import agentic_cpu.config as cfg
    monkeypatch.setattr(cfg, "under_gpu_pressure", lambda: True)
    r = client.post("/agent/risk-assessment/draft", json=_DRAFT_BODY)
    assert r.status_code == 503
    assert r.json()["error"] == "degraded_gpu_pressure"


def test_ready_reports_both_layers(client):
    body = client.get("/ready").json()
    for key in ("gpu_vision_ready", "agentic_cpu_ready", "agentic_cpu_enabled"):
        assert key in body
