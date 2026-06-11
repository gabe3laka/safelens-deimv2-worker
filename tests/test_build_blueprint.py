"""
tests/test_build_blueprint.py -- Build Mode / BlueprintFrame v2 tests (CPU-only).

Covers workflowMode storage + resolution, v2 response fields, fallback-contour
mask, Build vs Plan notes/steps, safe SAM2 fallback, and v2 replay. No GPU, no
SAM2, no model -- the fallback contour pipeline runs on plain OpenCV/Pillow.
"""

from __future__ import annotations

import asyncio
import base64
import io
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

import pytest

pytest.importorskip("cv2")
pytest.importorskip("PIL")

import build_blueprint as bb
from PIL import Image, ImageDraw

REGION = {"x": 0.1, "y": 0.2, "w": 0.4, "h": 0.3}


def _crop_b64() -> str:
    im = Image.new("RGB", (200, 150), (245, 245, 245))
    d = ImageDraw.Draw(im)
    d.rectangle([40, 30, 160, 120], fill=(20, 20, 20))
    d.ellipse([70, 50, 130, 100], fill=(200, 200, 200))
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode()


def _frame(sid, fid="f-0", mode=None, gesture=None, extra=None):
    payload = {
        "sessionId": sid, "frameId": fid, "timestampMs": 1,
        "selectedRegion": REGION, "image_b64": _crop_b64(),
        "handLandmarks": [{"role": "index-tip", "x": 0.3, "y": 0.35}],
        "gesture": gesture if gesture is not None else {"type": "pinch", "active": True, "strength": 0.8},
    }
    if mode is not None:
        payload["workflowMode"] = mode
    if extra:
        payload.update(extra)
    return asyncio.run(bb.process_frame_async(payload))


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    bb.BUILD_SESSIONS.clear()
    monkeypatch.setenv("BUILD_SEGMENTATION_BACKEND", "fallback")
    monkeypatch.setenv("BUILD_MASK_OUTPUT", "contour")
    monkeypatch.setenv("BUILD_SEGMENT_EVERY_N", "3")
    monkeypatch.setenv("BUILD_SEGMENT_ON_EXTRACT", "true")
    yield
    bb.BUILD_SESSIONS.clear()


# -- workflowMode storage / resolution ----------------------------------------

def test_start_session_stores_workflow_mode():
    r = bb.start_session({"workflowMode": "plan"})
    assert r["workflow_mode"] == "plan"
    assert bb.BUILD_SESSIONS[r["session_id"]]["workflow_mode"] == "plan"


def test_start_session_defaults_to_build():
    assert bb.start_session({})["workflow_mode"] == "build"


def test_lock_session_accepts_workflow_mode():
    sid = bb.start_session({})["session_id"]
    r = bb.lock_session({"sessionId": sid, "selectedRegion": REGION, "workflowMode": "plan"})
    assert r["locked"] is True and r["workflow_mode"] == "plan"


def test_process_frame_returns_workflow_mode():
    sid = bb.start_session({"workflowMode": "plan"})["session_id"]
    bf = _frame(sid)["blueprint_frame"]
    assert bf["workflowMode"] == "plan"


def test_process_frame_returns_version_2():
    sid = bb.start_session({})["session_id"]
    bf = _frame(sid)["blueprint_frame"]
    assert bf["version"] == 2


def test_old_payload_without_workflow_mode_defaults_build():
    sid = bb.start_session({})["session_id"]
    payload = {"sessionId": sid, "frameId": "f-0", "selectedRegion": REGION, "image_b64": _crop_b64()}
    bf = asyncio.run(bb.process_frame_async(payload))["blueprint_frame"]
    assert bf["workflowMode"] == "build"
    # Old v1 fields are still present.
    for key in ("outline", "anchors", "sparsePoints", "handLandmarks", "stepMarkers", "gesture"):
        assert key in bf


# -- Fallback mask / contour --------------------------------------------------

def test_fallback_contour_populates_mask_source():
    sid = bb.start_session({})["session_id"]
    bf = _frame(sid)["blueprint_frame"]
    assert bf["maskSource"] == "fallback-contour"
    assert len(bf["maskContour"]) >= 3
    assert bf["outline"] == bf["maskContour"]  # outline mirrors the mask contour


# -- Build vs Plan ------------------------------------------------------------

def test_build_mode_returns_ainotes_and_instruction():
    sid = bb.start_session({"workflowMode": "build"})["session_id"]
    bf = _frame(sid)["blueprint_frame"]
    assert bf["instruction"]
    assert len(bf["aiNotes"]) >= 1
    assert bf["planSteps"] == []
    assert bf["currentPlanStepIndex"] is None


def test_plan_mode_returns_plansteps_and_next_action():
    intent = {"taskType": "build", "text": "assemble these", "confirmed": True}
    sid = bb.start_session({"workflowMode": "plan", "userIntent": intent})["session_id"]
    bf = _frame(sid, extra={"userIntent": intent})["blueprint_frame"]
    assert bf["nextAction"]
    assert len(bf["planSteps"]) == 3
    assert bf["planSteps"][0]["id"] == "step-1"
    assert bf["currentPlanStepIndex"] is not None


# -- Plan Mode: intent-driven guidance + overlays -----------------------------

def _plan_frame(intent, gesture=None, fid="f-0", sid=None):
    if sid is None:
        sid = bb.start_session({"workflowMode": "plan", "userIntent": intent})["session_id"]
    bf = _frame(sid, fid=fid, mode="plan", gesture=gesture, extra={"userIntent": intent})["blueprint_frame"]
    return sid, bf


def test_plan_unconfirmed_asks_to_confirm():
    intent = {"taskType": "build", "text": "assemble these", "confirmed": False}
    _, bf = _plan_frame(intent)
    assert bf["detectedIntent"] is None
    assert "task goal" in bf["instruction"].lower()
    assert len(bf["planOverlays"]) >= 1  # acceptance #3 -- overlays even when confirming
    assert any(o["type"] == "highlight" for o in bf["planOverlays"])


def test_plan_confirmed_build_uses_intent_and_overlays():
    intent = {"taskType": "build", "text": "I want to assemble these pieces", "confirmed": True}
    _, bf = _plan_frame(intent)
    assert bf["detectedIntent"] and "assemble" in bf["detectedIntent"].lower()  # acceptance #1
    assert bf["nextAction"] and bf["instruction"]                                # acceptance #2
    types = {o["type"] for o in bf["planOverlays"]}                              # acceptance #3/#4
    assert "target" in types
    assert any(o["type"] == "arrow" and "from" in o and "to" in o for o in bf["planOverlays"])


def test_plan_overlay_types_supported():
    intent = {"taskType": "inspect", "text": "inspect this", "confirmed": True}
    _, bf = _plan_frame(intent)
    allowed = {"arrow", "target", "ghost-position", "highlight", "warning-zone"}
    assert bf["planOverlays"] and all(o["type"] in allowed for o in bf["planOverlays"])
    assert any(o["type"] == "target" for o in bf["planOverlays"])  # numbered inspection points


def test_plan_high_risk_is_safety_first():
    intent = {"taskType": "repair", "text": "fix the electrical wiring outlet", "confirmed": True}
    _, bf = _plan_frame(intent)
    assert "isolate" in (bf["safetyWarning"] or "").lower()      # acceptance #7
    assert bf["importance"] == "high"
    assert any(o["type"] == "warning-zone" for o in bf["planOverlays"])
    assert bf["planSteps"][0]["title"].lower().startswith("isolate")


def test_plan_repair_without_symptom_asks_symptom():
    intent = {"taskType": "repair", "text": "fix it", "confirmed": True}
    _, bf = _plan_frame(intent)
    assert "symptom" in bf["instruction"].lower() or "problem" in bf["instruction"].lower()


def test_build_mode_has_no_plan_overlays():
    sid = bb.start_session({"workflowMode": "build"})["session_id"]
    bf = _frame(sid)["blueprint_frame"]
    assert bf["planOverlays"] == []          # acceptance #5 -- build still documents
    assert bf["detectedIntent"] is None
    assert len(bf["aiNotes"]) >= 1


def test_plan_intent_stored_at_start_used_per_frame():
    intent = {"taskType": "inspect", "text": "look at this", "confirmed": True}
    sid = bb.start_session({"workflowMode": "plan", "userIntent": intent})["session_id"]
    # frame WITHOUT userIntent in the payload -> uses the session-stored intent
    bf = asyncio.run(bb.process_frame_async(
        {"sessionId": sid, "frameId": "f-0", "selectedRegion": REGION, "image_b64": _crop_b64(),
         "workflowMode": "plan"}))["blueprint_frame"]
    assert bf["detectedIntent"] and "inspect" in bf["detectedIntent"].lower()
    assert any(o["type"] == "target" for o in bf["planOverlays"])


def test_replay_includes_plan_overlays():
    intent = {"taskType": "build", "text": "assemble", "confirmed": True}
    sid, _ = _plan_frame(intent)
    rep = bb.get_replay(sid)  # acceptance #6 -- replay still JSON keyframes
    assert all("planOverlays" in f for f in rep["frames"])


# -- SAM2 safe fallback -------------------------------------------------------

def test_sam2_unavailable_falls_back_safely(monkeypatch):
    monkeypatch.setenv("BUILD_SEGMENTATION_BACKEND", "sam2")
    sid = bb.start_session({})["session_id"]
    bf = _frame(sid)["blueprint_frame"]  # SAM2 not installed -> must not crash
    assert bf["maskSource"] in ("fallback-contour", "none")
    assert bf["version"] == 2


# -- Replay -------------------------------------------------------------------

def test_replay_returns_v2_frames():
    sid = bb.start_session({"workflowMode": "plan"})["session_id"]
    _frame(sid, "f-0")
    _frame(sid, "f-1")
    rep = bb.get_replay(sid)
    assert rep["frame_count"] == 2
    assert rep["workflow_mode"] == "plan"
    assert all(f["version"] == 2 for f in rep["frames"])
    assert rep["frames"][0]["workflowMode"] == "plan"
    # JSON only -- no raw image bytes stored in replay frames.
    assert "image_b64" not in rep["frames"][0]


# -- SAM2 segmentation (optional; safe fallback) ------------------------------

def _fake_sam2_ok(image_bgr, *, prompt_box=None, session=None, frame_index=0):
    return {"ok": True, "mask_source": "sam2", "mask_b64": None, "confidence": 0.85,
            "error": None, "mask_contour": [
                {"x": 0.20, "y": 0.30}, {"x": 0.55, "y": 0.28},
                {"x": 0.58, "y": 0.70}, {"x": 0.24, "y": 0.72}]}


def test_segment_crop_disabled_returns_fallback(monkeypatch):
    import numpy as np
    import build_segmentation
    monkeypatch.setenv("BUILD_SEGMENTATION_BACKEND", "fallback")
    out = build_segmentation.segment_crop(np.zeros((64, 64, 3), dtype=np.uint8))
    assert out["ok"] is False and out["mask_source"] == "fallback-contour"


def test_segment_crop_sam2_unavailable_returns_fallback(monkeypatch):
    import numpy as np
    import build_segmentation
    monkeypatch.setenv("BUILD_SEGMENTATION_BACKEND", "sam2")
    build_segmentation._SAM2_STATE.update({"loaded": False, "predictor": None, "error": None})
    out = build_segmentation.segment_crop(np.zeros((64, 64, 3), dtype=np.uint8), frame_index=0)
    assert out["ok"] is False
    assert out["mask_source"] == "fallback-contour"
    assert out["error"]  # e.g. "sam2_unavailable"


def test_frame_falls_back_when_sam2_missing(monkeypatch):
    monkeypatch.setenv("BUILD_SEGMENTATION_BACKEND", "sam2")  # no package installed
    sid = bb.start_session({})["session_id"]
    r = _frame(sid)
    assert r["ok"] is True                      # acceptance #5 -- never 500 on missing SAM2
    bf = r["blueprint_frame"]
    assert bf["version"] == 2
    assert bf["maskSource"] in ("fallback-contour", "none")


def test_sam2_success_sets_masksource_contour_and_outline(monkeypatch):
    import build_segmentation
    monkeypatch.setenv("BUILD_SEGMENTATION_BACKEND", "sam2")
    monkeypatch.setattr(build_segmentation, "segment_crop", _fake_sam2_ok)
    sid = bb.start_session({})["session_id"]
    bf = _frame(sid)["blueprint_frame"]
    assert bf["maskSource"] == "sam2"                         # acceptance #6
    assert len(bf["maskContour"]) >= 3                         # acceptance #7
    for p in bf["maskContour"]:                                # normalized 0..1
        assert 0.0 <= p["x"] <= 1.0 and 0.0 <= p["y"] <= 1.0
    assert bf["outline"] == bf["maskContour"]                  # outline uses the mask
    assert len(bf["anchors"]) >= 1                             # anchors from mask geometry


def test_sam2_plan_mode_still_guides(monkeypatch):
    import build_segmentation
    monkeypatch.setenv("BUILD_SEGMENTATION_BACKEND", "sam2")
    monkeypatch.setattr(build_segmentation, "segment_crop", _fake_sam2_ok)
    intent = {"taskType": "inspect", "text": "inspect this", "confirmed": True}
    sid = bb.start_session({"workflowMode": "plan", "userIntent": intent})["session_id"]
    bf = _frame(sid, mode="plan", extra={"userIntent": intent})["blueprint_frame"]
    assert bf["maskSource"] == "sam2"
    assert len(bf["planSteps"]) >= 1 and bf["planOverlays"]    # acceptance #8/#10


def test_detect_unaffected_by_build_mode():
    from fastapi.testclient import TestClient
    import server
    server._start_warmup_background = lambda: None  # avoid spawning a warmup thread
    client = TestClient(server.app)
    img = base64.b64encode(b"\xff\xd8\xff\xe0jpg").decode()
    r = client.post("/detect", json={"image_b64": img})
    assert r.status_code == 503 and r.json()["error"] == "model_not_ready"  # acceptance #1
