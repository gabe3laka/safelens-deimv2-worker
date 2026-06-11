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
    sid = bb.start_session({"workflowMode": "plan"})["session_id"]
    bf = _frame(sid)["blueprint_frame"]
    assert bf["nextAction"]
    assert len(bf["planSteps"]) == 3
    assert bf["planSteps"][0]["id"] == "step-1"
    assert bf["currentPlanStepIndex"] is not None


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
