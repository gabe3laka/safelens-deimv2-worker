"""
tests/test_plan_context.py -- selected-crop Plan context + virtualBlueprintPoints.

CPU-only, no YOLO/depth models needed. Drives /build/session/frame through the
real HTTP route (geom comes from the fallback Canny contour when YOLO26 is not
warmed) and unit-tests the rule-based plan_context helpers directly.
"""

from __future__ import annotations

import base64
import io
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("cv2")
pytest.importorskip("PIL")

import build_blueprint as bb
import plan_context
from PIL import Image, ImageDraw

REGION = {"x": 0.1, "y": 0.2, "w": 0.4, "h": 0.3}


def _crop_b64():
    im = Image.new("RGB", (220, 160), (245, 245, 245))
    d = ImageDraw.Draw(im)
    d.rectangle([40, 30, 180, 130], fill=(20, 20, 20))
    d.ellipse([70, 50, 140, 110], fill=(200, 200, 200))
    buf = io.BytesIO(); im.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode()


def _plan_frame(intent=None, mode="plan", selected_label=None, gesture=None):
    import asyncio
    sid = bb.start_session({"workflowMode": mode,
                            **({"userIntent": intent} if intent else {})})["session_id"]
    payload = {"sessionId": sid, "frameId": "f-0", "selectedRegion": REGION,
               "image_b64": _crop_b64(), "workflowMode": mode,
               "handLandmarks": [{"role": "index-tip", "x": 0.3, "y": 0.35}],
               "gesture": gesture if gesture is not None else {"type": "pinch", "active": True}}
    if intent:
        payload["userIntent"] = intent
    if selected_label:
        payload["selectedLabel"] = selected_label
    return asyncio.run(bb.process_frame_async(payload))["blueprint_frame"]


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    bb.BUILD_SESSIONS.clear()
    monkeypatch.setenv("BUILD_SEGMENTATION_BACKEND", "fallback")
    monkeypatch.setenv("PLAN_CONTEXT_ENABLED", "true")
    monkeypatch.setenv("PLAN_DEPTH_ENABLED", "false")
    plan_context._DEPTH_STATE.update(attempted=False, model=None)
    yield
    bb.BUILD_SESSIONS.clear()


# ── Frame-level (Plan / Build) ────────────────────────────────────────────────

def test_build_mode_still_works():
    bf = _plan_frame(mode="build")
    assert bf["version"] == 2 and bf["workflowMode"] == "build"
    assert bf["planOverlays"] == []
    # Build Mode must not run depth / plan context heavy work.
    assert bf["depthPoints"] == [] and bf["virtualBlueprintPoints"] == []


def test_plan_returns_selected_label_and_context():
    bf = _plan_frame(selected_label="arduino board")
    assert bf["selectedLabel"]
    assert bf["planContext"] is not None
    assert bf["planContext"]["contextSource"] in ("yolo26", "rules")
    assert "objectCount" in bf["planContext"]
    assert bf["reasoningSource"] == "rules"


def test_plan_cropentities_and_cropsegments_present_keys():
    bf = _plan_frame()
    # keys always present (empty when YOLO26 isn't warmed -- the fallback path)
    assert "cropEntities" in bf and isinstance(bf["cropEntities"], list)
    assert "cropSegments" in bf and isinstance(bf["cropSegments"], list)


def test_plan_without_intent_returns_suggested_goals():
    bf = _plan_frame()  # no userIntent
    assert len(bf["suggestedGoals"]) >= 3
    assert bf["detectedIntent"] == "Waiting for goal"
    assert "tell me" in bf["nextAction"].lower()


def test_plan_returns_rule_based_virtual_points():
    bf = _plan_frame()
    vps = bf["virtualBlueprintPoints"]
    assert len(vps) >= 1
    roles = {p["role"] for p in vps}
    assert "anchor" in roles
    assert any(p["id"] == "vp-main-center" for p in vps)


def test_virtual_points_clamped_and_capped():
    bf = _plan_frame(intent={"taskType": "build", "text": "assemble", "confirmed": True})
    vps = bf["virtualBlueprintPoints"]
    assert len(vps) <= plan_context.MAX_VIRTUAL_POINTS
    for p in vps:
        assert 0.0 <= p["x"] <= 1.0 and 0.0 <= p["y"] <= 1.0


def test_depth_disabled_returns_no_points():
    bf = _plan_frame()
    assert bf["depthPoints"] == []
    assert bf["depthSource"] == "none"
    assert bf["depthWarning"] is None


def test_depth_enabled_but_unavailable_warns(monkeypatch):
    monkeypatch.setenv("PLAN_DEPTH_ENABLED", "true")
    monkeypatch.setenv("PLAN_DEPTH_BACKEND", "depth-anything-v2")
    plan_context._DEPTH_STATE.update(attempted=False, model=None)
    bf = _plan_frame()
    assert bf["depthPoints"] == []
    assert bf["depthSource"] == "depth-anything-v2"
    assert bf["depthWarning"] == "depth backend configured but model unavailable"


# ── plan_context unit helpers ─────────────────────────────────────────────────

def test_electronics_goals():
    goals = plan_context.suggested_goals("pcb board", [{"label": "pcb"}], electronics=True)
    assert "Locate connector points" in goals
    assert "Check safety before powering" in goals


def test_non_electronics_single_vs_multi_goals():
    one = plan_context.suggested_goals("cup", [{"label": "cup"}], electronics=False)
    assert "Identify this item" in one
    multi = plan_context.suggested_goals(None, [{"label": "a"}, {"label": "b"}], electronics=False)
    assert "Help assemble these pieces" in multi


def test_is_electronics_detects_keywords():
    assert plan_context.is_electronics("Arduino Uno", [], "")
    assert plan_context.is_electronics(None, [{"label": "connector"}], "")
    assert plan_context.is_electronics(None, [], "fix the wiring")
    assert not plan_context.is_electronics("wooden chair", [{"label": "chair"}], "")


def test_selected_label_priority():
    ents = [{"label": "screw", "confidence": 0.4}, {"label": "pcb", "confidence": 0.9}]
    assert plan_context.selected_label(ents, None, None) == "pcb"
    assert plan_context.selected_label([], "my hint", None) == "my hint"
    assert plan_context.selected_label([], None, "yolo26-seg") == "selected object"
    assert plan_context.selected_label([], None, None) == "selected item"


def test_electronics_virtual_points_and_safety():
    geom = {"outline": [{"x": 0.1, "y": 0.1}, {"x": 0.9, "y": 0.1},
                        {"x": 0.9, "y": 0.6}, {"x": 0.1, "y": 0.6}],
            "center": {"x": 0.5, "y": 0.35}, "detected_parts": []}
    pts = plan_context.virtual_blueprint_points(geom, electronics=True, safety=True)
    roles = {p["role"] for p in pts}
    assert "connection-point" in roles
    assert "warning-point" in roles
    assert len(pts) <= plan_context.MAX_VIRTUAL_POINTS


def test_depth_downsample_to_sample_points():
    pts = [{"x": i / 500.0, "y": 0.5} for i in range(500)]
    out = plan_context._downsample(pts, 120)
    assert len(out) == 120
    assert plan_context._downsample(pts[:50], 120) == pts[:50]  # fewer than n -> unchanged


def test_optional_paths_disabled_noop():
    assert plan_context.open_vocab_parts({}) == []        # open-vocab disabled
    assert plan_context.known_part_pose({}) is None        # FoundationPose disabled
    assert plan_context.assembly_state({}) is None         # assembly-state disabled


def test_config_reports_plan_context():
    cfg = plan_context.config()
    for key in ("enabled", "depth_enabled", "depth_backend", "depth_model_loaded",
                "depth_sample_points", "depth_every_n", "open_vocab_enabled",
                "known_part_pose_enabled", "assembly_state_enabled"):
        assert key in cfg, key
    assert cfg["enabled"] is True and cfg["depth_enabled"] is False


def test_build_safety_warning_for_electronics_intent():
    bf = _plan_frame(intent={"taskType": "repair", "text": "solder the battery wires",
                             "confirmed": True})
    # electronics + power/solder context -> a safety warning is present
    assert bf["safetyWarning"] and (
        "unpowered" in bf["safetyWarning"].lower() or "isolate" in bf["safetyWarning"].lower())
    assert bf["planContext"]["warnings"]
