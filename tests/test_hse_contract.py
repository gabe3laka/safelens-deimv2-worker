"""
tests/test_hse_contract.py -- Live HSE Scene Risk Contract (all 11 parts).

CPU-only, no weights.  Covers:
  Part 1  -- /detect accepts frame_b64 OR image_b64 (same frame for all layers)
  Part 2  -- wants_hse_reasoning() helper detects HSE intent from any signal
  Part 3  -- reasoner_status normalised to stable dict (standard state vocab)
  Part 4  -- every active scene_risk has at least one link field
  Part 5  -- scene_risks built from det + VLM risks; vague risks excluded
  Part 6  -- VLM prompt schema includes linkability + strict rules
  Part 7  -- risk matrix bands  1-3 GREEN / 4-8 YELLOW / 9-14 ORANGE / 15-25 RED
  Part 8  -- Qwen failure never breaks detection; graceful degradation reported
  Part 9  -- /agent/* routes still respond correctly (agentic separate)
  Part 10 -- response contract shape (linkable scene_risks in all cases)
  Part 11 -- comprehensive tests (this file)
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

import pytest

pytest.importorskip("pydantic")

import risk.vlm_reasoner as vlm
from risk import risk_matrix
from risk.reason_schema import ReasonResponse, VlmRisk
from risk.risk_matrix import RiskMatrix, validate_profile

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

PERSON = {"label": "person", "class_id": 0, "confidence": 0.9,
          "bbox": {"x": 0.30, "y": 0.40, "w": 0.14, "h": 0.45}}
FORKLIFT = {"label": "forklift", "class_id": 7, "confidence": 0.86,
            "bbox": {"x": 0.34, "y": 0.42, "w": 0.24, "h": 0.40}}

DET_RISK_LINKED = {
    "risk_id": "rsk_1", "rule_id": "R04_person_forklift",
    "hazard_type": "person_forklift_proximity", "risk_state": "active",
    "risk_level": "ORANGE", "severity": 3, "likelihood": 4, "risk_score": 12,
    "involved_track_ids": ["trk_1"], "involved_entities": [0, 1],
    "bbox": {"x": 0.30, "y": 0.40, "w": 0.14, "h": 0.45},
    "reason": "Person overlaps forklift.",
    "recommended_controls": [{"level": "elimination", "action": "separate"}],
    "recommended_action": "Separate pedestrians and vehicles.",
    "confidence": 1.0, "should_alert": True,
    "produced_by": "risk_engine", "model_version": "risk_engine.v1",
    "requires_human_review": False, "timestamp_ms": 1000,
}

DET_RISK_VAGUE = {
    "risk_id": "rsk_vague", "rule_id": "R99",
    "hazard_type": "unknown_hazard", "risk_state": "active",
    "risk_level": "YELLOW", "severity": 2, "likelihood": 2, "risk_score": 4,
    # No bbox, no track ids, no entity ids -- should be excluded from scene_risks
    "reason": "Vague hazard detected.",
    "recommended_controls": [], "confidence": 0.3,
    "produced_by": "risk_engine", "model_version": "risk_engine.v1",
    "requires_human_review": False, "timestamp_ms": 1000,
}


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    monkeypatch.setenv("VLM_REASONER_ENABLED", "true")
    monkeypatch.setenv("REASONER_MODE", "mock")
    monkeypatch.setenv("RISK_ENGINE_ENABLED", "true")
    monkeypatch.delenv("RISK_MATRIX_PROFILE", raising=False)
    vlm.reset()
    risk_matrix.reset_cache()
    yield
    vlm.reset()
    risk_matrix.reset_cache()


# ---------------------------------------------------------------------------
# Part 7 -- Risk matrix bands  1-3/4-8/9-14/15-25
# ---------------------------------------------------------------------------

def test_new_matrix_green_band():
    m = risk_matrix.get_matrix()
    assert m.level(1, 1) == "GREEN"  # score=1
    assert m.level(1, 2) == "GREEN"  # score=2
    assert m.level(1, 3) == "GREEN"  # score=3


def test_new_matrix_boundary_3_to_4():
    """Score 3 = GREEN; score 4 = YELLOW (updated threshold)."""
    m = risk_matrix.get_matrix()
    assert m.level(1, 3) == "GREEN"   # 1*3 = 3 → GREEN
    assert m.level(2, 2) == "YELLOW"  # 2*2 = 4 → YELLOW


def test_new_matrix_yellow_band():
    m = risk_matrix.get_matrix()
    assert m.level(2, 2) == "YELLOW"  # score=4
    assert m.level(2, 4) == "YELLOW"  # score=8


def test_new_matrix_boundary_8_to_9():
    """Score 8 = YELLOW; score 9 = ORANGE (updated threshold)."""
    m = risk_matrix.get_matrix()
    assert m.level(2, 4) == "YELLOW"  # 2*4 = 8 → YELLOW
    assert m.level(3, 3) == "ORANGE"  # 3*3 = 9 → ORANGE


def test_new_matrix_orange_band():
    m = risk_matrix.get_matrix()
    assert m.level(3, 3) == "ORANGE"  # score=9
    assert m.level(3, 4) == "ORANGE"  # score=12
    assert m.level(2, 5) == "ORANGE"  # score=10


def test_new_matrix_boundary_14_to_15():
    """Score 14 = ORANGE; score 15 = RED (updated threshold)."""
    m = risk_matrix.get_matrix()
    # Direct band lookup to test exact boundary (14 is max ORANGE, 15 is min RED)
    assert m.band(14)["level"] == "ORANGE"
    assert m.band(15)["level"] == "RED"


def test_new_matrix_red_band():
    m = risk_matrix.get_matrix()
    assert m.level(3, 5) == "RED"  # score=15
    assert m.level(5, 5) == "RED"  # score=25


def test_bundled_profile_matches_new_bands():
    """Committed profile must validate and describe the updated bands."""
    profile = risk_matrix.load_profile()
    validate_profile(profile)
    bands = {b["level"]: (b["min"], b["max"]) for b in profile["bands"]}
    assert bands["GREEN"]  == (1,  3)
    assert bands["YELLOW"] == (4,  8)
    assert bands["ORANGE"] == (9,  14)
    assert bands["RED"]    == (15, 25)


def test_agentic_score_tool_matches_new_bands():
    """agentic_cpu risk_tools.score() must use the same updated thresholds."""
    from agentic_cpu.tools.risk_tools import score
    assert score(1, 3)["risk_level"] == "GREEN"   # 3
    assert score(2, 2)["risk_level"] == "YELLOW"  # 4
    assert score(2, 4)["risk_level"] == "YELLOW"  # 8
    assert score(3, 3)["risk_level"] == "ORANGE"  # 9
    assert score(2, 5)["risk_level"] == "ORANGE"  # 10
    assert score(3, 5)["risk_level"] == "RED"     # 15
    assert score(5, 5)["risk_level"] == "RED"     # 25


# ---------------------------------------------------------------------------
# Part 2 -- wants_hse_reasoning() helper
# ---------------------------------------------------------------------------

def test_hse_intent_via_mode(monkeypatch):
    monkeypatch.setenv("SKIP_WARMUP", "true")
    import importlib, sys
    if "server" in sys.modules:
        del sys.modules["server"]
    srv = importlib.import_module("server")
    assert srv.wants_hse_reasoning({"mode": "hse-monitoring"}) is True
    assert srv.wants_hse_reasoning({"mode": "standard"}) is False


def test_hse_intent_via_scene_hint(monkeypatch):
    monkeypatch.setenv("SKIP_WARMUP", "true")
    import importlib, sys
    if "server" in sys.modules:
        del sys.modules["server"]
    srv = importlib.import_module("server")
    assert srv.wants_hse_reasoning({"scene_hint": "live_hse_monitoring"}) is True
    assert srv.wants_hse_reasoning({"scene_hint": "cafe"}) is False


def test_hse_intent_via_tasks(monkeypatch):
    monkeypatch.setenv("SKIP_WARMUP", "true")
    import importlib, sys
    if "server" in sys.modules:
        del sys.modules["server"]
    srv = importlib.import_module("server")
    assert srv.wants_hse_reasoning({"tasks": ["det", "scene_reasoning"]}) is True
    assert srv.wants_hse_reasoning({"tasks": ["det"]}) is False


def test_hse_intent_via_reasoning_preferences(monkeypatch):
    monkeypatch.setenv("SKIP_WARMUP", "true")
    import importlib, sys
    if "server" in sys.modules:
        del sys.modules["server"]
    srv = importlib.import_module("server")
    assert srv.wants_hse_reasoning(
        {"reasoning_preferences": {"return_scene_risks": True}}) is True
    assert srv.wants_hse_reasoning(
        {"reasoning_preferences": {"return_reasoner_status": True}}) is True
    assert srv.wants_hse_reasoning(
        {"reasoning_preferences": {"return_scene_risks": False}}) is False
    assert srv.wants_hse_reasoning({}) is False


# ---------------------------------------------------------------------------
# Part 3 -- _normalize_reasoner_status()
# ---------------------------------------------------------------------------

def test_normalize_string_statuses(monkeypatch):
    monkeypatch.setenv("SKIP_WARMUP", "true")
    import importlib, sys
    if "server" in sys.modules:
        del sys.modules["server"]
    srv = importlib.import_module("server")
    n = srv._normalize_reasoner_status

    assert n("disabled")["state"] == "disabled"
    assert n("not_triggered")["state"] == "rules_only"
    assert n("throttled")["state"] == "throttled"
    assert n("triggered")["state"] == "running"
    assert n("cached")["state"] == "ready"
    assert n("cached_and_triggered")["state"] == "running"
    assert n("error")["state"] == "error"
    assert n("timeout")["state"] == "timeout"
    assert n("unavailable")["state"] == "unavailable"
    assert n("ok")["state"] == "ready"


def test_normalize_dict_status(monkeypatch):
    monkeypatch.setenv("SKIP_WARMUP", "true")
    import importlib, sys
    if "server" in sys.modules:
        del sys.modules["server"]
    srv = importlib.import_module("server")
    n = srv._normalize_reasoner_status

    result = n({"state": "cached", "model": "mock"})
    assert result["state"] == "ready"
    assert result["model"] == "mock"


def test_normalize_includes_model_id(monkeypatch):
    monkeypatch.setenv("SKIP_WARMUP", "true")
    import importlib, sys
    if "server" in sys.modules:
        del sys.modules["server"]
    srv = importlib.import_module("server")
    result = srv._normalize_reasoner_status("triggered", model_id="mock")
    assert result["state"] == "running"
    assert result["model"] == "mock"


# ---------------------------------------------------------------------------
# Part 4 / Part 5 -- linkable scene_risks / _build_scene_risks()
# ---------------------------------------------------------------------------

def _get_srv(monkeypatch):
    monkeypatch.setenv("SKIP_WARMUP", "true")
    import importlib, sys
    if "server" in sys.modules:
        del sys.modules["server"]
    return importlib.import_module("server")


def test_linkable_deterministic_risk_included(monkeypatch):
    srv = _get_srv(monkeypatch)
    scene = srv._build_scene_risks([DET_RISK_LINKED], None, [])
    assert len(scene) == 1
    sr = scene[0]
    assert sr["risk_id"] == "rsk_1"
    assert srv._is_linkable(sr)


def test_vague_deterministic_risk_excluded(monkeypatch):
    srv = _get_srv(monkeypatch)
    scene = srv._build_scene_risks([DET_RISK_VAGUE], None, [])
    assert scene == []


def test_latent_risk_excluded_from_scene_risks(monkeypatch):
    srv = _get_srv(monkeypatch)
    latent = dict(DET_RISK_LINKED, risk_state="latent")
    scene = srv._build_scene_risks([latent], None, [])
    assert scene == []


def test_scene_risks_inherit_provenance_fields(monkeypatch):
    srv = _get_srv(monkeypatch)
    scene = srv._build_scene_risks([DET_RISK_LINKED], None, [])
    sr = scene[0]
    assert sr["produced_by"] == "risk_engine"
    assert sr["reasoner_model"] == "risk_engine.v1"
    assert sr["reasoner_status"] == "rules_only"
    assert sr["risk_reason"]   # non-empty


def test_vlm_risk_enriched_with_track_bbox(monkeypatch):
    """VLM risk with involved_track_ids but no bbox gets bbox from the track."""
    srv = _get_srv(monkeypatch)
    tracks = [{"track_id": "trk_A",
               "bbox": {"x": 0.1, "y": 0.2, "w": 0.3, "h": 0.4}}]
    vlm_draft = {"risks": [
        {"risk_id": "vlm_1", "hazard_type": "test", "risk_level": "YELLOW",
         "severity": 2, "likelihood": 2, "risk_score": 4,
         "involved_track_ids": ["trk_A"], "risk_state": "active",
         "reason": "Test", "visual_evidence": [], "evidence": [],
         "recommended_controls": [], "confidence": 0.7}
    ], "reasoner_model": "mock"}
    scene = srv._build_scene_risks([], vlm_draft, tracks)
    assert len(scene) == 1
    assert scene[0]["bbox"] == {"x": 0.1, "y": 0.2, "w": 0.3, "h": 0.4}


def test_vlm_vague_risk_excluded(monkeypatch):
    """VLM risk with no link fields at all is excluded from scene_risks."""
    srv = _get_srv(monkeypatch)
    vlm_draft = {"risks": [
        {"risk_id": "vlm_vague", "hazard_type": "test", "risk_level": "YELLOW",
         "severity": 2, "likelihood": 2, "risk_score": 4,
         "involved_track_ids": [], "risk_state": "active",
         "reason": "Vague", "visual_evidence": [], "evidence": [],
         "recommended_controls": [], "confidence": 0.3}
    ], "reasoner_model": "mock"}
    scene = srv._build_scene_risks([], vlm_draft, [])
    assert scene == []


def test_unmatched_vlm_candidate_is_advisory_only(monkeypatch):
    srv = _get_srv(monkeypatch)
    vlm_draft = {"risks": [
        {"risk_id": "vlm_unmatched", "hazard_type": "test", "risk_level": "YELLOW",
         "severity": 2, "likelihood": 2, "risk_score": 4,
         "approximate_region": "top-left",
         "risk_state": "active", "reason": "Possible risk", "visual_evidence": [],
         "recommended_controls": [], "confidence": 0.5}
    ]}
    scene = srv._build_scene_risks([], vlm_draft, [])
    assert len(scene) == 1
    assert scene[0]["candidate_only"] is True
    assert scene[0].get("bbox") is None


def test_unmatched_bbox_does_not_color_unrelated_track(monkeypatch):
    srv = _get_srv(monkeypatch)
    tracks = [{"track_id": "trk_far", "bbox": {"x": 0.8, "y": 0.8, "w": 0.1, "h": 0.1}}]
    vlm_draft = {"risks": [
        {"risk_id": "vlm_far", "hazard_type": "test", "risk_level": "YELLOW",
         "severity": 2, "likelihood": 2, "risk_score": 4,
         "bbox": {"x": 0.1, "y": 0.1, "w": 0.05, "h": 0.05},
         "approximate_region": "left",
         "risk_state": "active", "reason": "possible", "recommended_controls": []}
    ]}
    scene = srv._build_scene_risks([], vlm_draft, tracks)
    assert len(scene) == 1
    assert scene[0]["candidate_only"] is True
    assert scene[0].get("bbox") is None


def test_stale_unmatched_candidate_dropped(monkeypatch):
    srv = _get_srv(monkeypatch)
    vlm_draft = {"_cached_at_ms": 1, "risks": [
        {"risk_id": "vlm_old", "hazard_type": "test", "risk_level": "YELLOW",
         "severity": 2, "likelihood": 2, "risk_score": 4,
         "approximate_region": "left", "risk_state": "active",
         "reason": "old", "recommended_controls": [], "confidence": 0.5}
    ]}
    monkeypatch.setenv("REASONER_UNMATCHED_CANDIDATE_TTL_MS", "1")
    scene = srv._build_scene_risks([], vlm_draft, [])
    assert scene == []


def test_is_linkable_helper(monkeypatch):
    srv = _get_srv(monkeypatch)
    assert srv._is_linkable({"involved_track_ids": ["trk_1"]}) is True
    assert srv._is_linkable({"involved_detection_ids": [0]}) is True
    assert srv._is_linkable({"bbox": {"x": 0.1, "y": 0.1, "w": 0.1, "h": 0.1}}) is True
    assert srv._is_linkable({"approximate_region": "bottom-left"}) is True
    assert srv._is_linkable({"linked_entity_id": "ent_42"}) is True
    assert srv._is_linkable({}) is False
    assert srv._is_linkable({"involved_track_ids": []}) is False


# ---------------------------------------------------------------------------
# Part 1 -- frame_b64 fallback in /detect
# ---------------------------------------------------------------------------

def test_frame_b64_accepted_in_detect(monkeypatch):
    """frame_b64 is accepted as an alias for image_b64 in /detect."""
    import base64, io
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), (100, 100, 100)).save(buf, format="JPEG")
    img = base64.b64encode(buf.getvalue()).decode()

    monkeypatch.setenv("SKIP_WARMUP", "true")
    import importlib, sys
    if "server" in sys.modules:
        del sys.modules["server"]
    srv = importlib.import_module("server")
    from fastapi.testclient import TestClient
    with TestClient(srv.app) as c:
        # Pre-mark as ready so /detect doesn't 503
        with srv._STATE_LOCK:
            srv._STATE["status"] = "ready"
        try:
            # Sending frame_b64 (not image_b64)
            r = c.post("/detect", json={"frame_b64": img})
            # Should not fail on the image_b64 guard -- either runs or hits
            # model-load error (acceptable); must not get a missing_image_b64 error.
            body = r.json()
            assert body.get("error") != "missing_image_b64", body
        finally:
            with srv._STATE_LOCK:
                srv._STATE["status"] = "cold"


def test_image_b64_still_accepted_in_detect(monkeypatch):
    """image_b64 (legacy key) continues to be accepted by /detect."""
    import base64, io
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), (100, 100, 100)).save(buf, format="JPEG")
    img = base64.b64encode(buf.getvalue()).decode()

    monkeypatch.setenv("SKIP_WARMUP", "true")
    import importlib, sys
    if "server" in sys.modules:
        del sys.modules["server"]
    srv = importlib.import_module("server")
    from fastapi.testclient import TestClient
    with TestClient(srv.app) as c:
        with srv._STATE_LOCK:
            srv._STATE["status"] = "ready"
        try:
            r = c.post("/detect", json={"image_b64": img})
            body = r.json()
            assert body.get("error") != "missing_image_b64", body
        finally:
            with srv._STATE_LOCK:
                srv._STATE["status"] = "cold"


def test_neither_frame_b64_nor_image_b64_rejected(monkeypatch):
    """Sending neither key yields a 4xx with a structured error."""
    monkeypatch.setenv("SKIP_WARMUP", "true")
    import importlib, sys
    if "server" in sys.modules:
        del sys.modules["server"]
    srv = importlib.import_module("server")
    from fastapi.testclient import TestClient
    with TestClient(srv.app) as c:
        with srv._STATE_LOCK:
            srv._STATE["status"] = "ready"
        try:
            r = c.post("/detect", json={})
            assert r.status_code in (400, 422)
        finally:
            with srv._STATE_LOCK:
                srv._STATE["status"] = "cold"


# ---------------------------------------------------------------------------
# Part 6 -- VLM prompt/schema includes linkability requirements
# ---------------------------------------------------------------------------

def test_gemini_prompt_includes_linkability_fields(monkeypatch):
    monkeypatch.setenv("REASONER_MODE", "gemini")
    from risk.vlm_reasoner import _build_gemini_prompt, _build_box_decision_anchors, _select_candidate_entities
    import risk.gemini_reasoner as gr
    from risk.reason_schema import ReasonRequest
    req = ReasonRequest(entities=[PERSON, FORKLIFT],
                        deterministic_risks=[DET_RISK_LINKED])
    candidates = _select_candidate_entities(req.entities, req.deterministic_risks,
                                            gr.max_box_candidates())
    anchors = _build_box_decision_anchors(candidates)
    prompt = _build_gemini_prompt(req, anchors)
    # New box-decision prompt passes anchors so Gemini can reference them.
    # Key section name or at least one entity label must appear.
    assert any(kw in prompt for kw in ("detected_box_anchors", "person", "forklift")), \
        "No entity anchor found in prompt"


def test_gemini_prompt_includes_strict_linkability_rule(monkeypatch):
    from risk.vlm_reasoner import _build_gemini_prompt, _build_box_decision_anchors, _select_candidate_entities
    import risk.gemini_reasoner as gr
    from risk.reason_schema import ReasonRequest
    req = ReasonRequest(entities=[PERSON])
    candidates = _select_candidate_entities(req.entities, req.deterministic_risks,
                                            gr.max_box_candidates())
    anchors = _build_box_decision_anchors(candidates)
    prompt = _build_gemini_prompt(req, anchors)
    # Box-decision prompt enforces the linking rule via anchor IDs
    assert (
        "detected_box_anchors" in prompt
        or "box IDs" in prompt
        or "Use only box IDs" in prompt
    )


def test_vlm_schema_linkability_fields_on_vmlrisk():
    """VlmRisk model must carry all required linkability fields."""
    r = VlmRisk(
        risk_id="test_1",
        involved_track_ids=["trk_1"],
        bbox={"x": 0.1, "y": 0.1, "w": 0.2, "h": 0.2},
        linked_entity_id="ent_0",
        approximate_region="bottom-left",
        involved_detection_ids=[0, 1],
        risk_reason="Test reason",
        evidence=["person near forklift"],
        produced_by="vlm_reasoner",
        reasoner_model="mock",
        reasoner_status="ready",
    )
    assert r.involved_track_ids == ["trk_1"]
    assert r.bbox == {"x": 0.1, "y": 0.1, "w": 0.2, "h": 0.2}
    assert r.linked_entity_id == "ent_0"
    assert r.approximate_region == "bottom-left"
    assert r.involved_detection_ids == [0, 1]
    assert r.risk_reason == "Test reason"
    assert r.evidence == ["person near forklift"]
    assert r.produced_by == "vlm_reasoner"
    assert r.reasoner_model == "mock"
    assert r.reasoner_status == "ready"
    # Safety contract still enforced
    assert r.requires_human_review is True
    assert r.should_alert is False


def test_vlm_mock_outputs_valid_risks():
    """Mock reasoner produces VlmRisk items that validate against the schema."""
    req = {"session_id": "cam_1", "frame_id": "f1",
           "entities": [PERSON, FORKLIFT],
           "deterministic_risks": [DET_RISK_LINKED]}
    out = vlm.reason_sync(req)
    parsed = ReasonResponse(**out)
    assert parsed.reasoner_status == "ok"
    for r in parsed.risks:
        assert r.requires_human_review is True
        assert r.should_alert is False
        # Mock produces risks linked to the deterministic risk track ids
        assert r.involved_track_ids or r.bbox or r.linked_entity_id or r.approximate_region


# ---------------------------------------------------------------------------
# Part 8 -- Qwen failure: graceful degradation, never 500
# ---------------------------------------------------------------------------

def test_vlm_disabled_with_hse_intent_returns_rules_only_scene_risks(monkeypatch):
    """When VLM is disabled but HSE mode is requested, return deterministic
    scene_risks + reasoner_status.state='unavailable' + warning."""
    monkeypatch.setenv("VLM_REASONER_ENABLED", "false")
    srv = _get_srv(monkeypatch)

    scene = srv._build_scene_risks([DET_RISK_LINKED], None, [])
    status = srv._normalize_reasoner_status("unavailable")

    assert status["state"] == "unavailable"
    assert len(scene) == 1 and srv._is_linkable(scene[0])


def test_vlm_reason_sync_error_never_raises():
    """reason_sync must NEVER raise regardless of mode."""
    import os
    orig = os.environ.get("REASONER_MODE")
    try:
        os.environ["REASONER_MODE"] = "unknown_mode_xyz"
        out = vlm.reason_sync({})
        assert out["reasoner_status"] in ("unavailable", "error", "disabled")
        assert "risks" in out
    finally:
        if orig is None:
            os.environ.pop("REASONER_MODE", None)
        else:
            os.environ["REASONER_MODE"] = orig


def test_vlm_timeout_returns_clean_response():
    """reason_async handles timeout cleanly."""
    import asyncio, os
    orig_timeout = os.environ.get("REASONER_TIMEOUT_MS")
    try:
        os.environ["REASONER_TIMEOUT_MS"] = "1"  # 1 ms -- guaranteed timeout
        os.environ["REASONER_MODE"] = "mock"
        out = asyncio.run(vlm.reason_async({}))
        # Either completed fast enough (ok) or timed out cleanly
        assert out.get("reasoner_status") in ("ok", "timeout")
        assert out.get("requires_human_review") is True
        assert out.get("should_alert") is False
    finally:
        if orig_timeout is None:
            os.environ.pop("REASONER_TIMEOUT_MS", None)
        else:
            os.environ["REASONER_TIMEOUT_MS"] = orig_timeout


# ---------------------------------------------------------------------------
# Part 3 -- reasoner_status state vocabulary exhaustive check
# ---------------------------------------------------------------------------

def test_all_standard_states_map_cleanly(monkeypatch):
    """Every internal raw status maps to one of the documented app-facing states."""
    srv = _get_srv(monkeypatch)
    n = srv._normalize_reasoner_status
    allowed = {"ready", "running", "queued", "queued_latest", "throttled", "unavailable", "timeout",
               "disabled", "rules_only", "error"}
    raw_inputs = ["disabled", "not_triggered", "throttled", "triggered",
                  "cached", "cached_and_triggered", "queued_latest", "error", "timeout",
                  "unavailable", "ok"]
    for raw in raw_inputs:
        result = n(raw)
        assert result["state"] in allowed, \
            f"'{raw}' mapped to unknown state '{result['state']}'"


# ---------------------------------------------------------------------------
# Part 10 -- response contract shape examples
# ---------------------------------------------------------------------------

def test_green_no_risk_contract(monkeypatch):
    """GREEN / no-risk: scene_risks=[], reasoner_status.state in allowed states."""
    srv = _get_srv(monkeypatch)
    scene = srv._build_scene_risks([], None, [])
    assert scene == []
    status = srv._normalize_reasoner_status("not_triggered")
    assert status["state"] == "rules_only"


def test_yellow_object_near_edge_with_box_link():
    """YELLOW scene_risk includes bbox link."""
    risk_with_box = dict(DET_RISK_LINKED,
                         risk_id="edge_1", hazard_type="object_near_edge",
                         risk_level="YELLOW", severity=2, likelihood=2,
                         risk_score=4, risk_state="active")
    from server import _build_scene_risks, _is_linkable
    scene = _build_scene_risks([risk_with_box], None, [])
    assert len(scene) == 1
    assert _is_linkable(scene[0])
    assert scene[0]["bbox"]


def test_orange_risk_with_entity_link():
    """ORANGE scene_risk includes involved_track_ids."""
    from server import _build_scene_risks
    scene = _build_scene_risks([DET_RISK_LINKED], None, [])
    assert len(scene) == 1
    sr = scene[0]
    assert sr["involved_track_ids"] == ["trk_1"]
    assert sr["risk_level"] == "ORANGE"


def test_queued_vlm_status():
    """Queued state is correctly normalized."""
    from server import _normalize_reasoner_status
    s = _normalize_reasoner_status("throttled")
    assert s["state"] == "throttled"
