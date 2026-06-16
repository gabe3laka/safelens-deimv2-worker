"""
tests/test_risk_engine.py -- deterministic risk engine + per-session tracking (PR2).

CPU-only, no weights, no GPU. Covers:
  * risk matrix scoring/banding + malformed-profile fail-fast
  * controls hierarchy ordering
  * scene-graph near/overlap/edge geometry
  * per-session tracker isolation + TTL eviction (Build Mode pattern)
  * provenance stamping (produced_by=risk_engine, requires_human_review=False)
  * privacy blur (person regions changed)
  * deterministic rule output (same input -> same risk ids/levels)
  * additive wiring: /detect unchanged when disabled; risk block + schema_version
    when enabled; degradation never 500; /debug/state risk_engine block
  * ws-style risk enrichment
  * validation harness passes
"""

from __future__ import annotations

import base64
import io
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

import pytest

pytest.importorskip("pydantic")

import risk
from risk import controls, provenance, risk_matrix, scene_graph, tracking
from risk.risk_matrix import RiskMatrix, validate_profile

PERSON = {"label": "person", "class_id": 0, "confidence": 0.9,
          "bbox": {"x": 0.30, "y": 0.40, "w": 0.14, "h": 0.45}}
NO_HARDHAT = {"label": "no_hardhat", "class_id": 1, "confidence": 0.8,
              "bbox": {"x": 0.40, "y": 0.20, "w": 0.10, "h": 0.18}}
FORKLIFT = {"label": "forklift", "class_id": 7, "confidence": 0.86,
            "bbox": {"x": 0.34, "y": 0.42, "w": 0.24, "h": 0.40}}


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    monkeypatch.setenv("RISK_ENGINE_ENABLED", "true")
    monkeypatch.delenv("RISK_MATRIX_PROFILE", raising=False)
    tracking.reset()
    risk_matrix.reset_cache()
    yield
    tracking.reset()
    risk_matrix.reset_cache()


# -- risk matrix ---------------------------------------------------------------

def test_matrix_score_and_bands():
    m = risk_matrix.get_matrix()
    assert m.score(3, 4) == 12
    assert m.level(1, 1) == "GREEN"      # 1
    assert m.level(2, 3) == "YELLOW"     # 6
    assert m.level(3, 4) == "ORANGE"     # 12
    assert m.level(5, 5) == "RED"        # 25
    ev = m.evaluate(5, 4)
    assert ev["risk_score"] == 20 and ev["risk_level"] == "RED" and ev["should_alert"] is True


def test_matrix_clamps_out_of_range():
    m = risk_matrix.get_matrix()
    assert m.score(99, 99) == 25         # clamped to 5x5
    assert m.score(0, 0) == 1            # clamped to >=1


def test_malformed_matrix_fails_fast():
    # non-contiguous bands (gap 5..6 missing)
    bad = {"profile": "bad", "version": "0", "scale": {"severity_max": 5, "likelihood_max": 5},
           "bands": [{"level": "GREEN", "min": 1, "max": 4, "alert": False},
                     {"level": "RED", "min": 7, "max": 25, "alert": True}]}
    with pytest.raises(ValueError):
        validate_profile(bad)
    with pytest.raises(ValueError):
        RiskMatrix(bad)


def test_bundled_matrix_valid():
    # the committed default profile must validate
    validate_profile(risk_matrix.load_profile())


# -- controls ------------------------------------------------------------------

def test_controls_hierarchy_order():
    ctrls = controls.controls_for("person_forklift_proximity")
    order = {lvl: i for i, lvl in enumerate(controls.HIERARCHY)}
    idxs = [order[c["level"]] for c in ctrls]
    assert idxs == sorted(idxs)          # elimination before ... before ppe
    assert ctrls[0]["level"] == "elimination"
    assert controls.primary_action("fire")  # non-empty


def test_controls_unknown_hazard_has_default():
    ctrls = controls.controls_for("totally_unknown_hazard")
    assert ctrls and ctrls[-1]["level"] == "ppe"


# -- scene graph ---------------------------------------------------------------

def test_scene_graph_near_and_overlap():
    scene = scene_graph.build([PERSON, FORKLIFT], 1280, 720)
    rels = {r["relation"] for r in scene["relations"]}
    assert "overlaps" in rels or "near" in rels
    assert scene["object_count"] == 2


def test_scene_graph_edge_proximity():
    cup = {"label": "cup", "class_id": 41, "confidence": 0.6,
           "bbox": {"x": 0.95, "y": 0.5, "w": 0.04, "h": 0.1}}
    scene = scene_graph.build([cup], 1280, 720)
    assert scene["nodes"][0]["edges"]["right"] is True


# -- tracking: isolation + TTL -------------------------------------------------

def test_two_sessions_keep_isolated_tracks():
    tracking.update("cam_A", [PERSON], ts_ms=1000)
    tracking.update("cam_B", [PERSON, FORKLIFT], ts_ms=1000)
    assert tracking.session_track_count("cam_A") == 1
    assert tracking.session_track_count("cam_B") == 2
    # updating A again must not affect B
    tracking.update("cam_A", [PERSON, NO_HARDHAT], ts_ms=1100)
    assert tracking.session_track_count("cam_B") == 2
    assert tracking.active_session_count() == 2


def test_track_id_continuity_within_session():
    tracking.update("cam_A", [PERSON], ts_ms=1000)
    ids1 = tracking.get_track_ids("cam_A")
    # same object next frame keeps the same track id
    moved = {**PERSON, "bbox": {"x": 0.31, "y": 0.40, "w": 0.14, "h": 0.45}}
    tracking.update("cam_A", [moved], ts_ms=1100)
    assert tracking.get_track_ids("cam_A") == ids1


def test_stale_session_evicted_after_ttl(monkeypatch):
    monkeypatch.setenv("SESSION_TTL_MS", "5000")
    tracking.update("cam_old", [PERSON], ts_ms=1000)
    assert tracking.active_session_count() == 1
    assert tracking.sweep(now_ms=6001) == 0          # 5001ms idle > 5000 ttl
    assert tracking.active_session_count() == 0


def test_session_max_active_bounded(monkeypatch):
    monkeypatch.setenv("SESSION_MAX_ACTIVE", "3")
    for i in range(5):
        tracking.update(f"cam_{i}", [PERSON], ts_ms=1000 + i)
    assert tracking.active_session_count() <= 3


# -- provenance ----------------------------------------------------------------

def test_provenance_stamp():
    item = provenance.stamp({}, rule_id="R01_ppe_hardhat", ts_ms=42)
    assert item["produced_by"] == "risk_engine"
    assert item["rule_id"] == "R01_ppe_hardhat"
    assert item["requires_human_review"] is False
    assert item["timestamp_ms"] == 42
    assert item["model_version"]


# -- privacy -------------------------------------------------------------------

def test_privacy_blur_changes_person_region():
    np = pytest.importorskip("numpy")
    pytest.importorskip("PIL")
    from PIL import Image
    arr = np.zeros((100, 100, 3), dtype=np.uint8)
    arr[:, 50:] = 255                       # sharp vertical edge at x=50
    img = Image.fromarray(arr)
    from risk import privacy
    person = [{"label": "person", "bbox": {"x": 0.3, "y": 0.0, "w": 0.4, "h": 1.0}}]
    out = privacy.blur_persons(img, person, radius=8)
    assert np.array(out).tobytes() != np.array(img).tobytes()


def test_privacy_egress_guard_respects_flag(monkeypatch):
    np = pytest.importorskip("numpy")
    from PIL import Image
    from risk import privacy
    img = Image.fromarray(np.full((40, 40, 3), 200, dtype=np.uint8))
    person = [{"label": "person", "bbox": {"x": 0.1, "y": 0.1, "w": 0.8, "h": 0.8}}]
    monkeypatch.setenv("PRIVACY_BLUR_ENABLED", "false")
    _, blurred = privacy.sanitize_for_egress(img, person)
    assert blurred is False
    monkeypatch.setenv("PRIVACY_BLUR_ENABLED", "true")
    _, blurred = privacy.sanitize_for_egress(img, person)
    assert blurred is True


# -- engine: rules + determinism ----------------------------------------------

def test_engine_fires_expected_rules():
    out = risk.evaluate(entities=[NO_HARDHAT, PERSON, FORKLIFT],
                        img_w=1280, img_h=720, session_id="s1", ts_ms=1000)
    hz = {r["hazard_type"] for r in out["risks"]}
    assert "ppe_missing_hardhat" in hz
    assert "person_forklift_proximity" in hz
    assert out["highest_risk_level"] in ("ORANGE", "RED")
    # every risk carries provenance + controls
    for r in out["risks"]:
        assert r["produced_by"] == "risk_engine"
        assert r["requires_human_review"] is False
        assert r["recommended_controls"]
        assert r["risk_id"].startswith("rsk_")


def test_engine_deterministic():
    a = risk.evaluate(entities=[NO_HARDHAT, PERSON, FORKLIFT],
                      img_w=1280, img_h=720, session_id="sa", ts_ms=1000)
    tracking.reset()
    b = risk.evaluate(entities=[NO_HARDHAT, PERSON, FORKLIFT],
                      img_w=1280, img_h=720, session_id="sb", ts_ms=1000)
    norm = lambda o: sorted((r["risk_id"], r["risk_level"], r["risk_score"]) for r in o["risks"])
    assert norm(a) == norm(b)


def test_engine_empty_entities_no_risks():
    out = risk.evaluate(entities=[], img_w=640, img_h=480, session_id="s")
    assert out["risks"] == [] and out["highest_risk_level"] == "GREEN"
    assert out["schema_version"] == "risk.v1"


# -- additive attach: disabled vs enabled -------------------------------------

def test_attach_disabled_is_noop(monkeypatch):
    monkeypatch.setenv("RISK_ENGINE_ENABLED", "false")
    base = {"entities": [PERSON, NO_HARDHAT], "img_w": 1280, "img_h": 720,
            "backend": "yolo26", "poses": []}
    out = risk.attach_risk(dict(base), session_id="s")
    assert out == base                       # byte-for-byte legacy shape
    assert "schema_version" not in out and "risks" not in out


def test_attach_enabled_adds_block():
    base = {"entities": [PERSON, NO_HARDHAT], "img_w": 1280, "img_h": 720,
            "backend": "yolo26", "poses": []}
    out = risk.attach_risk(dict(base), session_id="cam_x", frame_id="f1")
    assert out["schema_version"] == "risk.v1"
    assert "risks" in out and "risk_engine" in out
    assert out["risk_engine"]["enabled"] is True
    assert out["entities"] == base["entities"]   # detection preserved


def test_attach_degrades_without_500(monkeypatch):
    # Force evaluate to blow up; attach_risk must keep detection + add warning.
    import risk.risk_engine as re
    monkeypatch.setattr(re, "evaluate", lambda **kw: (_ for _ in ()).throw(RuntimeError("boom")))
    base = {"entities": [PERSON], "img_w": 1280, "img_h": 720, "backend": "yolo26"}
    out = re.attach_risk(dict(base), session_id="s")
    assert out["entities"] == base["entities"]
    assert "risk_engine_error" in (out.get("warning") or "")
    assert out["risk_engine"]["degraded"] is True


# -- ws-style enrichment -------------------------------------------------------

def test_ws_style_message_enriched():
    vision_msg = {"type": "vision", "camera_id": "cam_ws", "frame_id": 7,
                  "entities": [PERSON, FORKLIFT], "poses": [], "img_w": 1280, "img_h": 720}
    out = risk.attach_risk(vision_msg, session_id="cam_ws", frame_id=7)
    assert out["type"] == "vision"               # original fields kept
    assert out["schema_version"] == "risk.v1"
    assert any(r["hazard_type"] == "person_forklift_proximity" for r in out["risks"])


# -- config block --------------------------------------------------------------

def test_config_reports_risk_engine():
    cfg = risk.config()
    for key in ("enabled", "tracking_enabled", "scene_graph_enabled",
                "provenance_enabled", "session_ttl_ms", "matrix"):
        assert key in cfg, key
    assert cfg["matrix_valid"] is True


# -- validation harness --------------------------------------------------------

def test_validation_harness_passes():
    from validation.run_validation import run, _DEFAULT_SCENARIOS
    report = run(_DEFAULT_SCENARIOS)
    assert report["passed"] is True
    assert report["critical_recall"] >= report["min_recall_critical"]


# -- /detect + /debug/state integration ---------------------------------------

@pytest.fixture()
def server_mod(monkeypatch):
    import importlib
    monkeypatch.setenv("SKIP_WARMUP", "true")
    monkeypatch.setenv("AUTO_WARMUP", "false")
    if "server" in sys.modules:
        del sys.modules["server"]
    return importlib.import_module("server")


def _tiny_jpeg_b64():
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), (255, 255, 255)).save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode()


def _fake_resp():
    from schema import BBox, Entity, InferResponse
    return InferResponse(
        entities=[Entity(label="person", class_id=0, confidence=0.9,
                         bbox=BBox(**PERSON["bbox"]), source="yolo26"),
                  Entity(label="no_hardhat", class_id=1, confidence=0.8,
                         bbox=BBox(**NO_HARDHAT["bbox"]), source="yolo26")],
        inference_ms=10, model="YOLO26", backend="yolo26", tasks=["det"],
        img_w=1280, img_h=720,
    )


def test_detect_risk_enabled_adds_block(server_mod, monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    import vision_backend
    monkeypatch.setenv("RISK_ENGINE_ENABLED", "true")
    monkeypatch.setattr(vision_backend, "run_inference", lambda **kw: _fake_resp())
    with server_mod._STATE_LOCK:
        server_mod._STATE["status"] = "ready"
    try:
        with TestClient(server_mod.app) as c:
            r = c.post("/detect", json={"image_b64": _tiny_jpeg_b64(), "session_id": "cam_1"})
            assert r.status_code == 200
            body = r.json()
            assert body["schema_version"] == "risk.v1"
            assert any(x["hazard_type"] == "ppe_missing_hardhat" for x in body["risks"])
            assert body["entities"][0]["label"] == "person"   # detection preserved
    finally:
        with server_mod._STATE_LOCK:
            server_mod._STATE["status"] = "cold"


def test_detect_risk_disabled_unchanged(server_mod, monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    import vision_backend
    monkeypatch.setenv("RISK_ENGINE_ENABLED", "false")
    monkeypatch.setattr(vision_backend, "run_inference", lambda **kw: _fake_resp())
    with server_mod._STATE_LOCK:
        server_mod._STATE["status"] = "ready"
    try:
        with TestClient(server_mod.app) as c:
            r = c.post("/detect", json={"image_b64": _tiny_jpeg_b64()})
            assert r.status_code == 200
            body = r.json()
            assert "schema_version" not in body and "risks" not in body
            assert body["backend"] == "yolo26"
    finally:
        with server_mod._STATE_LOCK:
            server_mod._STATE["status"] = "cold"


def test_debug_state_has_risk_engine(server_mod):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    with TestClient(server_mod.app) as c:
        body = c.get("/debug/state").json()
        assert "risk_engine" in body
        assert "enabled" in body["risk_engine"]
