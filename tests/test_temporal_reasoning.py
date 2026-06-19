"""
tests/test_temporal_reasoning.py -- event-triggered temporal VLM perception layer.

CPU-only, no real weights (REASONER_MODE=mock). Covers: backward-compat (off by
default), per-session memory isolation, missing session_id, trigger reasons
(low-confidence stable, label instability, scene mismatch, object-near-edge),
mock semantic corrections (suppress vehicle FPs, never real hazards, raw label
preserved), object-near-edge deterministic risk, and the non-blocking contract
(/detect never waits on the VLM; a slow/failing reasoner never breaks /detect).
"""

from __future__ import annotations

import base64
import io
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

import pytest

pytest.importorskip("pydantic")

import temporal_reasoning as tr
import temporal_reasoning.async_reasoning as ar
from temporal_reasoning import edge_risk, scene_context, semantic_corrections
from temporal_reasoning import session_memory as mem
from temporal_reasoning import triggers


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("TEMPORAL_REASONING_ENABLED", "true")
    monkeypatch.setenv("VLM_REASONER_ENABLED", "true")
    monkeypatch.setenv("REASONER_MODE", "mock")
    monkeypatch.setenv("TEMPORAL_REASONING_TRIGGER_MIN_INTERVAL_MS", "0")
    monkeypatch.setenv("REASONER_MIN_INTERVAL_MS", "0")
    tr.reset_all()
    yield
    tr.reset_all()


def _cup(y, tid="trk_cup"):
    return {"track_id": tid, "label": "cup", "confidence": 0.8,
            "bbox": {"x": 0.40, "y": y, "w": 0.06, "h": 0.09}}


# -- backward compatibility ----------------------------------------------------

def test_disabled_is_byte_for_byte_noop(monkeypatch):
    monkeypatch.setenv("TEMPORAL_REASONING_ENABLED", "false")
    resp = {"entities": [{"label": "cup"}], "backend": "yolo26"}
    out = tr.attach_temporal(dict(resp), session_id="cam_1")
    assert out == resp
    assert "temporal_reasoning" not in out and "reasoner_status" not in out


def test_old_detect_shape_still_parses_when_enabled():
    resp = {"entities": [{"label": "cup", "confidence": 0.8}], "backend": "yolo26",
            "poses": [], "model": "YOLO26"}
    out = tr.attach_temporal(dict(resp), session_id="cam_1")
    # additive only: original keys preserved untouched
    for k in ("entities", "backend", "poses", "model"):
        assert out[k] == resp[k]
    assert out["temporal_reasoning"]["enabled"] is True


# -- session memory ------------------------------------------------------------

def test_session_memory_isolation():
    mem.update("A", frame_id="f", entities=[],
               tracks=[{"track_id": "t1", "label": "cup", "bbox": {"x": .1, "y": .1, "w": .1, "h": .1}}])
    mem.update("B", frame_id="f", entities=[],
               tracks=[{"track_id": "t1", "label": "bottle", "bbox": {"x": .2, "y": .2, "w": .1, "h": .1}}])
    assert mem.label_history("A", "t1") == ["cup"]
    assert mem.label_history("B", "t1") == ["bottle"]
    assert mem.active_track_count("A") == 1 and mem.active_track_count("B") == 1


def test_missing_session_id_does_not_crash():
    out = tr.attach_temporal({"entities": [{"label": "cup", "confidence": 0.8}]},
                             session_id=None, frame_id=None)
    assert out["temporal_reasoning"]["enabled"] is True   # no exception


# -- triggers ------------------------------------------------------------------

def test_low_conf_stable_triggers():
    sid = "cam_lc"
    for _ in range(4):
        mem.update(sid, frame_id="f", entities=[],
                   tracks=[{"track_id": "t1", "label": "cup", "confidence": 0.2,
                            "bbox": {"x": .1, "y": .1, "w": .1, "h": .1}}])
    reasons = triggers.evaluate(sid, entities=[{"label": "cup", "confidence": 0.2}],
                                tracks=[], highest_level="GREEN",
                                deterministic_risks=[], edge_risks=[], payload={})
    assert "low_conf_stable" in reasons


def test_label_instability_triggers():
    sid = "cam_flip"
    for lab in ("cup", "bottle", "cup", "bottle"):
        mem.update(sid, frame_id="f", entities=[],
                   tracks=[{"track_id": "t1", "label": lab, "confidence": 0.8,
                            "bbox": {"x": .1, "y": .1, "w": .1, "h": .1}}])
    reasons = triggers.evaluate(sid, entities=[], tracks=[], highest_level="GREEN",
                                deterministic_risks=[], edge_risks=[], payload={})
    assert "label_instability" in reasons


def test_scene_mismatch_triggers_for_indoor_vehicle():
    reasons = triggers.evaluate("cam_x", entities=[{"label": "bus"}], tracks=[],
                                highest_level="GREEN", deterministic_risks=[],
                                edge_risks=[], payload={"scene_hint": "cafe"})
    assert "scene_mismatch" in reasons


def test_force_reason_triggers():
    reasons = triggers.evaluate("cam_x", entities=[], tracks=[], highest_level="GREEN",
                                deterministic_risks=[], edge_risks=[],
                                payload={"reasoning_preferences": {"force_reason": True}})
    assert "user_request" in reasons


# -- object-near-edge risk -----------------------------------------------------

def test_object_near_edge_frame_fallback():
    sid = "cam_edge"
    mem.update(sid, frame_id="f0", entities=[], tracks=[_cup(0.60)])
    mem.update(sid, frame_id="f1", entities=[], tracks=[_cup(0.80)])
    risks = edge_risk.evaluate(sid, entities=[], tracks=[_cup(0.90)])
    assert len(risks) == 1
    r = risks[0]
    assert r["hazard_type"] == "object_near_edge"
    assert r["edge_reference"] == "frame_fallback"
    assert r["risk_state"] == "latent" and r["requires_human_review"] is False
    assert r["risk_score"] == r["severity"] * r["likelihood"]


def test_object_near_edge_surface_reference():
    sid = "cam_edge2"
    table = {"label": "dining table", "bbox": {"x": 0.2, "y": 0.5, "w": 0.6, "h": 0.4}}
    cup = {"track_id": "c1", "label": "cup", "bbox": {"x": 0.4, "y": 0.42, "w": 0.06, "h": 0.09}}
    mem.update(sid, frame_id="f", entities=[table], tracks=[cup])
    risks = edge_risk.evaluate(sid, entities=[table], tracks=[cup])
    assert risks and risks[0]["edge_reference"] == "surface"


def test_persons_never_flagged_as_edge_risk():
    sid = "cam_p"
    person = {"track_id": "p1", "label": "person", "bbox": {"x": 0.1, "y": 0.9, "w": 0.2, "h": 0.1}}
    mem.update(sid, frame_id="f", entities=[], tracks=[person])
    assert edge_risk.evaluate(sid, entities=[], tracks=[person]) == []


# -- semantic corrections (perception, not safety) -----------------------------

def test_cafe_hint_suppresses_vehicle_fp_not_real_hazards():
    sc = scene_context.mock_scene_context(
        [{"label": "cup"}, {"label": "chair"}], {"scene_hint": "cafe"})
    corr = semantic_corrections.mock_corrections(
        [{"label": "bus"}, {"label": "cup"}, {"label": "person"}, {"label": "knife"}],
        sc, None)
    corrected_labels = {c["raw_label"] for c in corr}
    assert "bus" in corrected_labels            # vehicle FP suppressed
    assert "cup" not in corrected_labels        # real object kept
    assert "person" not in corrected_labels     # real hazard NEVER suppressed
    assert "knife" not in corrected_labels      # real hazard NEVER suppressed
    for c in corr:
        assert c["requires_human_review"] is False          # perception, not safety
        assert c["purpose"] == "perception_correction"
        assert c["authority"] == "advisory_perception"


def test_semantic_corrections_preserve_raw_label():
    sc = scene_context.mock_scene_context([], {"scene_hint": "office"})
    corr = semantic_corrections.mock_corrections([{"label": "truck"}], sc, None)
    assert corr and corr[0]["raw_label"] == "truck"
    assert corr[0]["corrected_label"] and corr[0]["corrected_label"] != "truck"


def test_corrections_skipped_outdoors():
    sc = scene_context.mock_scene_context([{"label": "car"}], {"scene_hint": "construction"})
    # construction maps to outdoor -> no contextual suppression of vehicles
    assert semantic_corrections.mock_corrections([{"label": "bus"}], sc, None) == []


# -- non-blocking contract -----------------------------------------------------

def test_detect_never_waits_on_vlm(monkeypatch):
    """attach_temporal returns immediately even if the VLM would take seconds."""
    import risk.vlm_reasoner as vlm
    monkeypatch.setenv("REASONER_MODE", "qwen_vl")   # real path -> uses generate_json

    def _slow(*a, **k):
        time.sleep(2.0)
        return None
    monkeypatch.setattr(vlm, "generate_json", _slow)
    monkeypatch.setattr(vlm, "enabled", lambda: True)

    t0 = time.perf_counter()
    out = tr.attach_temporal({"entities": [{"label": "bus", "confidence": 0.9}],
                              "highest_risk_level": "ORANGE", "risks": []},
                             session_id="cam_nb", frame_b64=None,
                             payload={"reasoning_preferences": {"force_reason": True}})
    elapsed = time.perf_counter() - t0
    assert elapsed < 1.0                       # did NOT block on the 2s VLM
    assert out["entities"][0]["label"] == "bus"   # detection preserved


def test_reasoner_failure_never_breaks_attach(monkeypatch):
    import temporal_reasoning.async_reasoning as ar
    monkeypatch.setattr(ar, "maybe_trigger",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    out = tr.attach_temporal({"entities": [{"label": "cup", "confidence": 0.8}]},
                             session_id="cam_err")
    # attach swallows the failure and still returns the detection
    assert out["entities"][0]["label"] == "cup"


def test_latest_frame_wins_replaces_pending(monkeypatch):
    monkeypatch.setenv("REASONER_LATEST_WINS", "true")
    sid = "cam_latest"
    with ar._LOCK:
        ar._INFLIGHT.add(sid)
        ar._PENDING.pop(sid, None)
    try:
        s1 = ar.maybe_trigger(
            sid,
            reasons=["risk_escalation"],
            entities=[{"label": "cup"}],
            tracks=[],
            frame_b64="frame_old",
            payload={},
        )
        s2 = ar.maybe_trigger(
            sid,
            reasons=["risk_escalation"],
            entities=[{"label": "cup"}],
            tracks=[],
            frame_b64="frame_new",
            payload={},
        )
        assert s1 == "queued_latest"
        assert s2 == "queued_latest"
        with ar._LOCK:
            assert sid in ar._PENDING
            assert ar._PENDING[sid]["ctx"]["frame_b64"] == "frame_new"
    finally:
        with ar._LOCK:
            ar._INFLIGHT.discard(sid)
            ar._PENDING.pop(sid, None)


def test_temporal_memory_does_not_store_raw_frames():
    sid = "cam_priv"
    mem.update(
        sid,
        frame_id="f_raw",
        entities=[{"label": "cup", "confidence": 0.7, "bbox": {"x": 0.1, "y": 0.1, "w": 0.1, "h": 0.1}}],
        tracks=[],
    )
    snap = mem.snapshot(sid)
    dumped = str(snap).lower()
    assert "frame_b64" not in dumped
    assert "image_b64" not in dumped


def test_privacy_blur_applied_before_vlm_frame_send(monkeypatch):
    """The temporal real path sends frames through vlm.generate_json, which blurs
    persons via _decode_blurred before the model ever sees the image."""
    np = pytest.importorskip("numpy")
    from PIL import Image
    import risk.vlm_reasoner as vlm
    monkeypatch.setenv("REASONER_MODE", "qwen_vl")
    monkeypatch.setenv("PRIVACY_BLUR_ENABLED", "true")
    arr = np.zeros((80, 80, 3), dtype=np.uint8); arr[:, 40:] = 255
    buf = io.BytesIO(); Image.fromarray(arr).save(buf, format="JPEG")
    raw_b64 = base64.b64encode(buf.getvalue()).decode()
    captured = {}

    def _fake_adapter(_m):
        def _gen(prompt, image):
            captured["image"] = image
            return "{}"
        return {"available": True, "generate": _gen}
    monkeypatch.setattr(vlm, "_get_adapter", _fake_adapter)
    monkeypatch.setattr(vlm, "enabled", lambda: True)

    vlm.generate_json("p", frame_b64=raw_b64,
                      entities=[{"label": "person", "bbox": {"x": 0.25, "y": 0.0, "w": 0.5, "h": 1.0}}])
    raw_img = Image.open(io.BytesIO(base64.b64decode(raw_b64))).convert("RGB")
    assert captured.get("image") is not None
    assert np.array(captured["image"]).tobytes() != np.array(raw_img).tobytes()


# -- /detect integration (mock, polled; no real weights) -----------------------

@pytest.fixture()
def server_mod(monkeypatch):
    import importlib
    monkeypatch.setenv("SKIP_WARMUP", "true")
    monkeypatch.setenv("AUTO_WARMUP", "false")
    monkeypatch.setenv("RISK_ENGINE_ENABLED", "true")
    if "server" in sys.modules:
        del sys.modules["server"]
    return importlib.import_module("server")


def _img_b64():
    from PIL import Image
    buf = io.BytesIO(); Image.new("RGB", (8, 8), (200, 200, 200)).save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode()


def _fake_resp():
    from schema import BBox, Entity, InferResponse
    return InferResponse(
        entities=[Entity(label="cup", class_id=41, confidence=0.2,
                         bbox=BBox(x=0.40, y=0.88, w=0.06, h=0.09), source="yolo26")],
        inference_ms=10, model="YOLO26", backend="yolo26", tasks=["det"],
        img_w=1280, img_h=720)


def test_detect_attaches_temporal_blocks(server_mod, monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    import vision_backend
    monkeypatch.setattr(vision_backend, "run_inference", lambda **kw: _fake_resp())
    with server_mod._STATE_LOCK:
        server_mod._STATE["status"] = "ready"
    try:
        with TestClient(server_mod.app) as c:
            r = c.post("/detect", json={"image_b64": _img_b64(), "session_id": "cam_t",
                                        "scene_hint": "cafe"})
            assert r.status_code == 200
            body = r.json()
            assert body["entities"][0]["label"] == "cup"          # detection preserved
            assert body["temporal_reasoning"]["enabled"] is True
            assert isinstance(body["reasoner_status"], dict)
            # mock job populates scene_context shortly (non-blocking)
            deadline = time.monotonic() + 3.0
            while time.monotonic() < deadline:
                sc = c.post("/detect", json={"image_b64": _img_b64(),
                                             "session_id": "cam_t", "scene_hint": "cafe"}).json()
                if sc.get("scene_context"):
                    break
                time.sleep(0.05)
            assert sc.get("scene_context", {}).get("scene_type") == "cafe"
    finally:
        with server_mod._STATE_LOCK:
            server_mod._STATE["status"] = "cold"
