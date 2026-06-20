"""
tests/test_reason_vlm.py -- event-driven /reason VLM + GroundingDINO scanner (PR3).

CPU-only, no real weights (REASONER_MODE=mock). Covers:
  * /reason returns strict, schema-valid JSON (mock adapter)
  * VLM outputs are ALWAYS AI drafts: produced_by=vlm_reasoner,
    requires_human_review=True, should_alert=False (enforced, not trusted)
  * disabled / unknown-mode / timeout degrade cleanly (never raise)
  * maybe_trigger is non-blocking, rate-limited, and level-gated
  * a /reason failure or timeout never breaks /detect
  * GroundingDINO scanner is candidate-only + human-review, never alerts
  * /debug/state exposes reasoner + open_vocab_scanner blocks
"""

from __future__ import annotations

import asyncio
import base64
import io
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

import pytest

pytest.importorskip("pydantic")

import risk.open_vocab_scanner as ovs
import risk.vlm_reasoner as vlm
from risk.reason_schema import ReasonResponse

PERSON = {"label": "person", "class_id": 0, "confidence": 0.9,
          "bbox": {"x": 0.30, "y": 0.40, "w": 0.14, "h": 0.45}}
FORKLIFT = {"label": "forklift", "class_id": 7, "confidence": 0.86,
            "bbox": {"x": 0.34, "y": 0.42, "w": 0.24, "h": 0.40}}

DET_RISK = {"risk_id": "rsk_R04", "hazard_type": "person_forklift_proximity",
            "risk_state": "active", "risk_level": "ORANGE", "severity": 4,
            "likelihood": 4, "risk_score": 16, "involved_track_ids": ["trk_1"],
            "should_alert": True,   # deterministic alert; VLM draft must NOT inherit it
            "recommended_controls": [{"level": "elimination", "action": "separate"}]}


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("VLM_REASONER_ENABLED", "true")
    monkeypatch.setenv("REASONER_MODE", "mock")
    monkeypatch.delenv("OPEN_VOCAB_SCANNER_ENABLED", raising=False)
    vlm.reset()
    ovs.reset()
    yield
    vlm.reset()
    ovs.reset()


def _req():
    return {"request_id": "r1", "session_id": "cam_A", "frame_id": "f1",
            "entities": [PERSON, FORKLIFT], "deterministic_risks": [DET_RISK]}


# -- strict schema + AI-draft contract ----------------------------------------

def test_reason_mock_strict_schema():
    out = vlm.reason_sync(_req())
    # must validate against the strict Pydantic response model
    parsed = ReasonResponse(**out)
    assert parsed.schema_version == "reason.v1"
    assert parsed.reasoner_status == "ok"
    assert parsed.reasoner_model == "mock"
    assert len(parsed.risks) == 1
    assert parsed.risks[0].hazard_type == "person_forklift_proximity"
    assert parsed.latency_ms is not None


def test_vlm_outputs_are_drafts_requiring_review():
    out = vlm.reason_sync(_req())
    assert out["produced_by"] == "vlm_reasoner"
    assert out["requires_human_review"] is True
    assert out["should_alert"] is False
    # even though the deterministic input had should_alert=True, the draft cannot inherit it
    for r in out["risks"]:
        assert r["requires_human_review"] is True
        assert r["should_alert"] is False


def test_reason_disabled(monkeypatch):
    monkeypatch.setenv("VLM_REASONER_ENABLED", "false")
    out = vlm.reason_sync(_req())
    assert out["reasoner_status"] == "disabled"
    assert out["risks"] == [] and out["requires_human_review"] is True


def test_reason_unknown_mode(monkeypatch):
    monkeypatch.setenv("REASONER_MODE", "not_a_mode")
    out = vlm.reason_sync(_req())
    assert out["reasoner_status"] == "unavailable"
    assert out["risks"] == []


# -- timeout + non-blocking + gating ------------------------------------------

def test_reason_async_timeout(monkeypatch):
    monkeypatch.setenv("REASONER_TIMEOUT_MS", "50")
    monkeypatch.setattr(vlm, "reason_sync",
                        lambda payload: (time.sleep(0.5), {"reasoner_status": "ok"})[1])
    out = asyncio.run(vlm.reason_async({"session_id": "s"}))
    assert out["reasoner_status"] == "timeout"
    assert out["requires_human_review"] is True and out["should_alert"] is False


def test_maybe_trigger_nonblocking_then_cached():
    draft, status = vlm.maybe_trigger("cam_A", frame_b64=None, highest_level="ORANGE",
                                      deterministic_risks=[DET_RISK], entities=[PERSON, FORKLIFT])
    assert status == "triggered" and draft is None     # returns immediately, async runs
    # background job populates the cache shortly
    deadline = time.monotonic() + 3.0
    cached = None
    while time.monotonic() < deadline:
        cached = vlm.get_cached_draft("cam_A")
        if cached:
            break
        time.sleep(0.05)
    assert cached is not None and cached["produced_by"] == "vlm_reasoner"
    # a second call within the interval is throttled but still returns the draft
    draft2, status2 = vlm.maybe_trigger("cam_A", frame_b64=None, highest_level="ORANGE",
                                        deterministic_risks=[DET_RISK])
    assert status2 == "cached" and draft2 is not None


def test_maybe_trigger_level_gated():
    draft, status = vlm.maybe_trigger("cam_G", frame_b64=None, highest_level="GREEN",
                                      deterministic_risks=[])
    assert status == "not_triggered" and draft is None


def test_maybe_trigger_disabled(monkeypatch):
    monkeypatch.setenv("VLM_REASONER_ENABLED", "false")
    draft, status = vlm.maybe_trigger("cam_A", frame_b64=None, highest_level="RED",
                                      deterministic_risks=[DET_RISK])
    assert status == "disabled" and draft is None


# -- GroundingDINO scanner: candidate-only ------------------------------------

def test_open_vocab_disabled_is_candidate_only():
    out = ovs.scan("ZmFrZQ==", session_id="s", frame_id="f")
    assert out["status"] == "disabled"
    assert out["produced_by"] == "open_vocab_scanner"
    assert out["candidate_only"] is True
    assert out["requires_human_review"] is True
    assert "should_alert" not in out          # scanner has no alert authority
    assert out["candidates"] == []


def test_open_vocab_enabled_degrades_without_weights(monkeypatch):
    # Enabled but no weights/network here -> must degrade, still candidate-only.
    monkeypatch.setenv("OPEN_VOCAB_SCANNER_ENABLED", "true")
    from PIL import Image
    buf = io.BytesIO(); Image.new("RGB", (16, 16), (200, 200, 200)).save(buf, format="JPEG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    out = ovs.scan(b64, session_id="s", force=True)
    assert out["status"] in ("ok", "unavailable", "error")   # never raises
    assert out["candidate_only"] is True and out["requires_human_review"] is True


def test_open_vocab_config():
    cfg = ovs.config()
    assert cfg["candidate_only"] is True
    for key in ("enabled", "backend", "scan_interval_ms"):
        assert key in cfg


# -- status snapshot -----------------------------------------------------------

def test_reasoner_status_snapshot():
    snap = vlm.status_snapshot()
    for key in ("enabled", "mode", "model_id", "trigger_level", "min_interval_ms",
                "timeout_ms", "active_sessions"):
        assert key in snap, key
    assert snap["mode"] == "mock"
    assert "diagnostics" in snap
    assert "vlm.quantization_requested" in snap["diagnostics"]
    assert "vlm.bitsandbytes_available" in snap["diagnostics"]


def test_live_default_model_is_qwen_3b(monkeypatch):
    monkeypatch.delenv("QWEN_VL_MODEL_ID", raising=False)
    monkeypatch.setenv("REASONER_MODE", "qwen_vl")
    assert vlm._model_id() == "Qwen/Qwen2.5-VL-3B-Instruct"
    assert "7B" not in vlm._model_id()


def test_live_default_output_and_image_limits(monkeypatch):
    monkeypatch.delenv("REASONER_MAX_NEW_TOKENS", raising=False)
    monkeypatch.delenv("REASONER_MAX_IMAGE_SIDE", raising=False)
    assert vlm._max_new_tokens() == 128
    assert vlm._max_image_side() == 512


def test_visual_token_pixels_from_env(monkeypatch):
    monkeypatch.setenv("QWEN_VL_MIN_VISUAL_TOKENS", "256")
    monkeypatch.setenv("QWEN_VL_MAX_VISUAL_TOKENS", "768")
    assert vlm._visual_pixels("QWEN_VL_MIN_VISUAL_TOKENS", 256) == 256 * (28 * 28)
    assert vlm._visual_pixels("QWEN_VL_MAX_VISUAL_TOKENS", 768) == 768 * (28 * 28)


def test_processor_receives_min_max_pixels(monkeypatch):
    monkeypatch.setenv("QWEN_VL_MIN_VISUAL_TOKENS", "300")
    monkeypatch.setenv("QWEN_VL_MAX_VISUAL_TOKENS", "700")
    captured = {}

    class _NoGrad:
        def __enter__(self): return self
        def __exit__(self, *_): return False

    class _Torch:
        @staticmethod
        def no_grad():
            return _NoGrad()

    class _Inputs(dict):
        def to(self, _device):
            return self

    class _Ids:
        shape = (1, 3)

    class _Processor:
        def __call__(self, **kwargs):
            return _Inputs({"input_ids": _Ids()})

        def apply_chat_template(self, *_a, **_k):
            return "prompt"

        def batch_decode(self, *_a, **_k):
            return ["{}"]

    class _AutoProcessor:
        @staticmethod
        def from_pretrained(_model_id, **kwargs):
            captured["min_pixels"] = kwargs.get("min_pixels")
            captured["max_pixels"] = kwargs.get("max_pixels")
            return _Processor()

    class _Model:
        def __init__(self):
            self.device = "cpu"

        @staticmethod
        def from_pretrained(*_a, **_k):
            return _Model()

        def eval(self):
            return self

        def to(self, *_a, **_k):
            return self

        def generate(self, **_kwargs):
            class _Out:
                def __getitem__(self, _k):
                    return [[4]]
            return _Out()

    fake_transformers = type(
        "T",
        (),
        {
            "AutoProcessor": _AutoProcessor,
            "Qwen2_5_VLForConditionalGeneration": _Model,
        },
    )
    monkeypatch.setitem(sys.modules, "torch", _Torch())
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    adapter = vlm._build_adapter("qwen_vl")
    adapter["generate"]("prompt", None)
    assert captured["min_pixels"] == 300 * (28 * 28)
    assert captured["max_pixels"] == 700 * (28 * 28)


def test_quantization_requested_and_available_activates(monkeypatch):
    kwargs = {}
    diag = {
        "vlm.quantization_requested": "4bit",
        "vlm.bitsandbytes_available": True,
        "vlm.quantization_active": False,
    }

    class _FakeBnb:
        def __init__(self, **kw):
            self.kw = kw

    monkeypatch.setitem(sys.modules, "transformers", type("T", (), {"BitsAndBytesConfig": _FakeBnb}))
    vlm._configure_quantization(kwargs, diag)
    assert "quantization_config" in kwargs
    assert diag["vlm.quantization_active"] is True


def test_quantization_requested_but_unavailable_degrades(monkeypatch):
    kwargs = {}
    diag = {
        "vlm.quantization_requested": "4bit",
        "vlm.bitsandbytes_available": False,
        "vlm.quantization_active": False,
    }
    vlm._configure_quantization(kwargs, diag)
    assert "quantization_config" not in kwargs
    assert diag["vlm.quantization_active"] is False


# -- server integration: /reason, /scan, /detect resilience, /debug/state ------

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
                  Entity(label="forklift", class_id=7, confidence=0.86,
                         bbox=BBox(**FORKLIFT["bbox"]), source="yolo26")],
        inference_ms=10, model="YOLO26", backend="yolo26", tasks=["det"],
        img_w=1280, img_h=720,
    )


def test_reason_route_strict_json(server_mod):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    with TestClient(server_mod.app) as c:
        r = c.post("/reason", json=_req())
        assert r.status_code == 200
        body = r.json()
        assert body["schema_version"] == "reason.v1"
        assert body["produced_by"] == "vlm_reasoner"
        assert body["requires_human_review"] is True and body["should_alert"] is False
        ReasonResponse(**body)            # strict schema-valid


def test_scan_route_candidate_only(server_mod):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    with TestClient(server_mod.app) as c:
        r = c.post("/scan", json={"frame_b64": _tiny_jpeg_b64(), "session_id": "s"})
        assert r.status_code == 200
        body = r.json()
        assert body["produced_by"] == "open_vocab_scanner"
        assert body["candidate_only"] is True and body["requires_human_review"] is True


def test_detect_triggers_reasoner_nonblocking(server_mod, monkeypatch):
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
            # deterministic risk present + non-blocking reasoner_status attached
            assert body["schema_version"] == "risk.v1"
            # reasoner_status is now a normalized dict; state must be one of the
            # standard app-facing values.
            rs = body.get("reasoner_status")
            assert isinstance(rs, dict), f"expected dict, got {rs!r}"
            assert rs.get("state") in (
                "running", "ready", "queued", "queued_latest", "throttled", "rules_only",
                "unavailable", "disabled", "error", "timeout"), rs
            assert body["entities"][0]["label"] == "person"   # detection preserved
    finally:
        with server_mod._STATE_LOCK:
            server_mod._STATE["status"] = "cold"


def test_detect_survives_reasoner_failure(server_mod, monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    import vision_backend
    import risk.vlm_reasoner as _vlm
    monkeypatch.setenv("RISK_ENGINE_ENABLED", "true")
    monkeypatch.setattr(vision_backend, "run_inference", lambda **kw: _fake_resp())
    monkeypatch.setattr(_vlm, "maybe_trigger",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("vlm boom")))
    with server_mod._STATE_LOCK:
        server_mod._STATE["status"] = "ready"
    try:
        with TestClient(server_mod.app) as c:
            r = c.post("/detect", json={"image_b64": _tiny_jpeg_b64(), "session_id": "cam_1"})
            assert r.status_code == 200          # reasoner failure never breaks /detect
            body = r.json()
            assert body["entities"][0]["label"] == "person"
            assert body["schema_version"] == "risk.v1"
    finally:
        with server_mod._STATE_LOCK:
            server_mod._STATE["status"] = "cold"


def test_debug_state_has_reasoner_blocks(server_mod):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    with TestClient(server_mod.app) as c:
        body = c.get("/debug/state").json()
        assert "reasoner" in body and "enabled" in body["reasoner"]
        assert "open_vocab_scanner" in body and "candidate_only" in body["open_vocab_scanner"]
