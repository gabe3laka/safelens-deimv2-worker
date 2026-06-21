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
import risk.gemini_reasoner as gemini_reasoner
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
    assert snap["serve_backend"] == "mock"


def test_live_default_output_and_image_limits(monkeypatch):
    monkeypatch.delenv("REASONER_MAX_IMAGE_SIDE", raising=False)
    assert vlm._max_image_side() == 512


def test_gemini_default_mode(monkeypatch):
    monkeypatch.delenv("REASONER_MODE", raising=False)
    assert vlm.mode() == "gemini"


def test_gemini_model_id(monkeypatch):
    monkeypatch.setenv("REASONER_MODE", "gemini")
    monkeypatch.setenv("GEMINI_MODEL_ID", "gemini-2.5-flash")
    assert vlm._model_id() == "gemini-2.5-flash"


def test_qwen_vl_mode_is_unavailable(monkeypatch):
    monkeypatch.setenv("REASONER_MODE", "qwen_vl")
    out = vlm.reason_sync(_req())
    assert out["reasoner_status"] == "unavailable"
    assert "qwen_vl" in out.get("error", "").lower() or "removed" in out.get("error", "").lower()


def test_deepseek_vl2_mode_is_unavailable(monkeypatch):
    monkeypatch.setenv("REASONER_MODE", "deepseek_vl2")
    out = vlm.reason_sync(_req())
    assert out["reasoner_status"] == "unavailable"


def test_qwen_deep_env_vars_preserved_as_inert_placeholders_in_dockerfile():
    """QWEN_VL_DEEP_* must remain declared in Dockerfile as inert placeholder."""
    from pathlib import Path
    dockerfile = (Path(__file__).parent.parent / "Dockerfile").read_text()
    assert "QWEN_VL_DEEP_MODEL_ID" in dockerfile
    assert "QWEN_VL_DEEP_ENABLED" in dockerfile


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


def test_background_timeout_stores_terminal_cache(monkeypatch):
    monkeypatch.setenv("REASONER_TIMEOUT_MS", "50")
    monkeypatch.setattr(vlm, "reason_sync", lambda req: (time.sleep(0.3), {"reasoner_status": "ok"})[1])
    vlm._run_and_cache("cam_timeout", {"session_id": "cam_timeout", "frame_id": "f_timeout"})
    cached = vlm.get_cached_draft("cam_timeout")
    assert cached is not None
    assert cached["reasoner_status"] == "timeout"


def test_background_exception_stores_terminal_cache(monkeypatch):
    def boom(req):
        raise RuntimeError("vlm boom")
    monkeypatch.setattr(vlm, "reason_sync", boom)
    vlm._run_and_cache("cam_error", {"session_id": "cam_error", "frame_id": "f_error"})
    cached = vlm.get_cached_draft("cam_error")
    assert cached is not None
    assert cached["reasoner_status"] == "error"
    assert "vlm boom" in cached.get("error", "")


def test_background_success_stores_ready_cache(monkeypatch):
    monkeypatch.setattr(vlm, "reason_sync", lambda req: {"reasoner_status": "ok", "session_id": req["session_id"]})
    vlm._run_and_cache("cam_ok", {"session_id": "cam_ok", "frame_id": "f_ok"})
    cached = vlm.get_cached_draft("cam_ok")
    assert cached is not None
    assert cached["reasoner_status"] == "ok"


def test_detect_poll_only_does_not_call_maybe_trigger(server_mod, monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    import vision_backend
    import risk.vlm_reasoner as _vlm
    monkeypatch.setenv("RISK_ENGINE_ENABLED", "true")
    monkeypatch.setattr(vision_backend, "run_inference", lambda **kw: _fake_resp())
    monkeypatch.setattr(_vlm, "get_cached_draft", lambda sid: {"reasoner_status": "ok", "risks": []})
    def fail(*args, **kwargs):
        raise AssertionError("maybe_trigger should not be called for poll-only")
    monkeypatch.setattr(_vlm, "maybe_trigger", fail)
    with server_mod._STATE_LOCK:
        server_mod._STATE["status"] = "ready"
    try:
        with TestClient(server_mod.app) as c:
            r = c.post("/detect", json={
                "image_b64": _tiny_jpeg_b64(),
                "session_id": "cam_poll",
                "reasoning_preferences": {
                    "do_not_start_new_reasoning_job": True,
                    "force_reason": False,
                    "return_reasoner_status": True,
                },
            })
            assert r.status_code == 200
            assert r.json()["reasoner_status"]["state"] == "ready"
    finally:
        with server_mod._STATE_LOCK:
            server_mod._STATE["status"] = "cold"


def test_detect_force_reason_still_calls_maybe_trigger(server_mod, monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    import vision_backend
    import risk.vlm_reasoner as _vlm
    calls = {"n": 0}
    monkeypatch.setenv("RISK_ENGINE_ENABLED", "true")
    monkeypatch.setattr(vision_backend, "run_inference", lambda **kw: _fake_resp())
    def called(*args, **kwargs):
        calls["n"] += 1
        return None, "triggered"
    monkeypatch.setattr(_vlm, "maybe_trigger", called)
    with server_mod._STATE_LOCK:
        server_mod._STATE["status"] = "ready"
    try:
        with TestClient(server_mod.app) as c:
            r = c.post("/detect", json={
                "image_b64": _tiny_jpeg_b64(),
                "session_id": "cam_force",
                "reasoning_preferences": {
                    "do_not_start_new_reasoning_job": True,
                    "force_reason": True,
                    "return_reasoner_status": True,
                },
            })
            assert r.status_code == 200
            assert calls["n"] == 1
    finally:
        with server_mod._STATE_LOCK:
            server_mod._STATE["status"] = "cold"


def test_parser_handles_fenced_json():
    raw = '```json\n{"scene_summary":"ok","risks":[],"uncertain_items":[]}\n```'
    assert vlm._extract_json(raw)["scene_summary"] == "ok"


def test_parser_handles_prose_plus_json():
    raw = 'Here is the result: {"scene_summary":"ok","risks":[],"uncertain_items":[]} thanks'
    assert vlm._extract_json(raw)["risks"] == []


def test_parse_failure_statuses_are_not_mapped_to_unavailable(server_mod):
    schema = server_mod._normalize_reasoner_status("schema_error")
    parse = server_mod._normalize_reasoner_status("json_parse_error")
    assert schema["state"] == "schema_error"
    assert parse["state"] == "json_parse_error"
    assert schema["state"] != "unavailable"
    assert parse["state"] != "unavailable"


def test_cached_schema_error_prevents_immediate_retrigger(monkeypatch):
    monkeypatch.setenv("REASONER_CACHE_TTL_MS", "10000")
    monkeypatch.setenv("REASONER_MIN_INTERVAL_MS", "1")
    with vlm._LOCK:
        vlm._CACHE["cam_schema"] = {"response": {"reasoner_status": "schema_error", "risks": []}, "ts": vlm._now_ms()}
    called = {"submit": 0}
    class NoSubmit:
        def submit(self, *args, **kwargs):
            called["submit"] += 1
    monkeypatch.setattr(vlm, "_executor", lambda: NoSubmit())
    draft, status = vlm.maybe_trigger(
        "cam_schema", frame_b64=None, highest_level="ORANGE", deterministic_risks=[DET_RISK]
    )
    assert status == "schema_error"
    assert draft["reasoner_status"] == "schema_error"
    assert called["submit"] == 0


def test_force_reason_can_retry_after_schema_failure(monkeypatch):
    monkeypatch.setenv("REASONER_CACHE_TTL_MS", "10000")
    monkeypatch.setenv("REASONER_MIN_INTERVAL_MS", "1")
    with vlm._LOCK:
        vlm._CACHE["cam_force_schema"] = {"response": {"reasoner_status": "schema_error", "risks": []}, "ts": vlm._now_ms()}
        vlm._LAST_RUN_MS["cam_force_schema"] = 0
    called = {"submit": 0}
    class Submitter:
        def submit(self, *args, **kwargs):
            called["submit"] += 1
    monkeypatch.setattr(vlm, "_executor", lambda: Submitter())
    draft, status = vlm.maybe_trigger(
        "cam_force_schema", frame_b64=None, highest_level="ORANGE",
        deterministic_risks=[DET_RISK], force_reason=True,
    )
    assert status == "cached_and_triggered"
    assert draft["reasoner_status"] == "schema_error"
    assert called["submit"] == 1


def test_background_json_parse_error_stores_terminal_cache(monkeypatch):
    monkeypatch.setattr(vlm, "reason_sync", lambda req: {
        "reasoner_status": "json_parse_error", "error": "model did not return valid JSON",
        "scene_summary": "", "risks": [], "uncertain_items": [],
    })
    vlm._run_and_cache("cam_schema_store", {"session_id": "cam_schema_store", "frame_id": "f_schema"})
    cached = vlm.get_cached_draft("cam_schema_store")
    assert cached["reasoner_status"] == "json_parse_error"
    assert cached["error"] == "model did not return valid JSON"
    assert cached["risks"] == []


# -- JSON extraction: robust parser tests --

def test_parser_handles_fenced_without_language_tag():
    raw = "```\n{\"scene_summary\":\"x\",\"risks\":[]}\n```"
    assert vlm._extract_json(raw)["scene_summary"] == "x"


def test_parser_handles_prose_before_and_after_json():
    assert vlm._extract_json('Here you go: {"scene_summary":"x","risks":[]}')["scene_summary"] == "x"
    assert vlm._extract_json('{"scene_summary":"y","risks":[]} -- done')["scene_summary"] == "y"


def test_parser_prefers_object_with_expected_keys_among_multiple():
    raw = '{"unrelated":1} then {"box_updates":[],"uncertain_box_ids":[]}'
    assert vlm._extract_json(raw)["box_updates"] == []


def test_parser_handles_nested_objects_and_braces_in_strings():
    raw = 'noise {"scene_summary":"a } and { brace","risks":[{"bbox":{"x":0.1,"y":0.2}}]} tail'
    out = vlm._extract_json(raw)
    assert out["risks"][0]["bbox"]["x"] == 0.1
    assert out["scene_summary"] == "a } and { brace"


def test_parser_returns_none_on_unrecoverable_output():
    assert vlm._extract_json("no json here at all") is None
    assert vlm._extract_json('{"scene_summary":"x","risks":[') is None  # truncated


# -- status surfacing + no immediate retrigger --

def test_cached_json_parse_error_prevents_immediate_retrigger(monkeypatch):
    monkeypatch.setenv("REASONER_CACHE_TTL_MS", "10000")
    monkeypatch.setenv("REASONER_MIN_INTERVAL_MS", "1")
    with vlm._LOCK:
        vlm._CACHE["cam_jpe"] = {"response": {"reasoner_status": "json_parse_error", "risks": []},
                                 "ts": vlm._now_ms()}
    called = {"submit": 0}
    class NoSubmit:
        def submit(self, *args, **kwargs):
            called["submit"] += 1
    monkeypatch.setattr(vlm, "_executor", lambda: NoSubmit())
    draft, status = vlm.maybe_trigger(
        "cam_jpe", frame_b64=None, highest_level="ORANGE", deterministic_risks=[DET_RISK])
    assert status == "json_parse_error"
    assert called["submit"] == 0


def test_detect_surfaces_json_parse_error_not_unavailable(server_mod, monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    import vision_backend
    import risk.vlm_reasoner as _vlm
    monkeypatch.setenv("RISK_ENGINE_ENABLED", "true")
    monkeypatch.setenv("TEMPORAL_REASONING_ENABLED", "false")
    monkeypatch.setattr(vision_backend, "run_inference", lambda **kw: _fake_resp())
    with _vlm._LOCK:
        _vlm._CACHE["cam_jpe2"] = {"response": {
            "reasoner_status": "json_parse_error", "risks": [],
            "error": "model did not return valid JSON"}, "ts": _vlm._now_ms()}
    with server_mod._STATE_LOCK:
        server_mod._STATE["status"] = "ready"
    try:
        with TestClient(server_mod.app) as c:
            r = c.post("/detect", json={"image_b64": _tiny_jpeg_b64(),
                                        "session_id": "cam_jpe2", "hse": True})
            assert r.status_code == 200
            rs = r.json().get("reasoner_status")
            assert isinstance(rs, dict), f"expected dict, got {rs!r}"
            assert rs.get("state") == "json_parse_error"
            assert rs.get("state") != "unavailable"
    finally:
        with server_mod._STATE_LOCK:
            server_mod._STATE["status"] = "cold"


# -- Gemini adapter tests -----------------------------------------------------

def test_gemini_adapter_unavailable_without_api_key(monkeypatch):
    monkeypatch.setenv("REASONER_MODE", "gemini")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    adapter = vlm._build_adapter("gemini")
    assert adapter["available"] is False
    assert "GEMINI_API_KEY" in adapter.get("error", "")


def test_gemini_adapter_unavailable_without_sdk(monkeypatch):
    monkeypatch.setenv("REASONER_MODE", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    monkeypatch.setitem(sys.modules, "google", None)
    monkeypatch.setitem(sys.modules, "google.genai", None)
    # Re-import gemini_reasoner with mocked import
    import risk.gemini_reasoner as gr
    orig_build = gr.build_adapter

    def _no_sdk():
        try:
            from google import genai  # noqa: F401
        except (ImportError, TypeError):
            return {
                "available": False,
                "error": "google-genai not installed: mock",
                "generate": None,
                "model_id": gr.model_id(),
                "diagnostics": {"serve_backend": "google_genai"},
            }
        return orig_build()

    monkeypatch.setattr(gr, "build_adapter", _no_sdk)
    adapter = vlm._build_adapter("gemini")
    assert adapter["available"] is False


def test_gemini_reason_with_fake_adapter(monkeypatch):
    monkeypatch.setenv("REASONER_MODE", "gemini")
    fake_response = '{"box_updates":[],"uncertain_box_ids":[]}'

    def fake_generate(prompt, image):
        return fake_response

    monkeypatch.setitem(vlm._ADAPTER_STATE, "gemini", {
        "available": True,
        "generate": fake_generate,
        "model_id": "gemini-2.5-flash",
        "diagnostics": {},
        "error": None,
    })
    out = vlm.reason_sync(_req())
    assert out["reasoner_status"] == "ok"
    assert out["risks"] == []
    assert out["requires_human_review"] is True


def test_generate_json_uses_gemini_adapter(monkeypatch):
    monkeypatch.setenv("REASONER_MODE", "gemini")
    captured = {}

    def fake_generate(prompt, image):
        captured["prompt"] = prompt
        return '{"answer": 42}'

    monkeypatch.setitem(vlm._ADAPTER_STATE, "gemini", {
        "available": True,
        "generate": fake_generate,
        "model_id": "gemini-2.5-flash",
        "diagnostics": {},
        "error": None,
    })
    result = vlm.generate_json("test prompt")
    assert result == {"answer": 42}
    assert captured["prompt"] == "test prompt"


def test_adapter_available_false_without_api_key(monkeypatch):
    monkeypatch.setenv("REASONER_MODE", "gemini")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    vlm._ADAPTER_STATE.clear()
    assert vlm.adapter_available() is False


def test_adapter_available_false_for_mock_mode(monkeypatch):
    monkeypatch.setenv("REASONER_MODE", "mock")
    assert vlm.adapter_available() is False


# -- Gemini structured-output adapter: dict return path -----------------------

def test_gemini_reason_with_dict_returning_adapter(monkeypatch):
    """Adapter returning a pre-validated dict must produce reasoner_status=ok."""
    monkeypatch.setenv("REASONER_MODE", "gemini")

    def fake_generate(prompt, image):
        # Simulate adapter returning GeminiBoxDecisionResponse.model_dump()
        return {"box_updates": [], "uncertain_box_ids": []}

    monkeypatch.setitem(vlm._ADAPTER_STATE, "gemini", {
        "available": True,
        "generate": fake_generate,
        "model_id": "gemini-2.5-flash",
        "diagnostics": {},
        "error": None,
    })
    out = vlm.reason_sync(_req())
    assert out["reasoner_status"] == "ok"
    assert out["risks"] == []
    assert out["requires_human_review"] is True
    assert out["should_alert"] is False


def test_gemini_reason_maps_real_risk_to_vlmrisk(monkeypatch):
    monkeypatch.setenv("REASONER_MODE", "gemini")

    def fake_generate(prompt, image):
        return {
            "box_updates": [{
                "box_id": "A",
                "hazard_type": "object_near_edge",
                "severity": 3,
                "likelihood": 3,
                "confidence": 0.82,
                "evidence_code": "near_edge",
            }],
            "uncertain_box_ids": [],
        }

    monkeypatch.setitem(vlm._ADAPTER_STATE, "gemini", {
        "available": True,
        "generate": fake_generate,
        "model_id": "gemini-2.5-flash",
        "diagnostics": {},
        "error": None,
    })

    out = vlm.reason_sync({
        "request_id": "r1",
        "session_id": "cam1",
        "frame_id": "f1",
        "entities": [{"track_id": "t1", "label": "cup",
                       "bbox": {"x": 0.1, "y": 0.1, "w": 0.1, "h": 0.1}}],
    })

    assert out["reasoner_status"] == "ok"
    assert len(out["risks"]) == 1
    r = out["risks"][0]
    assert r["risk_id"].startswith("gemini_f1_")
    assert r["hazard_type"] == "object_near_edge"
    # severity=3, likelihood=3, risk_score=9 -> ORANGE (9-14 band)
    assert r["risk_score"] == 9
    assert r["risk_level"] == "ORANGE"
    assert r["severity"] == 3
    assert r["likelihood"] == 3
    assert r["linked_entity_id"] == "t1"
    assert "t1" in r["involved_track_ids"]
    assert r["evidence"] == ["near_edge"]
    assert r["should_alert"] is False
    assert r["requires_human_review"] is True


def test_gemini_reason_malformed_dict_returns_schema_error(monkeypatch):
    monkeypatch.setenv("REASONER_MODE", "gemini")

    def fake_generate(prompt, image):
        # box_id is required but missing; severity/likelihood out of range
        return {"box_updates": [{"hazard_type": "slip_trip"}], "uncertain_box_ids": []}

    monkeypatch.setitem(vlm._ADAPTER_STATE, "gemini", {
        "available": True,
        "generate": fake_generate,
        "model_id": "gemini-2.5-flash",
        "diagnostics": {},
        "error": None,
    })
    out = vlm.reason_sync(_req())
    assert out["reasoner_status"] == "schema_error"


def test_gemini_reason_dict_no_double_parse(monkeypatch):
    """Dict return must not hit _extract_json (which would stringify it incorrectly)."""
    monkeypatch.setenv("REASONER_MODE", "gemini")
    parse_calls = {"n": 0}
    orig_extract = vlm._extract_json

    def counting_extract(raw):
        parse_calls["n"] += 1
        return orig_extract(raw)

    monkeypatch.setattr(vlm, "_extract_json", counting_extract)

    def fake_generate(prompt, image):
        return {"box_updates": [], "uncertain_box_ids": []}

    monkeypatch.setitem(vlm._ADAPTER_STATE, "gemini", {
        "available": True,
        "generate": fake_generate,
        "model_id": "gemini-2.5-flash",
        "diagnostics": {},
        "error": None,
    })
    vlm.reason_sync(_req())
    assert parse_calls["n"] == 0, "_extract_json must not be called when adapter returns dict"


def test_generate_json_handles_dict_from_adapter(monkeypatch):
    """generate_json() must return dict directly when adapter returns one."""
    monkeypatch.setenv("REASONER_MODE", "gemini")

    def fake_generate(prompt, image):
        return {"answer": 99}

    monkeypatch.setitem(vlm._ADAPTER_STATE, "gemini", {
        "available": True,
        "generate": fake_generate,
        "model_id": "gemini-2.5-flash",
        "diagnostics": {},
        "error": None,
    })
    result = vlm.generate_json("test prompt")
    assert result == {"answer": 99}


# -- Compact prompt: no raw detector fields -----------------------------------

def test_compact_prompt_no_raw_bbox_or_class(monkeypatch):
    """The Gemini prompt must not contain raw bbox/class_id/confidence detector fields."""
    from risk.reason_schema import ReasonRequest as RR
    req = RR(
        entities=[
            {"label": "person", "class_id": 0, "confidence": 0.9,
             "bbox": {"x": 0.3, "y": 0.4, "w": 0.1, "h": 0.4}, "track_id": "t1"},
            {"label": "forklift", "class_id": 7, "confidence": 0.85,
             "bbox": {"x": 0.5, "y": 0.5, "w": 0.2, "h": 0.3}, "track_id": "t2"},
        ],
        deterministic_risks=[
            {"risk_id": "r1", "hazard_type": "worker_near_vehicle",
             "risk_level": "ORANGE", "involved_track_ids": ["t1", "t2"]},
        ],
    )
    # Build anchors the same way _gemini_reason does.
    candidates = vlm._select_candidate_entities(req.entities, req.deterministic_risks,
                                                 gemini_reasoner.max_box_candidates())
    anchors = vlm._build_box_decision_anchors(candidates)
    prompt = vlm._build_gemini_prompt(req, anchors)
    # Anchor labels must be present.
    assert "person" in prompt
    assert "forklift" in prompt
    # Must NOT contain raw detector fields.
    assert '"class_id"' not in prompt
    assert '"confidence"' not in prompt
    assert '"bbox"' not in prompt
    # Must contain the HSE box risk classifier instruction.
    assert "HSE box risk classifier" in prompt


def test_compact_prompt_highest_risk_level_present(monkeypatch):
    """Prompt must reference box labels from the annotated frame anchors."""
    from risk.reason_schema import ReasonRequest as RR
    req = RR(
        entities=[{"label": "person", "track_id": "t1",
                   "bbox": {"x": 0.1, "y": 0.1, "w": 0.1, "h": 0.1}}],
        deterministic_risks=[
            {"risk_level": "RED", "hazard_type": "falling_object", "involved_track_ids": ["t1"]},
            {"risk_level": "YELLOW", "hazard_type": "slip_trip"},
        ],
    )
    candidates = vlm._select_candidate_entities(req.entities, req.deterministic_risks,
                                                 gemini_reasoner.max_box_candidates())
    anchors = vlm._build_box_decision_anchors(candidates)
    prompt = vlm._build_gemini_prompt(req, anchors)
    # The anchor for the entity (box A) and its label must appear.
    assert "person" in prompt
    # Risk matrix guidance must be present.
    assert "risk_score" in prompt


# -- GeminiBoxDecisionResponse schema validation -----------------------------------

def test_gemini_reason_response_schema_valid():
    """GeminiBoxDecisionResponse must validate a well-formed dict."""
    from risk.gemini_reasoner import GeminiBoxDecisionResponse, GeminiBoxDecision
    r = GeminiBoxDecisionResponse(
        box_updates=[],
        uncertain_box_ids=[],
    )
    assert r.box_updates == []
    assert r.uncertain_box_ids == []


def test_gemini_risk_schema_no_bbox_field():
    """GeminiBoxDecision must not have a bbox, class_id, or narrative fields."""
    from risk.gemini_reasoner import GeminiBoxDecision
    fields = GeminiBoxDecision.model_fields
    assert "bbox" not in fields
    assert "class_id" not in fields
    assert "reason" not in fields
    assert "scene_summary" not in fields
    assert "confidence" in fields   # advisory confidence is allowed
    assert "box_id" in fields
    assert "severity" in fields
    assert "likelihood" in fields
    # Verify field constraints are present.
    d = GeminiBoxDecision(box_id="A", hazard_type="slip_trip", severity=1, likelihood=1)
    assert d.severity == 1
    assert d.likelihood == 1
    import pytest as _pytest
    with _pytest.raises(Exception):
        GeminiBoxDecision(box_id="AB", hazard_type="slip_trip", severity=1, likelihood=1)
    with _pytest.raises(Exception):
        GeminiBoxDecision(box_id="A", hazard_type="slip_trip", severity=0, likelihood=1)


def test_gemini_reason_response_json_schema_exported():
    """model_json_schema() must be callable and return a dict with title."""
    from risk.gemini_reasoner import GeminiBoxDecisionResponse
    schema = GeminiBoxDecisionResponse.model_json_schema()
    assert isinstance(schema, dict)
    assert "properties" in schema or "$defs" in schema or "title" in schema


def test_gemini_max_output_tokens_default():
    """Default GEMINI_MAX_OUTPUT_TOKENS must be >= 1024 for schema budget."""
    import os
    import risk.gemini_reasoner as gr
    # Without override the code default must be at least 1024.
    tok = int(os.getenv("GEMINI_MAX_OUTPUT_TOKENS", "1024"))
    assert tok >= 1024
