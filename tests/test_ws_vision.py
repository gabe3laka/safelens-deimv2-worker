"""
tests/test_ws_vision.py -- tests for the /ws/vision streaming WebSocket layer.

All tests run on CPU with MOCK inference (no torch / no GPU / no model weights):

  * /ws/echo still works                       (existing Phase-0 probe untouched)
  * /ws/vision connects locally                (connected -> ready)
  * cold model warms up then signals ready     (warming -> ready via injected warmup)
  * invalid JSON does not crash the stream
  * frame message schema validates
  * latest-frame-wins drops stale frames
  * metrics update + GET /debug/stream
  * HTTP /detect still works (and 503 when cold)
  * the real server.app registers /ws/vision + /debug/stream and reuses /detect's path

The streaming module injects all model interaction, so these tests stub it with
a fake InferResponse -- the same contract /detect returns.
"""

from __future__ import annotations

import asyncio
import base64
import importlib
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

import pytest

pytest.importorskip("fastapi")
from fastapi import FastAPI
from fastapi.testclient import TestClient

import ws_vision
from schema import BBox, Entity, Keypoint, Pose, InferResponse

# The WS path only checks base64 validity (inference is mocked), but /detect's
# input guard now decodes the image header -- so use a real (tiny) JPEG here.
def _real_jpeg_b64():
    import io
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), (210, 210, 210)).save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode()

_FRAME_B64 = _real_jpeg_b64()


def _fake_response() -> InferResponse:
    """Mirror the contract /detect returns (one entity + one pose)."""
    return InferResponse(
        entities=[Entity(label="person", class_id=0, confidence=0.88,
                         bbox=BBox(x=0.1, y=0.1, w=0.2, h=0.6),
                         source="edgecrafter-det")],
        poses=[Pose(label="person", confidence=0.8,
                    keypoints=[Keypoint(name="nose", x=0.3, y=0.2, score=0.9)],
                    skeleton=[[5, 7], [7, 9]], source="edgecrafter-pose")],
        inference_ms=4.2, model="EdgeCrafter", backend="edgecrafter",
        tasks=["det", "pose"], img_w=640, img_h=480,
    )


def _fake_infer(image_b64=None, conf=None, img_size=None, class_filter=None):
    return _fake_response()


def _make_client(state, *, run_inference=None, backend="edgecrafter",
                 tasks=("det", "pose"), gpu=None, trigger_warmup=None,
                 metrics_interval_s=0.1):
    """Build a fresh FastAPI app with /ws/vision registered + injected fakes."""
    app = FastAPI()
    ws_vision.register_ws_vision(
        app,
        get_state=lambda: state,
        trigger_warmup=trigger_warmup or (lambda: None),
        run_inference=run_inference or _fake_infer,
        get_backend=lambda: backend,
        get_tasks=lambda: list(tasks),
        get_gpu_device=lambda: gpu,
        metrics_interval_s=metrics_interval_s,
        warmup_timeout_s=5.0,
        warmup_poll_s=0.02,
    )
    return TestClient(app)


def _collect(ws, want_types, max_msgs=40):
    """Read messages until every type in want_types is seen (or max_msgs)."""
    seen = {}
    for _ in range(max_msgs):
        msg = ws.receive_json()
        seen.setdefault(msg.get("type"), msg)
        if all(t in seen for t in want_types):
            break
    return seen


# ── 1. Frame validation (pure unit) ──────────────────────────────────────────

def test_validate_frame_accepts_valid():
    ok, err = ws_vision.validate_frame(
        {"type": "frame", "camera_id": "c", "frame_id": 1, "frame_b64": _FRAME_B64})
    assert ok is True and err is None


def test_validate_frame_missing_b64():
    ok, err = ws_vision.validate_frame({"type": "frame", "frame_id": 1})
    assert ok is False and err == "missing_frame_b64"


def test_validate_frame_invalid_b64():
    ok, err = ws_vision.validate_frame({"type": "frame", "frame_b64": "!!!notb64!!!"})
    assert ok is False and err == "invalid_base64"


def test_validate_frame_wrong_type():
    ok, err = ws_vision.validate_frame({"type": "ping", "frame_b64": _FRAME_B64})
    assert ok is False and err == "invalid_frame_type"


def test_validate_frame_non_dict():
    ok, err = ws_vision.validate_frame(["not", "a", "dict"])
    assert ok is False and err == "invalid_frame"


# ── 2. Latest-frame-wins slot (deterministic) ────────────────────────────────

def test_latest_frame_slot_drops_stale_and_keeps_latest():
    async def _run():
        slot = ws_vision._LatestFrameSlot()
        assert slot.put({"frame_id": 1}) is False      # nothing replaced
        assert slot.depth() == 1
        assert slot.put({"frame_id": 2}) is True        # replaced -> stale frame 1 dropped
        assert slot.put({"frame_id": 3}) is True        # replaced -> stale frame 2 dropped
        frame = await slot.get()                         # latest wins
        assert frame["frame_id"] == 3
        assert slot.depth() == 0
    asyncio.run(_run())


def test_latest_frame_slot_get_timeout_returns_none():
    async def _run():
        slot = ws_vision._LatestFrameSlot()
        assert await slot.get(timeout=0.05) is None
    asyncio.run(_run())


# ── 3. Rate window + metrics snapshot (pure unit) ─────────────────────────────

def test_rate_window_reports_positive_fps():
    rw = ws_vision._RateWindow(window_s=5.0)
    now = time.monotonic()
    for i in range(5):
        rw.mark(now + i * 0.1)
    assert rw.fps(now + 0.4) > 0.0


def test_metrics_snapshot_has_required_fields():
    state = {"status": "ready", "model_loaded": True}
    session = ws_vision._VisionStreamSession(
        None,
        get_state=lambda: state,
        trigger_warmup=lambda: None,
        run_inference=_fake_infer,
        get_backend=lambda: "edgecrafter",
        get_tasks=lambda: ["det", "pose"],
        get_gpu_device=lambda: "NVIDIA RTX A5000",
        default_conf=0.25, default_img_size=640,
        metrics_interval_s=2.0, warmup_timeout_s=600.0, warmup_poll_s=0.5,
    )
    session.received_frames = 10
    session.processed_frames = 7
    session.dropped_frames = 3
    snap = session.metrics_snapshot()
    for key in ("type", "received_fps", "processed_fps", "dropped_frames",
                "avg_inference_ms", "avg_end_to_end_latency_ms",
                "current_queue_depth", "model_ready", "backend", "tasks",
                "gpu_device"):
        assert key in snap, key
    assert snap["type"] == "metrics"
    assert snap["model_ready"] is True
    assert snap["backend"] == "edgecrafter"
    assert snap["tasks"] == ["det", "pose"]
    assert snap["dropped_frames"] == 3
    assert snap["gpu_device"] == "NVIDIA RTX A5000"


# ── 4. /ws/vision connect + ready ─────────────────────────────────────────────

def test_ws_vision_connect_then_ready():
    state = {"status": "ready", "model_loaded": True}
    client = _make_client(state, metrics_interval_s=5.0)
    with client.websocket_connect("/ws/vision") as ws:
        assert ws.receive_json() == {"type": "connected"}
        ready = ws.receive_json()
        assert ready["type"] == "ready"
        assert ready["backend"] == "edgecrafter"
        assert ready["tasks"] == ["det", "pose"]


def test_ws_vision_cold_warms_up_then_ready():
    state = {"status": "cold", "model_loaded": False}

    def warmup():
        # Simulate the existing background warmup completing.
        state["status"] = "ready"
        state["model_loaded"] = True

    client = _make_client(state, trigger_warmup=warmup, metrics_interval_s=5.0)
    with client.websocket_connect("/ws/vision") as ws:
        assert ws.receive_json()["type"] == "connected"
        seen = _collect(ws, ["warming", "ready"])
        assert "warming" in seen
        assert seen["ready"]["backend"] == "edgecrafter"
        assert seen["ready"]["tasks"] == ["det", "pose"]


# ── 5. Robustness: invalid JSON + bad frame schema ───────────────────────────

def test_ws_vision_invalid_json_does_not_crash():
    state = {"status": "ready", "model_loaded": True}
    client = _make_client(state, metrics_interval_s=5.0)
    with client.websocket_connect("/ws/vision") as ws:
        assert ws.receive_json()["type"] == "connected"
        _collect(ws, ["ready"])
        ws.send_text("{ this is not valid json ")
        err = ws.receive_json()
        assert err["type"] == "error"
        assert err["error"] == "invalid_json"
        # Connection survives -- a valid frame still produces a vision result.
        ws.send_json({"type": "frame", "frame_id": 1, "frame_b64": _FRAME_B64})
        assert ws.receive_json()["type"] == "vision"


def test_ws_vision_bad_frame_reports_error():
    state = {"status": "ready", "model_loaded": True}
    client = _make_client(state, metrics_interval_s=5.0)
    with client.websocket_connect("/ws/vision") as ws:
        assert ws.receive_json()["type"] == "connected"
        _collect(ws, ["ready"])
        ws.send_json({"type": "frame", "frame_id": 1, "frame_b64": "!!!notb64!!!"})
        assert ws.receive_json()["error"] == "invalid_base64"
        ws.send_json({"type": "frame", "frame_id": 2})  # missing frame_b64
        assert ws.receive_json()["error"] == "missing_frame_b64"


# ── 6. Frame -> vision message schema ─────────────────────────────────────────

def test_ws_vision_frame_returns_vision_message():
    state = {"status": "ready", "model_loaded": True}
    client = _make_client(state, metrics_interval_s=5.0)  # keep metrics out of the way
    with client.websocket_connect("/ws/vision") as ws:
        assert ws.receive_json()["type"] == "connected"
        _collect(ws, ["ready"])
        ws.send_json({"type": "frame", "camera_id": "browser-test", "frame_id": 123,
                      "sent_at": 1710000000000, "frame_b64": _FRAME_B64})
        v = ws.receive_json()
        assert v["type"] == "vision"
        # Critical output schema -- must carry everything /detect returns, plus stream fields.
        for key in ("camera_id", "frame_id", "backend", "tasks", "entities", "poses",
                    "model", "inference_ms", "img_w", "img_h",
                    "received_at", "completed_at", "end_to_end_latency_ms"):
            assert key in v, key
        assert v["camera_id"] == "browser-test"
        assert v["frame_id"] == 123
        assert v["backend"] == "edgecrafter"
        assert v["tasks"] == ["det", "pose"]
        assert v["model"] == "EdgeCrafter"
        assert v["img_w"] == 640 and v["img_h"] == 480
        assert len(v["entities"]) == 1 and v["entities"][0]["source"] == "edgecrafter-det"
        assert len(v["poses"]) == 1 and v["poses"][0]["keypoints"][0]["name"] == "nose"
        assert v["end_to_end_latency_ms"] >= 0


# ── 7. Metrics update over the stream ─────────────────────────────────────────

def test_ws_vision_metrics_update():
    state = {"status": "ready", "model_loaded": True}
    client = _make_client(state, metrics_interval_s=0.1)
    with client.websocket_connect("/ws/vision") as ws:
        assert ws.receive_json()["type"] == "connected"
        _collect(ws, ["ready"])
        ws.send_json({"type": "frame", "camera_id": "browser-test", "frame_id": 1,
                      "sent_at": int(time.time() * 1000), "frame_b64": _FRAME_B64})
        # Metrics flow continuously; read until one reports the processed frame.
        processed = 0
        metric = None
        for _ in range(40):
            m = ws.receive_json()
            if m.get("type") == "metrics":
                metric = m
                processed = m["processed_frames"]
                if processed >= 1:
                    break
        assert metric is not None
        for key in ("received_fps", "processed_fps", "dropped_frames",
                    "avg_inference_ms", "avg_end_to_end_latency_ms",
                    "current_queue_depth", "model_ready", "backend", "tasks",
                    "gpu_device"):
            assert key in metric, key
        assert processed >= 1
        assert metric["model_ready"] is True


def test_ws_vision_latest_frame_wins_drops_under_load():
    state = {"status": "ready", "model_loaded": True}

    def slow_infer(image_b64=None, conf=None, img_size=None, class_filter=None):
        time.sleep(0.03)  # inference slower than arrival -> backlog -> drops
        return _fake_response()

    client = _make_client(state, run_inference=slow_infer, metrics_interval_s=0.1)
    with client.websocket_connect("/ws/vision") as ws:
        assert ws.receive_json()["type"] == "connected"
        _collect(ws, ["ready"])
        for i in range(15):
            ws.send_json({"type": "frame", "camera_id": "c", "frame_id": i,
                          "sent_at": int(time.time() * 1000), "frame_b64": _FRAME_B64})
        received = dropped = 0
        for _ in range(80):
            m = ws.receive_json()
            if m.get("type") == "metrics":
                received = m["received_frames"]
                dropped = m["dropped_frames"]
                if received >= 15:
                    break
        assert received == 15
        assert dropped >= 1  # stale frames were dropped under load


# ── 8. GET /debug/stream ──────────────────────────────────────────────────────

def test_debug_stream_endpoint():
    state = {"status": "ready", "model_loaded": True}
    client = _make_client(state)
    body = client.get("/debug/stream").json()
    assert body["ok"] is True
    assert "active_connections" in body
    assert "metrics" in body
    assert "totals" in body


# ── 9. Real server wiring (server.app) ────────────────────────────────────────

@pytest.fixture()
def server_mod():
    os.environ["SKIP_WARMUP"] = "true"
    os.environ["AUTO_WARMUP"] = "false"
    os.environ["VISION_BACKEND"] = "edgecrafter"
    os.environ["EDGECRAFTER_TASKS"] = "det,pose"
    os.environ["WS_METRICS_INTERVAL_S"] = "0.2"
    # Clear backend state left behind by other test files (e.g. legacy
    # /debug/model-load tests can REALLY load yolo26 when ultralytics is
    # installed locally), so this file's edgecrafter env actually applies.
    import vision_backend
    vision_backend._BACKEND_STATE.update(
        requested=None, active=None, fallback_active=False, fallback_reason=None)
    if "server" in sys.modules:
        del sys.modules["server"]
    return importlib.import_module("server")


def test_server_registers_all_routes(server_mod):
    paths = {getattr(r, "path", None) for r in server_mod.app.routes}
    for p in ("/health", "/ping", "/debug/startup", "/debug/model-load", "/warmup",
              "/detect", "/ws/echo", "/ws/vision", "/debug/stream"):
        assert p in paths, p


def test_ws_echo_still_works(server_mod):
    with TestClient(server_mod.app) as client:
        with client.websocket_connect("/ws/echo") as ws:
            assert ws.receive_json() == {"type": "connected"}
            ws.send_json({"type": "x", "v": 5})
            assert ws.receive_json() == {"type": "x", "v": 5}


def test_detect_still_works_when_ready(server_mod, monkeypatch):
    import vision_backend
    monkeypatch.setattr(vision_backend, "run_inference", lambda **kw: _fake_response())
    with server_mod._STATE_LOCK:
        server_mod._STATE["status"] = "ready"
    try:
        with TestClient(server_mod.app) as client:
            r = client.post("/detect",
                            json={"image_b64": _FRAME_B64, "conf": 0.25, "img_size": 640})
            assert r.status_code == 200
            body = r.json()
            assert body["backend"] == "edgecrafter"
            assert body["tasks"] == ["det", "pose"]
            assert body["model"] == "EdgeCrafter"
            for key in ("entities", "poses", "inference_ms", "img_w", "img_h"):
                assert key in body
            assert len(body["entities"]) == 1
    finally:
        with server_mod._STATE_LOCK:
            server_mod._STATE["status"] = "cold"


def test_detect_cold_returns_503(server_mod):
    with server_mod._STATE_LOCK:
        server_mod._STATE["status"] = "cold"
    with TestClient(server_mod.app) as client:
        r = client.post("/detect", json={"image_b64": _FRAME_B64})
        assert r.status_code == 503
        body = r.json()
        assert body["error"] == "model_not_ready"
        assert body["entities"] == [] and body["poses"] == []


def test_real_server_ws_vision_roundtrip(server_mod, monkeypatch):
    import vision_backend
    monkeypatch.setattr(vision_backend, "run_inference", lambda **kw: _fake_response())
    with server_mod._STATE_LOCK:
        server_mod._STATE["status"] = "ready"
    try:
        with TestClient(server_mod.app) as client:
            with client.websocket_connect("/ws/vision") as ws:
                assert ws.receive_json()["type"] == "connected"
                seen = _collect(ws, ["ready"])
                assert seen["ready"]["backend"] == "edgecrafter"
                assert seen["ready"]["tasks"] == ["det", "pose"]
                ws.send_json({"type": "frame", "camera_id": "browser-test", "frame_id": 9,
                              "sent_at": int(time.time() * 1000), "frame_b64": _FRAME_B64})
                v = _collect(ws, ["vision"], max_msgs=20)["vision"]
                assert v["frame_id"] == 9
                assert v["backend"] == "edgecrafter"
                assert len(v["entities"]) == 1
    finally:
        with server_mod._STATE_LOCK:
            server_mod._STATE["status"] = "cold"
