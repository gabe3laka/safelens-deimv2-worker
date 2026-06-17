"""
tests/test_hardening.py -- production hardening (PR4+PR5).

CPU-only. Covers /ready, shared-secret auth, input guards (size + megapixels),
degradation ladder, /metrics, privacy egress guard, structured-log redaction,
graceful-shutdown gate, and Docker hardening presence.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("PIL")

import worker_guards as guards
import worker_runtime as runtime
import worker_security as security


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    monkeypatch.delenv("WORKER_SHARED_SECRET", raising=False)
    monkeypatch.delenv("RISK_ENGINE_ENABLED", raising=False)
    monkeypatch.delenv("MAX_IMAGE_MEGAPIXELS", raising=False)
    monkeypatch.delenv("MAX_REQUEST_BYTES", raising=False)
    runtime.reset_metrics()
    runtime.reset_shutdown()
    runtime.set_degradation("full")
    yield
    runtime.reset_shutdown()


def _img_b64(w=8, h=8):
    buf = io.BytesIO()
    from PIL import Image
    Image.new("RGB", (w, h), (200, 200, 200)).save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode()


@pytest.fixture()
def server_mod(monkeypatch):
    import importlib
    monkeypatch.setenv("SKIP_WARMUP", "true")
    monkeypatch.setenv("AUTO_WARMUP", "false")
    if "server" in sys.modules:
        del sys.modules["server"]
    return importlib.import_module("server")


def _fake_resp():
    from schema import BBox, Entity, InferResponse
    return InferResponse(
        entities=[Entity(label="person", class_id=0, confidence=0.9,
                         bbox=BBox(x=0.3, y=0.4, w=0.14, h=0.45), source="yolo26")],
        inference_ms=10, model="YOLO26", backend="yolo26", tasks=["det"],
        img_w=1280, img_h=720)


# -- /ready -------------------------------------------------------------------

def test_ready_503_when_model_not_loaded(server_mod):
    from fastapi.testclient import TestClient
    with TestClient(server_mod.app) as c:
        r = c.get("/ready")
        assert r.status_code == 503
        assert r.json()["ready"] is False


def test_ready_200_when_model_and_matrix_ok(server_mod):
    from fastapi.testclient import TestClient
    with TestClient(server_mod.app) as c:
        with server_mod._STATE_LOCK:
            server_mod._STATE["status"] = "ready"
        try:
            r = c.get("/ready")
            assert r.status_code == 200
            body = r.json()
            assert body["ready"] is True and body["model_loaded"] is True
        finally:
            with server_mod._STATE_LOCK:
                server_mod._STATE["status"] = "cold"


def test_ready_503_when_matrix_malformed(server_mod, monkeypatch, tmp_path):
    from fastapi.testclient import TestClient
    bad = tmp_path / "bad_matrix.json"
    bad.write_text(json.dumps({"profile": "bad", "scale": {"severity_max": 5, "likelihood_max": 5},
                               "bands": [{"level": "GREEN", "min": 1, "max": 4},
                                         {"level": "RED", "min": 7, "max": 25}]}))
    monkeypatch.setenv("RISK_ENGINE_ENABLED", "true")
    monkeypatch.setenv("RISK_MATRIX_PROFILE", str(bad))
    import risk.risk_matrix as rm
    rm.reset_cache()
    with TestClient(server_mod.app) as c:
        with server_mod._STATE_LOCK:
            server_mod._STATE["status"] = "ready"
        try:
            r = c.get("/ready")
            assert r.status_code == 503
            assert r.json()["matrix_valid"] is False
        finally:
            with server_mod._STATE_LOCK:
                server_mod._STATE["status"] = "cold"
            rm.reset_cache()


# -- shared-secret auth -------------------------------------------------------

def test_unauthenticated_protected_route_rejected(server_mod, monkeypatch):
    from fastapi.testclient import TestClient
    monkeypatch.setenv("WORKER_SHARED_SECRET", "s3cr3t")
    with TestClient(server_mod.app) as c:
        assert c.post("/detect", json={"image_b64": _img_b64()}).status_code == 401
        assert c.get("/debug/state").status_code == 401
        # correct secret passes the gate
        ok = c.get("/debug/state", headers={"x-worker-secret": "s3cr3t"})
        assert ok.status_code == 200


def test_health_and_ping_stay_public(server_mod, monkeypatch):
    from fastapi.testclient import TestClient
    monkeypatch.setenv("WORKER_SHARED_SECRET", "s3cr3t")
    with TestClient(server_mod.app) as c:
        assert c.get("/health").status_code == 200
        assert c.get("/ping").status_code == 200


def test_auth_disabled_compat_mode(server_mod):
    # No secret set -> compat/test mode: protected routes reachable.
    from fastapi.testclient import TestClient
    with TestClient(server_mod.app) as c:
        assert c.get("/debug/state").status_code == 200


# -- input protection ---------------------------------------------------------

def test_oversized_body_rejected_413(server_mod, monkeypatch):
    from fastapi.testclient import TestClient
    monkeypatch.setenv("MAX_REQUEST_BYTES", "10")   # any real body exceeds this
    with TestClient(server_mod.app) as c:
        r = c.post("/detect", json={"image_b64": _img_b64()})
        assert r.status_code == 413
        assert r.json()["error"] == "payload_too_large"


def test_too_many_megapixels_rejected_413(server_mod, monkeypatch):
    from fastapi.testclient import TestClient
    monkeypatch.setenv("MAX_IMAGE_MEGAPIXELS", "0.0001")  # 100 px cap
    with TestClient(server_mod.app) as c:
        with server_mod._STATE_LOCK:
            server_mod._STATE["status"] = "ready"
        try:
            r = c.post("/detect", json={"image_b64": _img_b64(64, 64)})  # 4096 px
            assert r.status_code == 413
            assert r.json()["error"] == "image_too_large"
        finally:
            with server_mod._STATE_LOCK:
                server_mod._STATE["status"] = "cold"


def test_bad_base64_is_4xx_not_500(server_mod):
    from fastapi.testclient import TestClient
    with TestClient(server_mod.app) as c:
        with server_mod._STATE_LOCK:
            server_mod._STATE["status"] = "ready"
        try:
            r = c.post("/detect", json={"image_b64": "not_valid_base64!!!"})
            assert r.status_code == 400
            assert r.json()["error"] in ("invalid_base64", "decode_failure")
        finally:
            with server_mod._STATE_LOCK:
                server_mod._STATE["status"] = "cold"


def test_guards_validate_image_helper():
    ok, err, info = guards.validate_image_b64(_img_b64(16, 16))
    assert ok and err is None and info["width"] == 16
    ok2, err2, _ = guards.validate_image_b64(None)
    assert not ok2 and err2 == "missing_image_b64"


# -- degradation ladder -------------------------------------------------------

def test_detect_risk_failure_preserves_detection(server_mod, monkeypatch):
    from fastapi.testclient import TestClient
    import vision_backend
    import risk as _risk
    monkeypatch.setenv("RISK_ENGINE_ENABLED", "true")
    monkeypatch.setattr(vision_backend, "run_inference", lambda **kw: _fake_resp())
    monkeypatch.setattr(_risk, "attach_risk",
                        lambda d, **k: (_ for _ in ()).throw(RuntimeError("risk boom")))
    with TestClient(server_mod.app) as c:
        with server_mod._STATE_LOCK:
            server_mod._STATE["status"] = "ready"
        try:
            r = c.post("/detect", json={"image_b64": _img_b64(), "session_id": "cam_1"})
            assert r.status_code == 200          # NOT a 500
            body = r.json()
            assert body["entities"][0]["label"] == "person"
            assert "risk_engine_error" in (body.get("warning") or "")
            assert body["degradation_mode"] in ("no_risk", "full")
        finally:
            with server_mod._STATE_LOCK:
                server_mod._STATE["status"] = "cold"


def test_degradation_surfaced_in_debug_state(server_mod):
    from fastapi.testclient import TestClient
    with TestClient(server_mod.app) as c:
        body = c.get("/debug/state").json()
        rt = body["runtime"]
        assert "degradation_mode" in rt and "degradation_ladder" in rt
        assert rt["degradation_ladder"] == ["full", "no_risk", "detect_only", "down"]
        assert "build_sha" in rt and "accepting_frames" in rt


# -- /metrics -----------------------------------------------------------------

def test_metrics_exposes_expected_names(server_mod, monkeypatch):
    from fastapi.testclient import TestClient
    import vision_backend
    monkeypatch.setenv("RISK_ENGINE_ENABLED", "true")
    monkeypatch.setattr(vision_backend, "run_inference", lambda **kw: _fake_resp())
    with TestClient(server_mod.app) as c:
        with server_mod._STATE_LOCK:
            server_mod._STATE["status"] = "ready"
        try:
            c.post("/detect", json={"image_b64": _img_b64(), "session_id": "cam_m"})
            text = c.get("/metrics").text
        finally:
            with server_mod._STATE_LOCK:
                server_mod._STATE["status"] = "cold"
    for name in ("safelens_model_ready", "safelens_ready", "safelens_active_sessions",
                 "safelens_detect_requests_total", "safelens_detect_latency_ms",
                 "safelens_degradation_rank"):
        assert name in text, name


# -- privacy egress guard -----------------------------------------------------

def test_privacy_egress_guard_blurs_persons(monkeypatch):
    np = pytest.importorskip("numpy")
    from PIL import Image
    from risk import privacy
    arr = np.zeros((80, 80, 3), dtype=np.uint8)
    arr[:, 40:] = 255
    img = Image.fromarray(arr)
    person = [{"label": "person", "bbox": {"x": 0.25, "y": 0.0, "w": 0.5, "h": 1.0}}]
    monkeypatch.setenv("PRIVACY_BLUR_ENABLED", "true")
    out, blurred = privacy.sanitize_for_egress(img, person)
    assert blurred is True
    assert np.array(out).tobytes() != np.array(img).tobytes()


def test_reasoner_receives_blurred_frame(monkeypatch):
    np = pytest.importorskip("numpy")
    from PIL import Image
    import risk.vlm_reasoner as vlm
    from risk.reason_schema import ReasonRequest
    arr = np.zeros((80, 80, 3), dtype=np.uint8)
    arr[:, 40:] = 255
    buf = io.BytesIO(); Image.fromarray(arr).save(buf, format="JPEG")
    raw_b64 = base64.b64encode(buf.getvalue()).decode()
    monkeypatch.setenv("PRIVACY_BLUR_ENABLED", "true")
    req = ReasonRequest(frame_b64=raw_b64,
                        entities=[{"label": "person", "bbox": {"x": 0.25, "y": 0.0, "w": 0.5, "h": 1.0}}])
    blurred_img = vlm._decode_blurred(req)
    raw_img = Image.open(io.BytesIO(base64.b64decode(raw_b64))).convert("RGB")
    assert np.array(blurred_img).tobytes() != np.array(raw_img).tobytes()


# -- structured-log redaction -------------------------------------------------

def test_logs_never_contain_secrets(caplog):
    with caplog.at_level(logging.INFO):
        runtime.log_event("detect", session_id="cam_1", frame_id="f1",
                          hf_token="HF_SUPER_SECRET", authorization="Bearer ABC123",
                          worker_shared_secret="topsecret",
                          image_b64="QUJDQUJD" * 200)
    text = "\n".join(r.getMessage() for r in caplog.records)
    assert "HF_SUPER_SECRET" not in text
    assert "ABC123" not in text
    assert "topsecret" not in text
    assert "QUJDQUJD" not in text       # image payload not logged
    assert "[redacted]" in text
    assert "cam_1" in text              # non-sensitive fields kept


def test_redact_helper():
    out = runtime.redact({"frame_b64": "x" * 1000, "api_key": "k", "label": "person"})
    assert out["frame_b64"] == "[redacted]"
    assert out["api_key"] == "[redacted]"
    assert out["label"] == "person"


# -- graceful shutdown --------------------------------------------------------

def test_detect_rejects_while_shutting_down(server_mod):
    from fastapi.testclient import TestClient
    with TestClient(server_mod.app) as c:
        with server_mod._STATE_LOCK:
            server_mod._STATE["status"] = "ready"
        runtime.begin_shutdown()
        try:
            r = c.post("/detect", json={"image_b64": _img_b64()})
            assert r.status_code == 503
            assert r.json()["error"] == "shutting_down"
        finally:
            runtime.reset_shutdown()
            with server_mod._STATE_LOCK:
                server_mod._STATE["status"] = "cold"


def test_ws_shutdown_all_no_sessions():
    import ws_vision
    assert ws_vision.shutdown_all() == 0   # no live sessions -> 0, never raises


# -- Docker hardening present -------------------------------------------------

def test_dockerignore_excludes_weights_and_secrets():
    di = (REPO_ROOT / ".dockerignore").read_text()
    for pat in ("*.pt", "*.safetensors", ".git", ".env", "datasets"):
        assert pat in di, pat


def test_dockerfile_hardening_markers():
    df = (REPO_ROOT / "Dockerfile").read_text()
    assert "BUILD_SHA" in df
    assert "USER " in df                 # non-root user
    assert "WORKER_SHARED_SECRET" in df  # documented auth env


def test_docker_build_if_available():
    import shutil
    import subprocess
    if shutil.which("docker") is None:
        pytest.skip("docker not available")
    if subprocess.run(["docker", "info"], capture_output=True).returncode != 0:
        pytest.skip("docker daemon not running")
    pytest.skip("full image build is heavy + network-bound; covered by GHCR CI")
