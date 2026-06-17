"""
tests/test_edgecrafter.py -- unit tests for the EdgeCrafter migration.

These tests avoid loading the real models. Heavy/optional deps (torch,
torchvision, fastapi) are imported lazily and skipped if unavailable, so the
suite runs in a minimal CI environment.
"""

import base64
import importlib
import io
import os
import sys

import pytest

# edgecrafter_loader imports torch/torchvision at module top; skip the whole
# module cleanly if those are not installed in this environment.
ec = pytest.importorskip("edgecrafter_loader")


# -- Env parsing --------------------------------------------------------------

def test_parse_tasks_default():
    assert ec.parse_tasks("det,pose") == ["det", "pose"]


def test_parse_tasks_whitespace_and_case():
    assert ec.parse_tasks(" DET , Pose ") == ["det", "pose"]


def test_parse_tasks_dedup_and_filter_unknown():
    # 'seg' is reserved for later and must be filtered out; duplicates removed.
    assert ec.parse_tasks("det,det,seg,pose") == ["det", "pose"]


def test_parse_tasks_empty_falls_back_to_det():
    assert ec.parse_tasks("") == ["det"]
    assert ec.parse_tasks("seg") == ["det"]


def test_parse_tasks_reads_env(monkeypatch):
    monkeypatch.setenv("EDGECRAFTER_TASKS", "pose")
    assert ec.parse_tasks() == ["pose"]


# -- Detection bbox normalization ---------------------------------------------

def test_normalize_bbox_basic():
    b = ec.normalize_bbox_xyxy(64, 48, 192, 240, 640, 480)
    assert b["x"] == pytest.approx(0.1)
    assert b["y"] == pytest.approx(0.1)
    assert b["w"] == pytest.approx(0.2)
    assert b["h"] == pytest.approx(0.4)


def test_normalize_bbox_clamps_to_unit_square():
    b = ec.normalize_bbox_xyxy(-10, -10, 1000, 1000, 640, 480)
    assert 0.0 <= b["x"] <= 1.0
    assert 0.0 <= b["y"] <= 1.0
    assert b["x"] + b["w"] <= 1.0 + 1e-6
    assert b["y"] + b["h"] <= 1.0 + 1e-6


# -- Pose keypoint decoding / normalization -----------------------------------

def test_decode_keypoints_k3():
    np = pytest.importorskip("numpy")
    arr = np.array([[100.0, 200.0, 0.9], [50.0, 60.0, 0.8]])
    xy, sc = ec._decode_keypoints(arr)
    assert xy.shape == (2, 2)
    assert sc.tolist() == pytest.approx([0.9, 0.8])


def test_decode_keypoints_k2_defaults_scores_to_one():
    np = pytest.importorskip("numpy")
    arr = np.array([[100.0, 200.0], [50.0, 60.0]])
    xy, sc = ec._decode_keypoints(arr)
    assert sc.tolist() == [1.0, 1.0]


def test_coco_keypoint_names_count_is_17():
    assert len(ec.COCO_KEYPOINT_NAMES) == 17
    assert ec.COCO_KEYPOINT_NAMES[0] == "nose"


def test_coco_skeleton_is_zero_based_pairs():
    assert all(len(e) == 2 for e in ec.COCO_SKELETON)
    flat = [i for e in ec.COCO_SKELETON for i in e]
    assert min(flat) >= 0
    assert max(flat) <= 16


def test_coco_label_person_is_zero():
    assert ec.coco_label(0) == "person"
    assert ec.coco_label(999).startswith("class_")


# -- Checkpoint path creation + download-skip helper --------------------------

def test_ensure_checkpoint_skips_existing(tmp_path, monkeypatch):
    ckpt = tmp_path / "ecdet_s.pth"
    ckpt.write_bytes(b"already-here")

    calls = {"n": 0}

    def _fail_download(url, dst):  # pragma: no cover - must not be called
        calls["n"] += 1
        raise AssertionError("download should be skipped for existing file")

    monkeypatch.setattr(ec.urllib.request, "urlretrieve", _fail_download)
    out = ec.ensure_checkpoint("http://example/ecdet_s.pth", str(ckpt))
    assert out == str(ckpt)
    assert calls["n"] == 0


def test_ensure_checkpoint_creates_parent_and_downloads(tmp_path, monkeypatch):
    dst = tmp_path / "nested" / "dir" / "ecpose_s.pth"

    def _fake_download(url, target):
        # parent dir must already exist by the time we are called
        assert os.path.isdir(os.path.dirname(target))
        with open(target, "wb") as fh:
            fh.write(b"downloaded-bytes")

    monkeypatch.setattr(ec.urllib.request, "urlretrieve", _fake_download)
    out = ec.ensure_checkpoint("http://example/ecpose_s.pth", str(dst))
    assert os.path.exists(out)
    assert dst.read_bytes() == b"downloaded-bytes"


# -- Server routes (FastAPI TestClient) ---------------------------------------

@pytest.fixture()
def client(monkeypatch):
    fastapi_testclient = pytest.importorskip("fastapi.testclient")
    monkeypatch.setenv("SKIP_WARMUP", "true")
    monkeypatch.setenv("AUTO_WARMUP", "false")
    monkeypatch.setenv("VISION_BACKEND", "edgecrafter")
    # Re-import server fresh so env + module state are clean.
    for mod in ("server",):
        if mod in sys.modules:
            del sys.modules[mod]
    server = importlib.import_module("server")
    importlib.reload(server)
    return server, fastapi_testclient.TestClient(server.app)


def test_health_without_model(client):
    _server, c = client
    r = c.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["backend"] == "edgecrafter"
    assert body["model_loaded"] is False


def test_detect_cold_returns_model_not_ready(client):
    _server, c = client
    img_b64 = base64.b64encode(b"not-a-real-jpeg").decode()
    r = c.post("/detect", json={"image_b64": img_b64})
    assert r.status_code == 503
    body = r.json()
    assert body["error"] == "model_not_ready"
    assert body["entities"] == []
    assert body["poses"] == []


def test_debug_model_load_structured_failure(client, monkeypatch):
    server, c = client

    def _boom():
        raise RuntimeError("checkpoint missing")

    import vision_backend
    monkeypatch.setattr(vision_backend, "load_models", _boom)
    r = c.post("/debug/model-load")
    assert r.status_code == 200  # never crashes
    body = r.json()
    assert body["ok"] is False
    assert body["backend"] == "edgecrafter"
    assert body["exception_type"] == "RuntimeError"


def test_debug_model_load_structured_success(client, monkeypatch):
    server, c = client

    def _ok():
        return {
            "ok": True, "backend": "edgecrafter",
            "tasks_loaded": ["det", "pose"],
            "model_classes": {"det": "DetModel", "pose": "PoseModel"},
            "checkpoint_paths": {"det": "/x/ecdet_s.pth", "pose": "/x/ecpose_s.pth"},
            "device": "cpu",
        }

    import vision_backend
    monkeypatch.setattr(vision_backend, "load_models", _ok)
    r = c.post("/debug/model-load")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["backend"] == "edgecrafter"
    assert body["tasks_loaded"] == ["det", "pose"]


def test_detect_mocked_returns_entities_and_poses(client, monkeypatch):
    server, c = client
    from schema import BBox, Entity, Keypoint, Pose, InferResponse

    fake = InferResponse(
        entities=[Entity(label="person", class_id=0, confidence=0.88,
                         bbox=BBox(x=0.12, y=0.10, w=0.32, h=0.72),
                         source="edgecrafter-det")],
        poses=[Pose(label="person", confidence=0.84,
                    keypoints=[Keypoint(name="nose", x=0.31, y=0.18, score=0.91)],
                    skeleton=[[5, 7], [7, 9], [6, 8]], source="edgecrafter-pose")],
        inference_ms=12.3, model="EdgeCrafter", backend="edgecrafter",
        tasks=["det", "pose"], img_w=640, img_h=480,
    )

    # Force the server into the ready state and stub run_inference.
    with server._STATE_LOCK:
        server._STATE["status"] = "ready"

    import vision_backend
    monkeypatch.setattr(vision_backend, "run_inference",
                        lambda **kw: fake)

    import io as _io
    from PIL import Image as _Image
    _buf = _io.BytesIO(); _Image.new("RGB", (8, 8), (200, 200, 200)).save(_buf, format="JPEG")
    img_b64 = base64.b64encode(_buf.getvalue()).decode()  # real image (input guard decodes header)
    r = c.post("/detect", json={"image_b64": img_b64, "conf": 0.25, "img_size": 640})
    assert r.status_code == 200
    body = r.json()
    assert body["backend"] == "edgecrafter"
    assert body["tasks"] == ["det", "pose"]
    assert len(body["entities"]) == 1
    assert body["entities"][0]["source"] == "edgecrafter-det"
    assert len(body["poses"]) == 1
    assert body["poses"][0]["keypoints"][0]["name"] == "nose"
    assert body["poses"][0]["source"] == "edgecrafter-pose"
