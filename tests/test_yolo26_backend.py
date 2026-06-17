"""
tests/test_yolo26_backend.py -- YOLO26-default / EdgeCrafter-fallback migration.

All tests run on CPU with mocks (no ultralytics, no GPU, no weights):

  1.  VISION_BACKEND=yolo26 routes load_models to the YOLO adapter
  2.  YOLO detections normalize into entities (0..1 bbox, source=yolo26)
  3.  YOLO pose output normalizes into poses (COCO-17 names)
  4.  YOLO seg contours are optional + normalized when present
  5.  YOLO load failure + AUTO_BACKEND_FALLBACK=true -> EdgeCrafter fallback
  6.  Fallback /detect keeps the same entity/pose shape (+ warning)
  7.  /build/session/* still works
  8/9. Plan Mode userIntent planSteps + planOverlays still work
  10. /detect route name + payload contract unchanged
"""

from __future__ import annotations

import base64
import importlib.util
import sys
import types
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("PIL")

import vision_backend
import yolo26_loader
from schema import BBox, Entity, InferResponse, Keypoint, Pose

_B64 = base64.b64encode(b"\xff\xd8\xff\xe0fake-jpeg").decode()


@pytest.fixture(autouse=True)
def _reset_backend_state(monkeypatch):
    if importlib.util.find_spec("torch") is None:
        fake_ec = types.ModuleType("edgecrafter_loader")
        fake_ec._STATE = types.SimpleNamespace(tasks=[], loaded=False)
        fake_ec.load = lambda: {
            "backend": "edgecrafter",
            "tasks_loaded": [],
            "model_classes": {},
            "checkpoint_paths": {},
            "device": "cpu",
        }
        fake_ec.infer = lambda pil, conf, class_filter=None: {
            "entities": [], "poses": [], "inference_ms": 0.0}
        fake_ec.is_ready = lambda: fake_ec._STATE.loaded
        monkeypatch.setitem(sys.modules, "edgecrafter_loader", fake_ec)

    monkeypatch.setenv("VISION_BACKEND", "yolo26")
    monkeypatch.setenv("FALLBACK_VISION_BACKEND", "edgecrafter")
    monkeypatch.setenv("AUTO_BACKEND_FALLBACK", "true")
    vision_backend.last_detect_effective_config = None
    vision_backend._BACKEND_STATE.update(
        requested=None, active=None, fallback_active=False, fallback_reason=None)
    yield
    vision_backend.last_detect_effective_config = None
    vision_backend._BACKEND_STATE.update(
        requested=None, active=None, fallback_active=False, fallback_reason=None)


# -- 1. backend routing --------------------------------------------------------

def test_active_backend_defaults_to_yolo26(monkeypatch):
    monkeypatch.delenv("VISION_BACKEND", raising=False)
    assert vision_backend.active_backend() == "yolo26"


def test_yolo26_routes_to_adapter(monkeypatch):
    called = {}

    def _fake_load(*a, **k):
        called["load"] = True
        return {"backend": "yolo26", "tasks_loaded": ["det", "pose"],
                "model_classes": {}, "model_ids": {}, "device": "cpu",
                "warnings": []}

    monkeypatch.setattr(yolo26_loader, "load", _fake_load)
    summary = vision_backend.load_models()
    assert called["load"] is True
    assert summary["active_backend"] == "yolo26"
    assert summary["fallback_active"] is False
    assert vision_backend.serving_backend() == "yolo26"


# -- 2-4. normalization helpers (pure, no ultralytics) --------------------------

def test_detection_normalization():
    ents = yolo26_loader.normalize_detections(
        boxes_xyxy=[(64, 48, 192, 240)], class_ids=[0], scores=[0.92],
        names={0: "person"}, img_w=640, img_h=480)
    assert len(ents) == 1
    e = ents[0]
    assert e["label"] == "person" and e["class_id"] == 0
    assert e["source"] == "yolo26"
    assert e["bbox"]["x"] == pytest.approx(0.1)
    assert e["bbox"]["y"] == pytest.approx(0.1)
    assert e["bbox"]["w"] == pytest.approx(0.2)
    assert e["bbox"]["h"] == pytest.approx(0.4)
    for v in e["bbox"].values():
        assert 0.0 <= v <= 1.0


def test_detection_class_filter():
    ents = yolo26_loader.normalize_detections(
        [(0, 0, 10, 10), (0, 0, 20, 20)], [0, 2], [0.9, 0.8],
        {0: "person", 2: "car"}, 100, 100, class_filter=[2])
    assert len(ents) == 1 and ents[0]["class_id"] == 2


def test_pose_normalization():
    kpts = [[(320, 240)] * 17]
    conf = [[0.9] * 17]
    poses = yolo26_loader.normalize_poses(kpts, conf, [0.88], 640, 480)
    assert len(poses) == 1
    p = poses[0]
    assert p["label"] == "person" and p["confidence"] == pytest.approx(0.88)
    assert p["source"] == "yolo26-pose"
    assert len(p["keypoints"]) == 17
    assert p["keypoints"][0]["name"] == "nose"
    assert p["keypoints"][9]["name"] == "left_wrist"
    assert p["keypoints"][0]["x"] == pytest.approx(0.5)
    assert p["keypoints"][0]["y"] == pytest.approx(0.5)
    assert p["skeleton"] == yolo26_loader.COCO_SKELETON


def test_segment_normalization_optional():
    segs = yolo26_loader.normalize_segments(
        [[(64, 48), (192, 48), (192, 240), (64, 240)]], [0], [0.8],
        {0: "person"}, 640, 480)
    assert len(segs) == 1
    s = segs[0]
    assert s["source"] == "yolo26-seg"
    assert len(s["maskContour"]) == 4
    for p in s["maskContour"]:
        assert 0.0 <= p["x"] <= 1.0 and 0.0 <= p["y"] <= 1.0
    # degenerate polygons are dropped, not crashed
    assert yolo26_loader.normalize_segments([[(1, 1)]], [0], [0.5], {}, 100, 100) == []


# -- 5. auto fallback to EdgeCrafter --------------------------------------------

def _boom(*a, **k):
    raise RuntimeError("simulated yolo26 load failure")


def _ec_ok(*a, **k):
    return {"backend": "edgecrafter", "tasks_loaded": ["det", "pose"],
            "model_classes": {}, "checkpoint_paths": {}, "device": "cpu"}


def test_yolo_failure_falls_back_to_edgecrafter(monkeypatch):
    import edgecrafter_loader as ec
    monkeypatch.setattr(yolo26_loader, "load", _boom)
    monkeypatch.setattr(ec, "load", _ec_ok)
    summary = vision_backend.load_models()
    assert summary["active_backend"] == "edgecrafter"
    assert summary["requested_backend"] == "yolo26"
    assert summary["fallback_active"] is True
    assert "simulated yolo26 load failure" in summary["fallback_reason"]
    assert vision_backend.serving_backend() == "edgecrafter"
    status = vision_backend.backend_status()
    assert status["fallback_active"] is True
    assert status["active_backend"] == "edgecrafter"


def test_yolo_failure_no_fallback_raises(monkeypatch):
    monkeypatch.setenv("AUTO_BACKEND_FALLBACK", "false")
    monkeypatch.setattr(yolo26_loader, "load", _boom)
    with pytest.raises(RuntimeError):
        vision_backend.load_models()


# -- 6. fallback /detect keeps the entity/pose shape ----------------------------

def _fake_ec_infer(pil, conf, class_filter=None):
    return {
        "entities": [{"label": "person", "class_id": 0, "confidence": 0.9,
                      "bbox": {"x": 0.1, "y": 0.1, "w": 0.2, "h": 0.4},
                      "source": "edgecrafter-det"}],
        "poses": [{"label": "person", "confidence": 0.8,
                   "keypoints": [{"name": "nose", "x": 0.5, "y": 0.4, "score": 0.9}],
                   "skeleton": [[5, 7]], "source": "edgecrafter-pose"}],
        "inference_ms": 9.0,
    }


def _tiny_jpeg_b64():
    import io
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), (255, 255, 255)).save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode()


def test_fallback_detect_shape_and_warning(monkeypatch):
    import edgecrafter_loader as ec
    monkeypatch.setattr(yolo26_loader, "load", _boom)
    monkeypatch.setattr(ec, "load", _ec_ok)
    vision_backend.load_models()

    monkeypatch.setattr(ec, "infer", _fake_ec_infer)
    monkeypatch.setattr(ec._STATE, "tasks", ["det", "pose"], raising=False)
    resp = vision_backend.run_inference(image_b64=_tiny_jpeg_b64(), conf=0.25)
    body = resp.model_dump()
    # Same app-facing contract.
    for key in ("entities", "poses", "backend", "tasks", "model",
                "inference_ms", "img_w", "img_h"):
        assert key in body, key
    assert body["backend"] == "edgecrafter"
    assert body["entities"][0]["bbox"] == {"x": 0.1, "y": 0.1, "w": 0.2, "h": 0.4}
    assert body["poses"][0]["keypoints"][0]["name"] == "nose"
    assert body["warning"] and "backend_fallback" in body["warning"]


def test_yolo26_response_shape(monkeypatch):
    monkeypatch.setattr(
        yolo26_loader, "infer",
        lambda pil, conf, class_filter=None, **kwargs: {
            "entities": [{"label": "person", "class_id": 0, "confidence": 0.92,
                          "bbox": {"x": 0.1, "y": 0.2, "w": 0.3, "h": 0.4},
                          "source": "yolo26"}],
            "poses": [{"label": "person", "confidence": 0.88,
                       "keypoints": [{"name": "left_wrist", "x": 0.4, "y": 0.6, "score": 0.9}],
                       "skeleton": [], "source": "yolo26-pose"}],
            "segments": [{"label": "person", "class_id": 0, "confidence": 0.9,
                          "maskContour": [{"x": 0.2, "y": 0.3}, {"x": 0.25, "y": 0.34},
                                          {"x": 0.3, "y": 0.4}],
                          "source": "yolo26-seg"}],
            "inference_ms": 22, "tasks": ["det", "seg", "pose"], "model": "YOLO26",
        })
    vision_backend._BACKEND_STATE.update(active="yolo26", fallback_active=False)
    body = vision_backend.run_inference(image_b64=_tiny_jpeg_b64(), conf=0.25).model_dump()
    assert body["backend"] == "yolo26" and body["model"] == "YOLO26"
    assert body["tasks"] == ["det", "seg", "pose"]
    assert body["entities"][0]["source"] == "yolo26"
    assert body["poses"][0]["keypoints"][0]["name"] == "left_wrist"
    assert body["segments"][0]["source"] == "yolo26-seg"
    assert len(body["segments"][0]["maskContour"]) == 3
    assert body["warning"] is None


# -- Backend-specific effective config ----------------------------------------

def _empty_yolo_infer(called):
    def _infer(pil, conf, class_filter=None, tasks=None, img_size=None,
               iou=None, max_det=None):
        called.update(
            conf=conf,
            img_size=img_size,
            iou=iou,
            max_det=max_det,
            class_filter=class_filter,
        )
        return {
            "entities": [],
            "poses": [],
            "segments": [],
            "inference_ms": 1.0,
            "tasks": ["det"],
            "model": "YOLO26",
        }
    return _infer


def test_yolo_uses_yolo_specific_env_config(monkeypatch):
    called = {}
    monkeypatch.setenv("YOLO26_CONF", "0.31")
    monkeypatch.setenv("YOLO26_IMG_SIZE", "832")
    monkeypatch.setenv("YOLO26_IOU", "0.47")
    monkeypatch.setenv("YOLO26_MAX_DETECTIONS", "91")
    monkeypatch.setenv("EDGECRAFTER_CONF", "0.99")
    monkeypatch.setenv("EDGECRAFTER_IMG_SIZE", "123")
    monkeypatch.setattr(yolo26_loader, "infer", _empty_yolo_infer(called))
    vision_backend._BACKEND_STATE.update(requested="yolo26", active="yolo26")

    vision_backend.run_inference(image_b64=_tiny_jpeg_b64(), payload={})

    assert called == {
        "conf": pytest.approx(0.31),
        "img_size": 832,
        "iou": pytest.approx(0.47),
        "max_det": 91,
        "class_filter": None,
    }
    assert vision_backend.get_last_detect_config()["backend"] == "yolo26"


def test_edgecrafter_uses_edgecrafter_specific_env_config(monkeypatch):
    import edgecrafter_loader as ec
    called = {}
    monkeypatch.setenv("EDGECRAFTER_CONF", "0.42")
    monkeypatch.setenv("EDGECRAFTER_IMG_SIZE", "704")
    monkeypatch.setenv("YOLO26_CONF", "0.91")
    monkeypatch.setenv("YOLO26_IMG_SIZE", "1280")
    monkeypatch.setenv("YOLO26_IOU", "0.19")
    monkeypatch.setenv("YOLO26_MAX_DETECTIONS", "999")

    def _infer(pil, conf, class_filter=None):
        called.update(conf=conf, class_filter=class_filter)
        return {"entities": [], "poses": [], "inference_ms": 1.0}

    monkeypatch.setattr(ec, "infer", _infer)
    monkeypatch.setattr(ec._STATE, "tasks", ["det"], raising=False)
    vision_backend._BACKEND_STATE.update(
        requested="edgecrafter", active="edgecrafter")

    vision_backend.run_inference(image_b64=_tiny_jpeg_b64(), payload={})

    assert called == {"conf": pytest.approx(0.42), "class_filter": None}
    effective = vision_backend.get_last_detect_config()
    assert effective["backend"] == "edgecrafter"
    assert effective["img_size"] == 704
    assert effective["iou"] is None
    assert effective["max_det"] is None


def test_payload_conf_and_img_size_override_yolo_env(monkeypatch):
    called = {}
    monkeypatch.setenv("YOLO26_CONF", "0.31")
    monkeypatch.setenv("YOLO26_IMG_SIZE", "832")
    monkeypatch.setattr(yolo26_loader, "infer", _empty_yolo_infer(called))
    vision_backend._BACKEND_STATE.update(requested="yolo26", active="yolo26")

    vision_backend.run_inference(
        image_b64=_tiny_jpeg_b64(),
        payload={"conf": 0.63, "img_size": 512},
    )

    assert called["conf"] == pytest.approx(0.63)
    assert called["img_size"] == 512
    effective = vision_backend.get_last_detect_config()
    assert effective["conf_source"] == "payload.conf"
    assert effective["img_size_source"] == "payload.img_size"


def test_fallback_recomputes_edgecrafter_config(monkeypatch):
    import edgecrafter_loader as ec
    resolved_backends = []
    called = {}
    real_resolve = vision_backend.resolve_effective_inference_config
    monkeypatch.setenv("YOLO26_CONF", "0.11")
    monkeypatch.setenv("YOLO26_IMG_SIZE", "1280")
    monkeypatch.setenv("EDGECRAFTER_CONF", "0.44")
    monkeypatch.setenv("EDGECRAFTER_IMG_SIZE", "608")

    def _track_resolve(backend, payload=None):
        resolved_backends.append(backend)
        return real_resolve(backend, payload)

    def _infer(pil, conf, class_filter=None):
        called["conf"] = conf
        return {"entities": [], "poses": [], "inference_ms": 1.0}

    monkeypatch.setattr(
        vision_backend, "resolve_effective_inference_config", _track_resolve)
    monkeypatch.setattr(ec, "infer", _infer)
    monkeypatch.setattr(ec._STATE, "tasks", ["det"], raising=False)
    vision_backend._BACKEND_STATE.update(
        requested="yolo26",
        active="edgecrafter",
        fallback_active=True,
        fallback_reason="simulated load failure",
    )

    vision_backend.run_inference(image_b64=_tiny_jpeg_b64(), payload={})

    assert resolved_backends == ["yolo26", "edgecrafter"]
    assert called["conf"] == pytest.approx(0.44)
    effective = vision_backend.get_last_detect_config()
    assert effective["backend"] == "edgecrafter"
    assert effective["img_size"] == 608


# -- 7-10. routes + Build/Plan unchanged ----------------------------------------

@pytest.fixture()
def server_mod(monkeypatch):
    import importlib
    monkeypatch.setenv("SKIP_WARMUP", "true")
    monkeypatch.setenv("AUTO_WARMUP", "false")
    if "server" in sys.modules:
        del sys.modules["server"]
    import importlib as _il
    return _il.import_module("server")


def test_routes_unchanged(server_mod):
    paths = {getattr(r, "path", None) for r in server_mod.app.routes}
    for p in ("/health", "/ping", "/debug/startup", "/debug/state", "/warmup",
              "/detect", "/build/session/start", "/build/session/lock",
              "/build/session/frame", "/build/session/finish",
              "/build/session/{session_id}/replay"):
        assert p in paths, p
    # No new backend-specific routes.
    assert not [p for p in paths if p and (p.startswith("/yolo") or p.startswith("/sam")
                                           or p.startswith("/plan"))]


def test_debug_state_includes_backend_status(server_mod):
    from fastapi.testclient import TestClient
    with TestClient(server_mod.app) as c:
        body = c.get("/debug/state").json()
        bs = body["backend_status"]
        for key in ("requested_backend", "active_backend", "fallback_backend",
                    "auto_backend_fallback", "fallback_active",
                    "yolo26_model_loaded", "edgecrafter_available"):
            assert key in bs, key
        for key in ("VISION_BACKEND", "FALLBACK_VISION_BACKEND",
                    "AUTO_BACKEND_FALLBACK", "YOLO26_MODEL_ID", "YOLO26_TASKS",
                    "YOLO26_DEVICE"):
            assert key in body["env_subset"], key
        assert "effective_config" in body
        assert body["last_detect_effective_config"] is None


def test_detect_contract_unchanged(server_mod, monkeypatch):
    from fastapi.testclient import TestClient
    fake = InferResponse(
        entities=[Entity(label="person", class_id=0, confidence=0.92,
                         bbox=BBox(x=0.1, y=0.2, w=0.3, h=0.4), source="yolo26")],
        poses=[Pose(label="person", confidence=0.88,
                    keypoints=[Keypoint(name="left_wrist", x=0.4, y=0.6, score=0.9)],
                    skeleton=[], source="yolo26-pose")],
        inference_ms=22, model="YOLO26", backend="yolo26",
        tasks=["det", "pose"], img_w=512, img_h=384,
    )
    monkeypatch.setattr(vision_backend, "run_inference", lambda **kw: fake)
    with server_mod._STATE_LOCK:
        server_mod._STATE["status"] = "ready"
    try:
        with TestClient(server_mod.app) as c:
            r = c.post("/detect", json={"image_b64": _tiny_jpeg_b64(), "conf": 0.25, "img_size": 640})
            assert r.status_code == 200
            body = r.json()
            assert body["backend"] == "yolo26" and body["model"] == "YOLO26"
            assert body["entities"][0]["bbox"] == {"x": 0.1, "y": 0.2, "w": 0.3, "h": 0.4}
            assert body["img_w"] == 512 and body["img_h"] == 384
    finally:
        with server_mod._STATE_LOCK:
            server_mod._STATE["status"] = "cold"


# -- Task-based modes (live det-only; lazy seg/pose) ----------------------------

def _fresh_yolo_state(monkeypatch):
    state = yolo26_loader._YoloState()
    monkeypatch.setattr(yolo26_loader, "_STATE", state)
    return state


def test_live_tasks_default_det_only(monkeypatch):
    monkeypatch.delenv("YOLO26_LIVE_TASKS", raising=False)
    monkeypatch.delenv("YOLO26_TASKS", raising=False)
    monkeypatch.delenv("YOLO26_POSE_ENABLED", raising=False)
    assert yolo26_loader.mode_tasks("live") == ["det"]
    assert yolo26_loader.mode_tasks("build") == ["det", "seg"]
    assert yolo26_loader.mode_tasks("plan") == ["det", "seg"]


def test_pose_only_when_enabled_or_explicit(monkeypatch):
    monkeypatch.delenv("YOLO26_LIVE_TASKS", raising=False)
    monkeypatch.delenv("YOLO26_TASKS", raising=False)
    monkeypatch.setenv("YOLO26_POSE_ENABLED", "true")
    assert yolo26_loader.mode_tasks("live") == ["det", "pose"]
    monkeypatch.setenv("YOLO26_POSE_ENABLED", "false")
    monkeypatch.setenv("YOLO26_LIVE_TASKS", "det,pose")  # explicit opt-in
    assert "pose" in yolo26_loader.mode_tasks("live")
    monkeypatch.setenv("YOLO26_LIVE_TASKS", "det")
    assert yolo26_loader.mode_tasks("live") == ["det"]


def test_warmup_loads_det_only_no_seg_no_pose(monkeypatch):
    state = _fresh_yolo_state(monkeypatch)
    monkeypatch.delenv("YOLO26_LIVE_TASKS", raising=False)
    monkeypatch.delenv("YOLO26_TASKS", raising=False)
    monkeypatch.delenv("YOLO26_POSE_ENABLED", raising=False)
    monkeypatch.setattr(yolo26_loader, "_load_one", lambda task: f"model-{task}")
    yolo26_loader.load()
    assert "det" in state.models           # 1/2/3: det loaded...
    assert "seg" not in state.models       # ...seg NOT loaded by default
    assert "pose" not in state.models      # ...pose NOT loaded unless enabled
    st = yolo26_loader.status()
    assert st["det_loaded"] and not st["seg_loaded"] and not st["pose_loaded"]
    assert st["live_tasks"] == ["det"]


def test_live_infer_runs_det_only(monkeypatch):
    from PIL import Image
    state = _fresh_yolo_state(monkeypatch)
    monkeypatch.delenv("YOLO26_LIVE_TASKS", raising=False)
    monkeypatch.delenv("YOLO26_TASKS", raising=False)
    monkeypatch.delenv("YOLO26_POSE_ENABLED", raising=False)
    state.models["det"] = object()
    state.loaded = True
    ran = []

    class _Res:
        boxes = None
        names = {}

    monkeypatch.setattr(yolo26_loader, "_predict",
                        lambda task, *a, **k: ran.append(task) or _Res())
    # ensure_task must NOT be invoked for seg/pose in live mode
    monkeypatch.setattr(yolo26_loader, "ensure_task",
                        lambda task: (_ for _ in ()).throw(AssertionError("lazy-loaded " + task)))
    out = yolo26_loader.infer(Image.new("RGB", (64, 64)), 0.25)
    assert ran == ["det"]
    assert out["tasks"] == ["det"]
    assert out["poses"] == [] and out["segments"] == []


def test_predict_passes_iou_and_max_det_to_ultralytics(monkeypatch):
    from PIL import Image
    state = _fresh_yolo_state(monkeypatch)
    called = {}
    result = object()

    class _Model:
        def __call__(self, image, **kwargs):
            called.update(kwargs)
            return [result]

    state.models["det"] = _Model()
    state.device = "cpu"

    actual = yolo26_loader._predict(
        "det", Image.new("RGB", (64, 64)), 0.28, 736, 0.46, 87)

    assert actual is result
    assert called == {
        "conf": pytest.approx(0.28),
        "imgsz": 736,
        "device": "cpu",
        "iou": pytest.approx(0.46),
        "max_det": 87,
        "verbose": False,
    }


def test_lazy_seg_failure_drops_task(monkeypatch):
    state = _fresh_yolo_state(monkeypatch)

    def _boom_load(task):
        raise RuntimeError("no seg weights")

    monkeypatch.setattr(yolo26_loader, "_load_one", _boom_load)
    assert yolo26_loader.ensure_task("seg") is None
    assert "seg" in state.failed
    # failure is cached -- not retried per frame
    monkeypatch.setattr(yolo26_loader, "_load_one",
                        lambda task: (_ for _ in ()).throw(AssertionError("retried")))
    assert yolo26_loader.ensure_task("seg") is None


# -- Build/Plan crop integration -------------------------------------------------

def _crop_b64():
    import io
    from PIL import Image, ImageDraw
    im = Image.new("RGB", (200, 150), (245, 245, 245))
    d = ImageDraw.Draw(im)
    d.rectangle([40, 30, 160, 120], fill=(20, 20, 20))
    buf = io.BytesIO(); im.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode()


_REGION = {"x": 0.1, "y": 0.2, "w": 0.4, "h": 0.3}

_YOLO_CROP = {
    "ok": True, "mask_source": "yolo26-seg", "confidence": 0.9,
    "mask_contour": [{"x": 0.22, "y": 0.31}, {"x": 0.6, "y": 0.3},
                     {"x": 0.62, "y": 0.72}, {"x": 0.24, "y": 0.74}],
    "parts": [{"label": "box", "class_id": 73, "confidence": 0.81,
               "bbox": {"x": 0.2, "y": 0.3, "w": 0.4, "h": 0.4}, "source": "yolo26"}],
}


def _run_frame(monkeypatch, crop_result, workflow="build", intent=None):
    import asyncio
    import build_blueprint as bb
    bb.BUILD_SESSIONS.clear()
    monkeypatch.setattr(bb, "_try_yolo26_crop",
                        lambda img, mode: crop_result)
    start = {"workflowMode": workflow}
    if intent:
        start["userIntent"] = intent
    sid = bb.start_session(start)["session_id"]
    payload = {"sessionId": sid, "frameId": "f-0", "selectedRegion": _REGION,
               "image_b64": _crop_b64(), "workflowMode": workflow,
               "handLandmarks": [{"role": "index-tip", "x": 0.3, "y": 0.35}],
               "gesture": {"type": "pinch", "active": True}}
    if intent:
        payload["userIntent"] = intent
    return asyncio.run(bb.process_frame_async(payload))["blueprint_frame"]


def test_build_crop_uses_yolo26_segmentation(monkeypatch):
    bf = _run_frame(monkeypatch, _YOLO_CROP, workflow="build")
    assert bf["maskSource"] == "yolo26-seg"
    assert bf["maskContour"] == _YOLO_CROP["mask_contour"]
    assert bf["outline"] == bf["maskContour"]
    for p in bf["maskContour"]:
        assert 0.0 <= p["x"] <= 1.0 and 0.0 <= p["y"] <= 1.0


def test_build_falls_back_when_yolo_seg_fails(monkeypatch):
    bf = _run_frame(monkeypatch, None, workflow="build")  # yolo produced nothing
    assert bf["maskSource"] in ("fallback-contour", "none")
    assert bf["version"] == 2  # frame still returned -- never 500


def test_plan_overlays_grounded_on_detected_parts(monkeypatch):
    intent = {"taskType": "inspect", "text": "inspect this part", "confirmed": True}
    bf = _run_frame(monkeypatch, _YOLO_CROP, workflow="plan", intent=intent)
    types = [o["type"] for o in bf["planOverlays"]]
    assert "highlight" in types and "callout" in types
    grounded = [o for o in bf["planOverlays"] if o.get("label") == "box"
                or "box" in str(o.get("label", ""))]
    assert grounded, bf["planOverlays"]
    allowed = {"arrow", "target", "ghost-position", "highlight", "warning-zone",
               "callout", "step-marker"}
    assert all(o["type"] in allowed for o in bf["planOverlays"])
    for o in bf["planOverlays"]:
        if "x" in o and o["x"] is not None:
            assert 0.0 <= o["x"] <= 1.0 and 0.0 <= o["y"] <= 1.0
        if "from" in o:
            assert 0.0 <= o["from"]["x"] <= 1.0 and 0.0 <= o["to"]["x"] <= 1.0


def test_fallback_fields_exposed(monkeypatch):
    import edgecrafter_loader as ec
    monkeypatch.setattr(yolo26_loader, "load", _boom)
    monkeypatch.setattr(ec, "load", _ec_ok)
    vision_backend.load_models()
    monkeypatch.setattr(ec, "infer", _fake_ec_infer)
    monkeypatch.setattr(ec._STATE, "tasks", ["det", "pose"], raising=False)
    body = vision_backend.run_inference(image_b64=_tiny_jpeg_b64(), conf=0.25).model_dump()
    assert body["fallbackUsed"] is True
    assert "yolo26_load_failed" in body["fallbackReason"]


def test_health_works_without_models(server_mod):
    from fastapi.testclient import TestClient
    with TestClient(server_mod.app) as c:
        r = c.get("/health")
        assert r.status_code == 200 and r.json()["ok"] is True


def test_build_and_plan_still_work(server_mod):
    import io
    from PIL import Image, ImageDraw
    from fastapi.testclient import TestClient
    im = Image.new("RGB", (200, 150), (245, 245, 245))
    d = ImageDraw.Draw(im)
    d.rectangle([40, 30, 160, 120], fill=(20, 20, 20))
    buf = io.BytesIO(); im.save(buf, format="JPEG")
    crop = base64.b64encode(buf.getvalue()).decode()
    region = {"x": 0.1, "y": 0.2, "w": 0.4, "h": 0.3}
    intent = {"taskType": "build", "text": "assemble", "confirmed": True}

    with TestClient(server_mod.app) as c:
        sid = c.post("/build/session/start", json={"workflowMode": "plan",
                                                   "userIntent": intent}).json()["session_id"]
        assert c.post("/build/session/lock",
                      json={"sessionId": sid, "selectedRegion": region}).json()["ok"]
        bf = c.post("/build/session/frame", json={
            "sessionId": sid, "frameId": "f-0", "selectedRegion": region,
            "image_b64": crop, "userIntent": intent,
            "handLandmarks": [{"role": "index-tip", "x": 0.3, "y": 0.35}],
            "gesture": {"type": "pinch", "active": True}}).json()["blueprint_frame"]
        assert bf["workflowMode"] == "plan"
        assert len(bf["planSteps"]) == 3            # 8. userIntent-driven steps
        assert bf["planOverlays"]                    # 9. overlays
        assert bf["maskSource"] in ("fallback-contour", "none")
        rep = c.get(f"/build/session/{sid}/replay").json()
        assert rep["ok"] and rep["frame_count"] == 1  # 7. replay works
