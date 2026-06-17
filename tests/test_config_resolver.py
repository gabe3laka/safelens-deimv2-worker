"""
tests/test_config_resolver.py -- effective per-backend inference config wiring.

Proves the config_resolver fix the inference path was missing:

  1. YOLO26 uses YOLO26_* env when the payload does not override.
  2. EdgeCrafter uses EDGECRAFTER_* env (its own values, not YOLO26's).
  3. payload conf/img_size override env (HSE profiles tune per request).
  4. an auto-fallback to EdgeCrafter recomputes EdgeCrafter's config
     separately (serving_backend() drives the resolve, never the YOLO values).
  5. invalid env values fall back safely (never raise).

Plus the loader/route wiring:
  6. yolo26_loader._predict() passes iou + max_det into the Ultralytics call.
  7. GET /debug/state exposes effective_config (actual backend/conf/.../max_det).

All CPU-only, no ultralytics / GPU / weights needed.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

import pytest

import config_resolver

# Distinct, non-default env values so a "wrong backend" leak is obvious.
_YOLO_ENV = {
    "YOLO26_CONF": "0.42",
    "YOLO26_IMG_SIZE": "512",
    "YOLO26_IOU": "0.55",
    "YOLO26_MAX_DETECTIONS": "77",
}
_EC_ENV = {
    "EDGECRAFTER_CONF": "0.11",
    "EDGECRAFTER_IMG_SIZE": "768",
}


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    # Start from a known-empty slate so defaults are predictable.
    for k in (*_YOLO_ENV, *_EC_ENV, "EDGECRAFTER_IOU", "EDGECRAFTER_MAX_DETECTIONS"):
        monkeypatch.delenv(k, raising=False)
    yield


# -- 1. YOLO uses YOLO26_* env -------------------------------------------------

def test_yolo26_uses_yolo_env(monkeypatch):
    for k, v in _YOLO_ENV.items():
        monkeypatch.setenv(k, v)
    # EdgeCrafter env present but must NOT leak into the yolo26 result.
    for k, v in _EC_ENV.items():
        monkeypatch.setenv(k, v)

    cfg = config_resolver.resolve_effective_inference_config("yolo26", {})
    assert cfg["backend"] == "yolo26"
    assert cfg["conf"] == pytest.approx(0.42)
    assert cfg["img_size"] == 512
    assert cfg["iou"] == pytest.approx(0.55)
    assert cfg["max_det"] == 77


def test_yolo26_defaults_when_env_unset():
    cfg = config_resolver.resolve_effective_inference_config("yolo26", {})
    assert cfg == {"backend": "yolo26", "conf": 0.25, "img_size": 640,
                   "iou": 0.45, "max_det": 300}


def test_unknown_backend_defaults_to_yolo26():
    cfg = config_resolver.resolve_effective_inference_config(None, {})
    assert cfg["backend"] == "yolo26"
    assert config_resolver.resolve_effective_inference_config("mystery", {})["backend"] == "yolo26"


# -- 2. EdgeCrafter uses EDGECRAFTER_* env -------------------------------------

def test_edgecrafter_uses_its_own_env(monkeypatch):
    for k, v in {**_YOLO_ENV, **_EC_ENV}.items():
        monkeypatch.setenv(k, v)
    cfg = config_resolver.resolve_effective_inference_config("edgecrafter", {})
    assert cfg["backend"] == "edgecrafter"
    assert cfg["conf"] == pytest.approx(0.11)      # EDGECRAFTER_CONF
    assert cfg["img_size"] == 768                  # EDGECRAFTER_IMG_SIZE
    # NOT the YOLO26 values:
    assert cfg["conf"] != pytest.approx(0.42)
    assert cfg["img_size"] != 512


# -- 3. payload overrides env (HSE profiles) -----------------------------------

def test_payload_overrides_env(monkeypatch):
    for k, v in _YOLO_ENV.items():
        monkeypatch.setenv(k, v)
    cfg = config_resolver.resolve_effective_inference_config(
        "yolo26", {"conf": 0.7, "img_size": 1024})
    assert cfg["conf"] == pytest.approx(0.7)       # payload wins over env
    assert cfg["img_size"] == 1024
    assert cfg["iou"] == pytest.approx(0.55)       # untouched -> env value


def test_payload_alias_keys(monkeypatch):
    cfg = config_resolver.resolve_effective_inference_config(
        "yolo26", {"confidence": 0.6, "imgsz": 320, "max_detections": 10})
    assert cfg["conf"] == pytest.approx(0.6)
    assert cfg["img_size"] == 320
    assert cfg["max_det"] == 10


def test_payload_out_of_range_is_clamped(monkeypatch):
    cfg = config_resolver.resolve_effective_inference_config(
        "yolo26", {"conf": 5.0, "img_size": 2})
    assert cfg["conf"] == pytest.approx(1.0)       # clamped to <= 1.0
    assert cfg["img_size"] == 32                   # clamped to >= 32


# -- 4. fallback recomputes EdgeCrafter config separately ----------------------

def test_fallback_serving_backend_drives_edgecrafter_config(monkeypatch):
    """Mirror /detect: resolve for serving_backend() after a yolo26->ec fallback."""
    import vision_backend
    for k, v in {**_YOLO_ENV, **_EC_ENV}.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setenv("VISION_BACKEND", "yolo26")
    # Simulate the post-fallback state set by load_models().
    monkeypatch.setitem(vision_backend._BACKEND_STATE, "active", "edgecrafter")
    monkeypatch.setitem(vision_backend._BACKEND_STATE, "fallback_active", True)

    assert vision_backend.serving_backend() == "edgecrafter"
    cfg = config_resolver.resolve_effective_inference_config(
        vision_backend.serving_backend(), {})
    assert cfg["backend"] == "edgecrafter"
    assert cfg["conf"] == pytest.approx(0.11)      # EDGECRAFTER_CONF, not YOLO26
    assert cfg["img_size"] == 768                  # EDGECRAFTER_IMG_SIZE, not 512


# -- 5. invalid env values fall back safely ------------------------------------

def test_invalid_env_falls_back_to_defaults(monkeypatch):
    monkeypatch.setenv("YOLO26_CONF", "not-a-number")
    monkeypatch.setenv("YOLO26_IMG_SIZE", "")
    monkeypatch.setenv("YOLO26_IOU", "NaNish")
    monkeypatch.setenv("YOLO26_MAX_DETECTIONS", "abc")
    cfg = config_resolver.resolve_effective_inference_config("yolo26", {})
    assert cfg == {"backend": "yolo26", "conf": 0.25, "img_size": 640,
                   "iou": 0.45, "max_det": 300}


def test_invalid_payload_falls_through_to_env(monkeypatch):
    monkeypatch.setenv("YOLO26_CONF", "0.33")
    cfg = config_resolver.resolve_effective_inference_config(
        "yolo26", {"conf": "garbage"})
    assert cfg["conf"] == pytest.approx(0.33)      # bad payload -> env value


def test_never_raises_on_weird_payload():
    for bad in (None, [], "string", 42, {"conf": object()}):
        cfg = config_resolver.resolve_effective_inference_config("yolo26", bad)
        assert cfg["backend"] == "yolo26"          # always a valid dict


# -- 6. _predict passes iou + max_det into the Ultralytics call ----------------

def test_predict_passes_iou_and_max_det(monkeypatch):
    pytest.importorskip("PIL")
    from PIL import Image
    import yolo26_loader

    state = yolo26_loader._YoloState()
    state.device = "cpu"
    captured = {}

    class _FakeModel:
        def __call__(self, img, **kwargs):
            captured.update(kwargs)
            return [object()]  # results[0]

    state.models["det"] = _FakeModel()
    monkeypatch.setattr(yolo26_loader, "_STATE", state)

    yolo26_loader._predict("det", Image.new("RGB", (32, 32)), 0.3, 512,
                           iou=0.6, max_det=50)
    assert captured["conf"] == pytest.approx(0.3)
    assert captured["imgsz"] == 512
    assert captured["iou"] == pytest.approx(0.6)
    assert captured["max_det"] == 50
    assert captured["device"] == "cpu"
    assert captured["verbose"] is False


def test_predict_iou_max_det_default_from_env(monkeypatch):
    pytest.importorskip("PIL")
    from PIL import Image
    import yolo26_loader

    monkeypatch.setenv("YOLO26_IOU", "0.7")
    monkeypatch.setenv("YOLO26_MAX_DETECTIONS", "12")
    state = yolo26_loader._YoloState()
    state.device = "cpu"
    captured = {}

    class _FakeModel:
        def __call__(self, img, **kwargs):
            captured.update(kwargs)
            return [object()]

    state.models["det"] = _FakeModel()
    monkeypatch.setattr(yolo26_loader, "_STATE", state)

    yolo26_loader._predict("det", Image.new("RGB", (16, 16)), 0.25, 640)
    assert captured["iou"] == pytest.approx(0.7)   # from YOLO26_IOU
    assert captured["max_det"] == 12               # from YOLO26_MAX_DETECTIONS


# -- 7. /debug/state exposes effective_config ----------------------------------

def test_debug_state_exposes_effective_config(monkeypatch):
    pytest.importorskip("fastapi")
    import importlib
    monkeypatch.setenv("SKIP_WARMUP", "true")
    monkeypatch.setenv("AUTO_WARMUP", "false")
    monkeypatch.setenv("VISION_BACKEND", "yolo26")
    monkeypatch.setenv("YOLO26_IOU", "0.5")
    monkeypatch.setenv("YOLO26_MAX_DETECTIONS", "99")
    if "server" in sys.modules:
        del sys.modules["server"]
    server_mod = importlib.import_module("server")

    from fastapi.testclient import TestClient
    with TestClient(server_mod.app) as c:
        ec = c.get("/debug/state").json()["effective_config"]
    for key in ("backend", "conf", "img_size", "iou", "max_det"):
        assert key in ec, key
    assert ec["backend"] == "yolo26"
    assert ec["iou"] == pytest.approx(0.5)
    assert ec["max_det"] == 99
