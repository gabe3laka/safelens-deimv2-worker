"""
tests/test_detector_config.py -- A1 detector config upgrade (generic YOLO_*,
legacy YOLO26_* fallback, ultralytics backend, /ws/vision active-backend config).

All CPU-only, no ultralytics / GPU / weights:

  40. generic YOLO_* env resolves correctly
  41. legacy YOLO26_* maps when generic YOLO_* is absent (and generic wins when both set)
  42. /ws/vision resolves the ACTIVE backend config (not stale EdgeCrafter defaults)
  43. stronger YOLO demo config (yolo11s.pt / 960 / 0.10) is visible in /debug/state summary
  +   VISION_BACKEND=ultralytics routes through the YOLO adapter path
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

import pytest

import config_resolver


_YOLO_ENV = ("YOLO_DET_MODEL_ID", "YOLO_CONF", "YOLO_IMG_SIZE", "YOLO_IOU",
             "YOLO_MAX_DETECTIONS")
_YOLO26_ENV = ("YOLO26_DET_MODEL_ID", "YOLO26_MODEL_ID", "YOLO26_CONF",
               "YOLO26_IMG_SIZE", "YOLO26_IOU", "YOLO26_MAX_DETECTIONS")


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for k in (*_YOLO_ENV, *_YOLO26_ENV, "VISION_BACKEND"):
        monkeypatch.delenv(k, raising=False)
    yield


# -- 40. generic YOLO_* resolves ----------------------------------------------

def test_generic_yolo_env_resolves(monkeypatch):
    monkeypatch.setenv("YOLO_CONF", "0.10")
    monkeypatch.setenv("YOLO_IMG_SIZE", "960")
    monkeypatch.setenv("YOLO_IOU", "0.60")
    monkeypatch.setenv("YOLO_MAX_DETECTIONS", "300")
    cfg = config_resolver.resolve_effective_inference_config("ultralytics", {})
    assert cfg.conf == pytest.approx(0.10)
    assert cfg.img_size == 960
    assert cfg.iou == pytest.approx(0.60)
    assert cfg.max_det == 300
    assert "YOLO_CONF" in cfg.conf_source
    # ultralytics shares the YOLO resolution path with yolo26
    assert config_resolver.resolve_effective_inference_config("yolo26", {}).img_size == 960


def test_generic_yolo_model_id(monkeypatch):
    monkeypatch.setenv("YOLO_DET_MODEL_ID", "yolo11s.pt")
    assert config_resolver.resolve_detector_model_id() == "yolo11s.pt"


# -- 41. legacy YOLO26_* fallback + generic precedence ------------------------

def test_legacy_yolo26_used_when_generic_absent(monkeypatch):
    monkeypatch.setenv("YOLO26_CONF", "0.25")
    monkeypatch.setenv("YOLO26_IMG_SIZE", "640")
    monkeypatch.setenv("YOLO26_DET_MODEL_ID", "yolo26n.pt")
    cfg = config_resolver.resolve_effective_inference_config("yolo26", {})
    assert cfg.conf == pytest.approx(0.25)
    assert cfg.img_size == 640
    assert "YOLO26_CONF" in cfg.conf_source
    assert config_resolver.resolve_detector_model_id() == "yolo26n.pt"


def test_generic_overrides_legacy_when_both_set(monkeypatch):
    monkeypatch.setenv("YOLO26_CONF", "0.25")
    monkeypatch.setenv("YOLO_CONF", "0.10")
    monkeypatch.setenv("YOLO26_DET_MODEL_ID", "yolo26n.pt")
    monkeypatch.setenv("YOLO_DET_MODEL_ID", "yolo11s.pt")
    cfg = config_resolver.resolve_effective_inference_config("yolo26", {})
    assert cfg.conf == pytest.approx(0.10)  # generic wins
    assert config_resolver.resolve_detector_model_id() == "yolo11s.pt"


def test_invalid_env_falls_back_to_default(monkeypatch):
    monkeypatch.setenv("YOLO_CONF", "not-a-number")
    cfg = config_resolver.resolve_effective_inference_config("yolo26", {})
    assert cfg.conf == pytest.approx(0.20)  # safe default, never raises


# -- 42. /ws/vision uses the active backend config ----------------------------

def test_ws_vision_resolves_active_backend_config(monkeypatch):
    import ws_vision
    monkeypatch.setenv("VISION_BACKEND", "ultralytics")
    monkeypatch.setenv("YOLO_IMG_SIZE", "960")
    monkeypatch.setenv("YOLO_CONF", "0.10")
    cfg = ws_vision._resolve_active_stream_config()
    assert cfg["img_size"] == 960            # NOT the old 640 EdgeCrafter default
    assert cfg["conf"] == pytest.approx(0.10)

    monkeypatch.setenv("VISION_BACKEND", "edgecrafter")
    monkeypatch.setenv("EDGECRAFTER_IMG_SIZE", "608")
    monkeypatch.setenv("EDGECRAFTER_CONF", "0.30")
    ec = ws_vision._resolve_active_stream_config()
    assert ec["img_size"] == 608
    assert ec["conf"] == pytest.approx(0.30)


# -- 43. stronger demo detector config visible in /debug/state summary --------

def test_debug_state_summary_shows_active_detector(monkeypatch):
    monkeypatch.setenv("VISION_BACKEND", "ultralytics")
    monkeypatch.setenv("YOLO_DET_MODEL_ID", "yolo11s.pt")
    monkeypatch.setenv("YOLO_IMG_SIZE", "960")
    monkeypatch.setenv("YOLO_CONF", "0.10")
    summary = config_resolver.get_effective_config_summary()
    det = summary["active_detector"]
    assert det["active_backend"] == "ultralytics"
    assert det["active_model_id"] == "yolo11s.pt"
    assert det["img_size"] == 960
    assert det["conf"] == pytest.approx(0.10)
    assert det["weights_source"]  # descriptive, runtime-resolved (never baked)


# -- ultralytics routes through the YOLO adapter path -------------------------

def test_ultralytics_backend_routes_to_yolo_adapter(monkeypatch):
    pytest.importorskip("PIL")
    import vision_backend
    import yolo26_loader

    monkeypatch.setenv("VISION_BACKEND", "ultralytics")
    monkeypatch.setenv("AUTO_BACKEND_FALLBACK", "false")
    vision_backend._BACKEND_STATE.update(
        requested=None, active=None, fallback_active=False, fallback_reason=None)

    called = {}

    def _fake_load(*a, **k):
        called["load"] = True
        return {"backend": "yolo26", "tasks_loaded": ["det"], "model_classes": {},
                "model_ids": {}, "device": "cpu", "warnings": []}

    monkeypatch.setattr(yolo26_loader, "load", _fake_load)
    summary = vision_backend.load_models()
    assert called["load"] is True
    assert summary["active_backend"] == "ultralytics"
    assert vision_backend.serving_backend() == "ultralytics"

    vision_backend._BACKEND_STATE.update(
        requested=None, active=None, fallback_active=False, fallback_reason=None)
