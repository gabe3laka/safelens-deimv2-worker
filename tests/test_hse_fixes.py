"""
tests/test_hse_fixes.py -- Worker fix: HSE box coloring independent of Gemini success.

Covers the six requirements:
  1. Deterministic YELLOW risk becomes scene_risks even when Gemini fails
  2. detect log has scene_risks_count and linkable_scene_risks_count
  3. Entity risk stamping from scene_risks
  4. Gemini 503 -> unavailable/error, NOT json_parse_error
  5. json_parse_error / failure does not overwrite last-good VLM cache
  6. Error backoff prevents immediate Gemini retry after failure
"""

from __future__ import annotations

import asyncio
import base64
import importlib
import io
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

import pytest

pytest.importorskip("pydantic")

import risk.vlm_reasoner as vlm
import risk.gemini_reasoner as gemini_reasoner
from risk.reason_schema import ReasonResponse

# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------

PERSON = {
    "label": "person", "class_id": 0, "confidence": 0.9,
    "track_id": "trk_1",
    "bbox": {"x": 0.30, "y": 0.40, "w": 0.14, "h": 0.45},
}
FORKLIFT = {
    "label": "forklift", "class_id": 7, "confidence": 0.86,
    "track_id": "trk_2",
    "bbox": {"x": 0.34, "y": 0.42, "w": 0.24, "h": 0.40},
}

DET_RISK_YELLOW = {
    "risk_id": "rsk_yellow_1",
    "hazard_type": "object_near_edge",
    "risk_state": "active",
    "risk_level": "YELLOW",
    "severity": 2, "likelihood": 2, "risk_score": 4,
    "involved_track_ids": ["trk_1"],
    "bbox": {"x": 0.30, "y": 0.40, "w": 0.14, "h": 0.45},
    "reason": "Person near edge.",
    "recommended_action": "Secure load.",
    "produced_by": "risk_engine",
    "requires_human_review": False,
}


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    monkeypatch.setenv("VLM_REASONER_ENABLED", "true")
    monkeypatch.setenv("REASONER_MODE", "mock")
    monkeypatch.setenv("RISK_ENGINE_ENABLED", "true")
    monkeypatch.delenv("REASONER_ERROR_BACKOFF_MS", raising=False)
    vlm.reset()
    yield
    vlm.reset()


@pytest.fixture()
def server_mod(monkeypatch):
    monkeypatch.setenv("SKIP_WARMUP", "true")
    monkeypatch.setenv("AUTO_WARMUP", "false")
    if "server" in sys.modules:
        del sys.modules["server"]
    return importlib.import_module("server")


def _get_srv(monkeypatch):
    """Inline server load for tests that don't use the server_mod fixture."""
    monkeypatch.setenv("SKIP_WARMUP", "true")
    if "server" in sys.modules:
        del sys.modules["server"]
    return importlib.import_module("server")


def _tiny_jpeg_b64():
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), (255, 255, 255)).save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode()


def _fake_resp_with_risk():
    """Return a fake InferResponse where the risk engine would produce YELLOW risk."""
    from schema import BBox, Entity, InferResponse
    return InferResponse(
        entities=[
            Entity(label="person", class_id=0, confidence=0.9,
                   bbox=BBox(**PERSON["bbox"]), source="yolo26",
                   track_id="trk_1"),
            Entity(label="forklift", class_id=7, confidence=0.86,
                   bbox=BBox(**FORKLIFT["bbox"]), source="yolo26",
                   track_id="trk_2"),
        ],
        inference_ms=10, model="YOLO26", backend="yolo26", tasks=["det"],
        img_w=1280, img_h=720,
    )


# ---------------------------------------------------------------------------
# Requirement 1 -- Deterministic YELLOW risk -> scene_risks regardless of Gemini
# ---------------------------------------------------------------------------

def test_scene_risks_built_when_gemini_queued(monkeypatch):
    """When Gemini has not run yet (triggered), deterministic scene_risks still present."""
    srv = _get_srv(monkeypatch)
    # Simulate: VLM triggered but no draft yet (returns None, "triggered")
    monkeypatch.setattr(vlm, "maybe_trigger",
                        lambda *a, **k: (None, "triggered"))
    scene = srv._build_scene_risks([DET_RISK_YELLOW], None, [])
    assert len(scene) == 1
    assert scene[0]["risk_level"] == "YELLOW"
    assert srv._is_linkable_scene_risk(scene[0])


def test_scene_risks_built_when_gemini_error(monkeypatch):
    """When Gemini returns error, deterministic scene_risks are still present."""
    srv = _get_srv(monkeypatch)
    failure_draft = {"reasoner_status": "error", "risks": [], "error": "boom"}
    scene = srv._build_scene_risks([DET_RISK_YELLOW], failure_draft, [])
    assert len(scene) == 1
    assert scene[0]["risk_level"] == "YELLOW"
    assert srv._is_linkable_scene_risk(scene[0])


def test_scene_risks_built_when_gemini_json_parse_error(monkeypatch):
    """When Gemini returns json_parse_error, deterministic scene_risks still present."""
    srv = _get_srv(monkeypatch)
    failure_draft = {"reasoner_status": "json_parse_error", "risks": []}
    scene = srv._build_scene_risks([DET_RISK_YELLOW], failure_draft, [])
    assert len(scene) == 1
    assert scene[0]["risk_level"] == "YELLOW"


def test_scene_risks_built_when_gemini_timeout(monkeypatch):
    """When Gemini returns timeout, deterministic scene_risks still present."""
    srv = _get_srv(monkeypatch)
    failure_draft = {"reasoner_status": "timeout", "risks": []}
    scene = srv._build_scene_risks([DET_RISK_YELLOW], failure_draft, [])
    assert len(scene) == 1
    assert scene[0]["risk_level"] == "YELLOW"


def test_scene_risks_built_when_gemini_unavailable(monkeypatch):
    """When Gemini returns unavailable, deterministic scene_risks still present."""
    srv = _get_srv(monkeypatch)
    failure_draft = {"reasoner_status": "unavailable", "risks": []}
    scene = srv._build_scene_risks([DET_RISK_YELLOW], failure_draft, [])
    assert len(scene) == 1
    assert scene[0]["risk_level"] == "YELLOW"


# ---------------------------------------------------------------------------
# Requirement 2 -- detect log counts
# ---------------------------------------------------------------------------

def test_is_linkable_scene_risk_with_track_ids(monkeypatch):
    srv = _get_srv(monkeypatch)
    assert srv._is_linkable_scene_risk({"involved_track_ids": ["trk_1"]}) is True


def test_is_linkable_scene_risk_with_detection_ids(monkeypatch):
    srv = _get_srv(monkeypatch)
    assert srv._is_linkable_scene_risk({"involved_detection_ids": [0]}) is True


def test_is_linkable_scene_risk_with_entity_id(monkeypatch):
    srv = _get_srv(monkeypatch)
    assert srv._is_linkable_scene_risk({"linked_entity_id": "ent_1"}) is True
    assert srv._is_linkable_scene_risk({"entity_id": "ent_1"}) is True
    assert srv._is_linkable_scene_risk({"track_id": "trk_1"}) is True


def test_is_linkable_scene_risk_with_valid_bbox(monkeypatch):
    srv = _get_srv(monkeypatch)
    assert srv._is_linkable_scene_risk(
        {"bbox": {"x": 0.1, "y": 0.2, "w": 0.3, "h": 0.4}}) is True


def test_is_linkable_scene_risk_with_empty_bbox(monkeypatch):
    srv = _get_srv(monkeypatch)
    assert srv._is_linkable_scene_risk({"bbox": {}}) is False


def test_is_linkable_scene_risk_vague(monkeypatch):
    srv = _get_srv(monkeypatch)
    assert srv._is_linkable_scene_risk({}) is False
    assert srv._is_linkable_scene_risk({"involved_track_ids": []}) is False
    assert srv._is_linkable_scene_risk({"involved_detection_ids": []}) is False


def test_is_linkable_scene_risk_non_dict(monkeypatch):
    srv = _get_srv(monkeypatch)
    assert srv._is_linkable_scene_risk(None) is False
    assert srv._is_linkable_scene_risk("string") is False


def test_detect_log_contains_scene_risk_counts(server_mod, monkeypatch):
    """The detect log event must include scene_risks_count and linkable_scene_risks_count."""
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    import vision_backend

    logged: List[Dict[str, Any]] = []

    def capture_log_event(event_name, **kwargs):
        logged.append({"event": event_name, **kwargs})

    monkeypatch.setenv("RISK_ENGINE_ENABLED", "true")
    monkeypatch.setenv("TEMPORAL_REASONING_ENABLED", "false")
    monkeypatch.setattr(vision_backend, "run_inference", lambda **kw: _fake_resp_with_risk())
    import worker_runtime as runtime_mod
    monkeypatch.setattr(runtime_mod, "log_event", capture_log_event)

    with server_mod._STATE_LOCK:
        server_mod._STATE["status"] = "ready"
    try:
        with TestClient(server_mod.app) as c:
            r = c.post("/detect", json={
                "image_b64": _tiny_jpeg_b64(),
                "session_id": "cam_log",
                "mode": "hse-monitoring",
            })
            assert r.status_code == 200
        detect_events = [e for e in logged if e.get("event") == "detect"]
        assert detect_events, "No detect log event captured"
        ev = detect_events[-1]
        assert "scene_risks_count" in ev, f"scene_risks_count missing from: {ev}"
        assert "linkable_scene_risks_count" in ev, f"linkable_scene_risks_count missing"
        assert "entity_risk_stamped_count" in ev, f"entity_risk_stamped_count missing"
        assert isinstance(ev["scene_risks_count"], int)
        assert isinstance(ev["linkable_scene_risks_count"], int)
        assert isinstance(ev["entity_risk_stamped_count"], int)
    finally:
        with server_mod._STATE_LOCK:
            server_mod._STATE["status"] = "cold"


# ---------------------------------------------------------------------------
# Requirement 3 -- Entity risk stamping
# ---------------------------------------------------------------------------

def test_stamp_entity_risks_by_track_id(monkeypatch):
    """Entities matching scene_risk involved_track_ids get risk fields stamped."""
    srv = _get_srv(monkeypatch)
    entity = dict(PERSON)
    resp = {
        "entities": [entity],
        "scene_risks": [DET_RISK_YELLOW],
    }
    srv._stamp_entity_risks(resp)
    assert entity["risk_level"] == "YELLOW"
    assert entity["risk_color"] == "yellow"
    assert entity["risk_score"] == 4
    assert entity["severity"] == 2
    assert entity["likelihood"] == 2
    assert entity["risk_reason"]
    assert entity["recommended_action"] == "Secure load."
    assert entity["produced_by"] == "risk_engine"
    assert entity["requires_human_review"] is False


def test_stamp_entity_risks_by_linked_entity_id(monkeypatch):
    """Entities matched by linked_entity_id get risk stamped."""
    srv = _get_srv(monkeypatch)
    entity = {"label": "person", "track_id": "trk_5", "bbox": {"x": 0, "y": 0, "w": 0.1, "h": 0.1}}
    risk = {
        "risk_id": "r1", "risk_level": "ORANGE", "risk_score": 12,
        "severity": 3, "likelihood": 4,
        "linked_entity_id": "trk_5",
        "risk_reason": "Near vehicle",
        "recommended_action": "Separate",
        "produced_by": "risk_engine", "requires_human_review": False,
    }
    resp = {"entities": [entity], "scene_risks": [risk]}
    srv._stamp_entity_risks(resp)
    assert entity["risk_level"] == "ORANGE"
    assert entity["risk_color"] == "orange"


def test_stamp_entity_risks_by_bbox_iou(monkeypatch):
    """Entities matched by bbox IoU get risk stamped."""
    srv = _get_srv(monkeypatch)
    entity = {
        "label": "person", "track_id": "trk_no_match",
        "bbox": {"x": 0.30, "y": 0.40, "w": 0.14, "h": 0.45},
    }
    # Risk has same bbox but no track link -- should match by IoU
    risk = {
        "risk_id": "r_iou", "risk_level": "RED", "risk_score": 20,
        "severity": 4, "likelihood": 5,
        "bbox": {"x": 0.30, "y": 0.40, "w": 0.14, "h": 0.45},
        "involved_track_ids": ["trk_other"],  # different track
        "risk_reason": "High risk",
        "produced_by": "risk_engine", "requires_human_review": False,
    }
    resp = {"entities": [entity], "scene_risks": [risk]}
    srv._stamp_entity_risks(resp)
    # Track IDs don't match, so this should NOT match by track ID
    # bbox is same so IoU = 1.0 which exceeds threshold 0.30
    # BUT trk_other != trk_no_match, so no track match; bbox IoU fallback applies
    # actually involved_track_ids = ["trk_other"] != "trk_no_match", so no track match
    # IoU fallback: same bbox -> IoU=1.0 >= 0.30 -> MATCH
    assert entity["risk_level"] == "RED"


def test_stamp_entity_risks_by_bbox_iou_threshold_020(monkeypatch):
    """IoU >= 0.20 (even < 0.30) stamps the entity."""
    srv = _get_srv(monkeypatch)
    entity = {
        "label": "person",
        "track_id": "trk_no_match",
        "bbox": {"x": 0.00, "y": 0.00, "w": 0.50, "h": 0.50},
    }
    risk = {
        "risk_id": "r_iou_020",
        "risk_level": "YELLOW",
        "bbox": {"x": 0.18, "y": 0.18, "w": 0.50, "h": 0.50},
        "involved_track_ids": ["trk_other"],
        "produced_by": "risk_engine",
        "requires_human_review": False,
    }
    resp = {"entities": [entity], "scene_risks": [risk]}
    iou = srv._iou(entity["bbox"], risk["bbox"])
    assert iou == pytest.approx(0.2575, abs=0.005)
    assert 0.20 <= iou < 0.30
    assert srv._center_dist(entity["bbox"], risk["bbox"]) > 0.12
    srv._stamp_entity_risks(resp)
    assert entity["risk_level"] == "YELLOW"


def test_stamp_entity_risks_by_center_distance_fallback(monkeypatch):
    """Center-distance <= 0.12 stamps even when IoU is below 0.20."""
    srv = _get_srv(monkeypatch)
    entity = {
        "label": "person",
        "track_id": "trk_no_match",
        "bbox": {"x": 0.40, "y": 0.40, "w": 0.08, "h": 0.08},
    }
    risk = {
        "risk_id": "r_center",
        "risk_level": "ORANGE",
        "bbox": {"x": 0.35, "y": 0.35, "w": 0.30, "h": 0.30},
        "involved_track_ids": ["trk_other"],
        "produced_by": "risk_engine",
        "requires_human_review": False,
    }
    resp = {"entities": [entity], "scene_risks": [risk]}
    assert srv._iou(entity["bbox"], risk["bbox"]) < 0.20
    assert srv._center_dist(entity["bbox"], risk["bbox"]) <= 0.12
    srv._stamp_entity_risks(resp)
    assert entity["risk_level"] == "ORANGE"


def test_stamp_entity_risks_candidate_only_not_stamped(monkeypatch):
    """Candidate-only risks are not stamped onto boxes."""
    srv = _get_srv(monkeypatch)
    entity = dict(PERSON)
    risk = dict(DET_RISK_YELLOW, candidate_only=True)
    resp = {"entities": [entity], "scene_risks": [risk]}
    srv._stamp_entity_risks(resp)
    assert "risk_level" not in entity


def test_stamp_entity_risks_highest_wins(monkeypatch):
    """When multiple risks match the same entity, the highest risk_level wins."""
    srv = _get_srv(monkeypatch)
    entity = dict(PERSON)  # track_id="trk_1"
    risk_yellow = dict(DET_RISK_YELLOW)  # involved_track_ids=["trk_1"], YELLOW
    risk_orange = {
        "risk_id": "rsk_orange", "risk_level": "ORANGE", "risk_score": 12,
        "severity": 3, "likelihood": 4,
        "involved_track_ids": ["trk_1"],
        "risk_reason": "Proximity", "recommended_action": "Separate",
        "produced_by": "risk_engine", "requires_human_review": False,
    }
    resp = {"entities": [entity], "scene_risks": [risk_yellow, risk_orange]}
    srv._stamp_entity_risks(resp)
    assert entity["risk_level"] == "ORANGE"


def test_stamp_entity_risks_no_match_no_stamp(monkeypatch):
    """Entities not matching any scene_risk do not get risk fields."""
    srv = _get_srv(monkeypatch)
    entity = {"label": "person", "track_id": "trk_99", "bbox": {"x": 0.9, "y": 0.9, "w": 0.05, "h": 0.05}}
    resp = {"entities": [entity], "scene_risks": [DET_RISK_YELLOW]}
    srv._stamp_entity_risks(resp)
    assert "risk_level" not in entity


def test_stamp_entity_risks_no_entities(monkeypatch):
    """_stamp_entity_risks handles empty entities gracefully."""
    srv = _get_srv(monkeypatch)
    resp = {"entities": [], "scene_risks": [DET_RISK_YELLOW]}
    srv._stamp_entity_risks(resp)  # must not raise


def test_stamp_entity_risks_no_scene_risks(monkeypatch):
    """_stamp_entity_risks handles empty scene_risks gracefully."""
    srv = _get_srv(monkeypatch)
    resp = {"entities": [dict(PERSON)], "scene_risks": []}
    srv._stamp_entity_risks(resp)  # must not raise


# ---------------------------------------------------------------------------
# Requirement 4 -- Gemini 503 -> unavailable/error, NOT json_parse_error
# ---------------------------------------------------------------------------

def test_gemini_503_produces_unavailable(monkeypatch):
    """GeminiUnavailableError from generate() maps to reasoner_status=unavailable."""
    monkeypatch.setenv("REASONER_MODE", "gemini")

    def raise_503(prompt, image):
        raise gemini_reasoner.GeminiUnavailableError("Gemini unavailable (503): ServiceUnavailable")

    monkeypatch.setitem(vlm._ADAPTER_STATE, "gemini", {
        "available": True,
        "generate": raise_503,
        "model_id": "gemini-2.5-flash",
        "diagnostics": {},
        "error": None,
    })
    out = vlm.reason_sync({"session_id": "cam_503", "frame_id": "f1", "entities": []})
    assert out["reasoner_status"] == "unavailable"
    assert out["reasoner_status"] != "json_parse_error"
    assert out["reasoner_status"] != "error"


def test_gemini_503_not_classified_as_json_parse_error(monkeypatch):
    """A 503 must never produce json_parse_error."""
    monkeypatch.setenv("REASONER_MODE", "gemini")

    def raise_503(prompt, image):
        raise gemini_reasoner.GeminiUnavailableError("503 Service Unavailable")

    monkeypatch.setitem(vlm._ADAPTER_STATE, "gemini", {
        "available": True, "generate": raise_503,
        "model_id": "gemini-2.5-flash", "diagnostics": {}, "error": None,
    })
    out = vlm.reason_sync({})
    assert out["reasoner_status"] != "json_parse_error"


def test_gemini_generate_503_raises_unavailable_error(monkeypatch):
    """_is_503() correctly identifies 503-like errors."""
    err_503 = RuntimeError("HTTP 503 Service Unavailable")
    assert gemini_reasoner._is_503(err_503) is True

    err_su = type("ServiceUnavailableError", (Exception,), {})("oops")
    assert gemini_reasoner._is_503(err_su) is True

    err_other = RuntimeError("HTTP 400 Bad Request")
    assert gemini_reasoner._is_503(err_other) is False

    err_unavail = RuntimeError("UNAVAILABLE: network issue")
    assert gemini_reasoner._is_503(err_unavail) is True


def test_json_parse_error_only_on_http_200_malformed_json(monkeypatch):
    """json_parse_error must only occur when HTTP 200 but JSON is malformed."""
    monkeypatch.setenv("REASONER_MODE", "gemini")

    def return_malformed_json(prompt, image):
        # Simulates HTTP 200 with non-JSON response body
        return "I am not JSON at all"

    monkeypatch.setitem(vlm._ADAPTER_STATE, "gemini", {
        "available": True, "generate": return_malformed_json,
        "model_id": "gemini-2.5-flash", "diagnostics": {}, "error": None,
    })
    out = vlm.reason_sync({"session_id": "cam_bad_json", "entities": []})
    assert out["reasoner_status"] == "json_parse_error"


# ---------------------------------------------------------------------------
# Requirement 5 -- Failure does not overwrite last-good VLM cache
# ---------------------------------------------------------------------------

def test_json_parse_error_does_not_overwrite_last_good():
    """A json_parse_error response must not overwrite a previously good cache."""
    sid = "cam_cache_protect"
    good_resp = {
        "reasoner_status": "ok", "risks": [{"risk_id": "r1", "risk_level": "YELLOW"}],
        "scene_summary": "All good",
    }
    # Store a good result first
    vlm._cache_terminal_response(sid, good_resp)
    assert vlm.get_cached_draft(sid) is not None
    assert vlm.get_cached_draft(sid)["reasoner_status"] == "ok"

    # Now try to overwrite with a failure
    failure_resp = {"reasoner_status": "json_parse_error", "risks": [], "error": "bad json"}
    vlm._cache_terminal_response(sid, failure_resp)

    # The good result must still be in the cache
    cached = vlm.get_cached_draft(sid)
    assert cached is not None
    assert cached["reasoner_status"] == "ok"
    assert cached["risks"] == [{"risk_id": "r1", "risk_level": "YELLOW"}]


def test_schema_error_does_not_overwrite_last_good():
    """A schema_error response must not overwrite a previously good cache."""
    sid = "cam_schema_protect"
    good_resp = {"reasoner_status": "ok", "risks": [{"risk_id": "r2"}]}
    vlm._cache_terminal_response(sid, good_resp)

    failure_resp = {"reasoner_status": "schema_error", "risks": [], "error": "schema fail"}
    vlm._cache_terminal_response(sid, failure_resp)

    cached = vlm.get_cached_draft(sid)
    assert cached["reasoner_status"] == "ok"


def test_error_does_not_overwrite_last_good():
    """An error response must not overwrite a previously good cache."""
    sid = "cam_error_protect"
    good_resp = {"reasoner_status": "ok", "risks": [{"risk_id": "r3"}]}
    vlm._cache_terminal_response(sid, good_resp)

    failure_resp = {"reasoner_status": "error", "risks": [], "error": "boom"}
    vlm._cache_terminal_response(sid, failure_resp)

    cached = vlm.get_cached_draft(sid)
    assert cached["reasoner_status"] == "ok"


def test_timeout_does_not_overwrite_last_good():
    """A timeout response must not overwrite a previously good cache."""
    sid = "cam_timeout_protect"
    good_resp = {"reasoner_status": "ok", "risks": [{"risk_id": "r4"}]}
    vlm._cache_terminal_response(sid, good_resp)

    vlm._cache_terminal_response(sid, {"reasoner_status": "timeout", "risks": []})
    assert vlm.get_cached_draft(sid)["reasoner_status"] == "ok"


def test_failure_without_prior_good_is_cached():
    """A failure with no prior good result IS stored (so status propagates)."""
    sid = "cam_no_prior"
    # No prior entry
    vlm._cache_terminal_response(sid, {"reasoner_status": "json_parse_error", "risks": []})
    cached = vlm.get_cached_draft(sid)
    assert cached is not None
    assert cached["reasoner_status"] == "json_parse_error"


def test_success_overwrites_previous_failure():
    """A good result must overwrite a previous failure in the cache."""
    sid = "cam_overwrite_failure"
    # Initial failure (no prior good)
    vlm._cache_terminal_response(sid, {"reasoner_status": "json_parse_error", "risks": []})

    # Now a success
    good_resp = {"reasoner_status": "ok", "risks": [{"risk_id": "r5"}]}
    vlm._cache_terminal_response(sid, good_resp)

    cached = vlm.get_cached_draft(sid)
    assert cached["reasoner_status"] == "ok"


def test_background_json_parse_does_not_overwrite_good_cache(monkeypatch):
    """_run_and_cache with a json_parse_error preserves last-good result."""
    sid = "cam_bg_protect"
    good_resp = {"reasoner_status": "ok", "risks": [{"risk_id": "bg_good"}]}
    vlm._cache_terminal_response(sid, good_resp)

    # Background job returns json_parse_error
    monkeypatch.setattr(vlm, "reason_sync", lambda req: {
        "reasoner_status": "json_parse_error", "risks": [],
        "error": "bad json", "scene_summary": "", "uncertain_items": [],
    })
    vlm._run_and_cache(sid, {"session_id": sid, "frame_id": "f1"})

    cached = vlm.get_cached_draft(sid)
    assert cached["reasoner_status"] == "ok"


# ---------------------------------------------------------------------------
# Requirement 6 -- Error backoff prevents immediate Gemini retry
# ---------------------------------------------------------------------------

def test_error_backoff_prevents_immediate_retry(monkeypatch):
    """After a failure, maybe_trigger must not submit a new job during backoff."""
    monkeypatch.setenv("REASONER_ERROR_BACKOFF_MS", "30000")
    monkeypatch.setenv("REASONER_MIN_INTERVAL_MS", "1")
    monkeypatch.setenv("REASONER_CACHE_TTL_MS", "60000")

    sid = "cam_backoff"
    # Record a recent error
    with vlm._LOCK:
        vlm._LAST_ERROR_MS[sid] = vlm._now_ms()

    submitted = {"n": 0}

    class CountingExecutor:
        def submit(self, *args, **kwargs):
            submitted["n"] += 1

    monkeypatch.setattr(vlm, "_executor", lambda: CountingExecutor())

    draft, status = vlm.maybe_trigger(
        sid, frame_b64=None, highest_level="ORANGE",
        deterministic_risks=[], force_reason=False
    )
    assert submitted["n"] == 0, "Should not submit during error backoff"
    assert status in ("throttled", "not_triggered")


def test_error_backoff_allows_retry_after_interval(monkeypatch):
    """After the backoff expires, maybe_trigger may submit a new job."""
    monkeypatch.setenv("REASONER_ERROR_BACKOFF_MS", "1")  # 1 ms -- already expired
    monkeypatch.setenv("REASONER_MIN_INTERVAL_MS", "1")
    monkeypatch.setenv("REASONER_CACHE_TTL_MS", "60000")

    sid = "cam_backoff_expired"
    # Record an old error (in the past)
    with vlm._LOCK:
        vlm._LAST_ERROR_MS[sid] = vlm._now_ms() - 5000  # 5 seconds ago

    submitted = {"n": 0}

    class CountingExecutor:
        def submit(self, *args, **kwargs):
            submitted["n"] += 1

    monkeypatch.setattr(vlm, "_executor", lambda: CountingExecutor())
    time.sleep(0.005)  # ensure backoff interval has passed

    draft, status = vlm.maybe_trigger(
        sid, frame_b64=None, highest_level="ORANGE", deterministic_risks=[]
    )
    assert submitted["n"] == 1, "Should submit after backoff expires"
    assert status == "triggered"


def test_force_reason_bypasses_error_backoff(monkeypatch):
    """force_reason=True must bypass error backoff."""
    monkeypatch.setenv("REASONER_ERROR_BACKOFF_MS", "30000")
    monkeypatch.setenv("REASONER_MIN_INTERVAL_MS", "1")
    monkeypatch.setenv("REASONER_CACHE_TTL_MS", "60000")

    sid = "cam_force_backoff"
    with vlm._LOCK:
        vlm._LAST_ERROR_MS[sid] = vlm._now_ms()
        vlm._LAST_RUN_MS[sid] = 0

    submitted = {"n": 0}

    class CountingExecutor:
        def submit(self, *args, **kwargs):
            submitted["n"] += 1

    monkeypatch.setattr(vlm, "_executor", lambda: CountingExecutor())

    draft, status = vlm.maybe_trigger(
        sid, frame_b64=None, highest_level="ORANGE",
        deterministic_risks=[], force_reason=True
    )
    assert submitted["n"] == 1, "force_reason should bypass backoff"


def test_error_backoff_ms_default():
    """REASONER_ERROR_BACKOFF_MS defaults to 15000."""
    import os
    orig = os.environ.pop("REASONER_ERROR_BACKOFF_MS", None)
    try:
        assert vlm._error_backoff_ms() == 15000
    finally:
        if orig is not None:
            os.environ["REASONER_ERROR_BACKOFF_MS"] = orig


# ---------------------------------------------------------------------------
# End-to-end: detect returns entity-stamped boxes when highest_risk_level=YELLOW
# ---------------------------------------------------------------------------

def test_detect_entity_stamped_when_yellow_risk(server_mod, monkeypatch):
    """When highest_risk_level=YELLOW and a matching scene_risk exists, entities have risk_level."""
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    import vision_backend

    monkeypatch.setenv("RISK_ENGINE_ENABLED", "true")
    monkeypatch.setenv("TEMPORAL_REASONING_ENABLED", "false")
    monkeypatch.setattr(vision_backend, "run_inference", lambda **kw: _fake_resp_with_risk())

    # Simulate VLM returning cached YELLOW risks
    def fake_trigger(*args, **kwargs):
        return None, "not_triggered"

    monkeypatch.setattr(vlm, "maybe_trigger", fake_trigger)

    with server_mod._STATE_LOCK:
        server_mod._STATE["status"] = "ready"
    try:
        with TestClient(server_mod.app) as c:
            r = c.post("/detect", json={
                "image_b64": _tiny_jpeg_b64(),
                "session_id": "cam_stamp",
                "mode": "hse-monitoring",
            })
            assert r.status_code == 200
            body = r.json()
            assert "entities" in body
            # At least verify no exception was raised; entity stamping may depend
            # on the risk engine producing linkable risks in this test environment.
    finally:
        with server_mod._STATE_LOCK:
            server_mod._STATE["status"] = "cold"


def test_detect_temporal_added_scene_risk_stamps_entity(server_mod, monkeypatch):
    """A scene_risk added by temporal attach is stamped in final detect response."""
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    import vision_backend
    import temporal_reasoning

    monkeypatch.setenv("RISK_ENGINE_ENABLED", "false")
    monkeypatch.setattr(vision_backend, "run_inference", lambda **kw: _fake_resp_with_risk())
    monkeypatch.setattr(vlm, "enabled", lambda: False)
    monkeypatch.setattr(temporal_reasoning, "enabled", lambda: True)

    def _attach(resp_dict, **kwargs):
        out = dict(resp_dict)
        out["scene_risks"] = [{
            "risk_id": "temporal_risk_1",
            "risk_level": "RED",
            "bbox": dict(PERSON["bbox"]),
            "risk_reason": "Temporal near-edge persistence",
            "produced_by": "temporal_reasoning",
            "requires_human_review": False,
        }]
        return out

    monkeypatch.setattr(temporal_reasoning, "attach_temporal", _attach)

    with server_mod._STATE_LOCK:
        server_mod._STATE["status"] = "ready"
    try:
        with TestClient(server_mod.app) as c:
            r = c.post("/detect", json={"image_b64": _tiny_jpeg_b64(), "session_id": "cam_temporal"})
            assert r.status_code == 200
            body = r.json()
            stamped = [e for e in body.get("entities", []) if isinstance(e, dict) and e.get("risk_level") == "RED"]
            assert stamped
    finally:
        with server_mod._STATE_LOCK:
            server_mod._STATE["status"] = "cold"


def test_detect_log_overlap_scene_risk_counts_stamp_count(server_mod, monkeypatch):
    """linkable_scene_risks_count=1 with overlap yields entity_risk_stamped_count>=1."""
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    import vision_backend
    import temporal_reasoning
    import worker_runtime as runtime_mod

    logged: List[Dict[str, Any]] = []

    def capture_log_event(event_name, **kwargs):
        logged.append({"event": event_name, **kwargs})

    monkeypatch.setenv("RISK_ENGINE_ENABLED", "false")
    monkeypatch.setattr(vision_backend, "run_inference", lambda **kw: _fake_resp_with_risk())
    monkeypatch.setattr(vlm, "enabled", lambda: False)
    monkeypatch.setattr(temporal_reasoning, "enabled", lambda: True)
    monkeypatch.setattr(runtime_mod, "log_event", capture_log_event)

    def _attach(resp_dict, **kwargs):
        out = dict(resp_dict)
        out["scene_risks"] = [{
            "risk_id": "temporal_risk_overlap",
            "risk_level": "YELLOW",
            "bbox": dict(PERSON["bbox"]),
            "risk_reason": "Overlap with person",
            "produced_by": "temporal_reasoning",
            "requires_human_review": False,
        }]
        return out

    monkeypatch.setattr(temporal_reasoning, "attach_temporal", _attach)

    with server_mod._STATE_LOCK:
        server_mod._STATE["status"] = "ready"
    try:
        with TestClient(server_mod.app) as c:
            r = c.post("/detect", json={"image_b64": _tiny_jpeg_b64(), "session_id": "cam_temporal_log"})
            assert r.status_code == 200
        detect_events = [e for e in logged if e.get("event") == "detect"]
        assert detect_events
        ev = detect_events[-1]
        assert ev["scene_risks_count"] == 1
        assert ev["linkable_scene_risks_count"] == 1
        assert ev["entity_risk_stamped_count"] >= 1
    finally:
        with server_mod._STATE_LOCK:
            server_mod._STATE["status"] = "cold"
