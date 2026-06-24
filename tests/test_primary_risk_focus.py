"""
tests/test_primary_risk_focus.py -- Unit tests for risk/primary_risk_focus.py

Covers the Part-10 checklist for PR #29:
  [x] semantic cache / focus cache does not keep a box yellow after risk is gone
      (focus never creates risk; invalid without current-frame support)
  [x] cached VLM yellow is ignored when current deterministic support is missing
  [x] primary risk focus expires after PRIMARY_RISK_FOCUS_TTL_MS
  [x] primary object stays stampable when current frame still supports the risk
  [x] context/support box becomes candidate_only (not stamped) in apply mode
  [x] shadow mode logs would_suppress but does NOT change scene_risks
  [x] duplicate objects with same label remain separate by track_id
  [x] same semantic label is NOT used as a risk key (keyed by hazard+track)
  [x] PPE/person/vehicle hazards are not suppressed by object-near-edge focus
  [x] RED/ORANGE independent risks are not suppressed
  [x] no images/crops/base64/full Gemini responses are stored
  [x] GeminiBoxDecision risk_role/context_for_box_id are optional + accepted
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

import pytest

import risk.primary_risk_focus as prf


@pytest.fixture(autouse=True)
def _reset_cache():
    prf.reset()
    yield
    prf.reset()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _entity(track_id, x=0.10):
    return {"track_id": track_id, "bbox": {"x": x, "y": 0.1, "w": 0.1, "h": 0.1}}


def _det(hazard, track_id):
    return {"hazard_type": hazard, "involved_track_ids": [str(track_id)]}


def _risk(risk_id, hazard, level, track_id):
    return {"risk_id": risk_id, "hazard_type": hazard, "risk_level": level,
            "involved_track_ids": [str(track_id)]}


def _apply_env(monkeypatch):
    monkeypatch.setenv("PRIMARY_RISK_FOCUS_APPLY_ENABLED", "true")
    monkeypatch.setenv("PRIMARY_RISK_CONTEXT_SUPPRESS_APPLY_ENABLED", "true")


# ---------------------------------------------------------------------------
# update() keying + gating  (Part 8: keyed by hazard+track, never by label)
# ---------------------------------------------------------------------------

def test_update_keys_by_hazard_and_track_not_label():
    assert prf.update("s1", hazard_type="object_near_edge", risk_level="YELLOW",
                      confidence=0.72, track_id=17) is True
    entry = prf.lookup("s1", hazard_type="object_near_edge", track_id=17)
    assert entry is not None
    # No label/text key is used; only small metadata is stored.
    assert "semantic_label" not in entry and "label" not in entry
    # A different track / hazard is a different (absent) key.
    assert prf.lookup("s1", hazard_type="object_near_edge", track_id=22) is None
    assert prf.lookup("s1", hazard_type="blocked_path", track_id=17) is None


def test_duplicate_objects_same_label_separate_by_track_id():
    # Two objects that a semantic cache would call "plant pot" stay separate here.
    prf.update("s1", hazard_type="object_near_edge", risk_level="YELLOW", confidence=0.7, track_id=17)
    prf.update("s1", hazard_type="object_near_edge", risk_level="YELLOW", confidence=0.7, track_id=22)
    assert prf.cache_size("s1") == 2
    assert prf.lookup("s1", hazard_type="object_near_edge", track_id=17) is not None
    assert prf.lookup("s1", hazard_type="object_near_edge", track_id=22) is not None


def test_update_rejects_low_conf_excluded_hazard_and_non_risk_level():
    assert prf.update("s1", hazard_type="object_near_edge", risk_level="YELLOW",
                      confidence=0.50, track_id=1) is False          # below min conf
    assert prf.update("s1", hazard_type="ppe_missing", risk_level="YELLOW",
                      confidence=0.9, track_id=1) is False           # not a focus hazard
    assert prf.update("s1", hazard_type="object_near_edge", risk_level="GREEN",
                      confidence=0.9, track_id=1) is False           # GREEN is not a risk
    assert prf.cache_size("s1") == 0


def test_update_requires_track_or_detection_id():
    assert prf.update("s1", hazard_type="object_near_edge", risk_level="YELLOW",
                      confidence=0.9) is False


def test_disabled_rejects_all(monkeypatch):
    monkeypatch.setenv("PRIMARY_RISK_FOCUS_ENABLED", "false")
    assert prf.update("s1", hazard_type="object_near_edge", risk_level="YELLOW",
                      confidence=0.9, track_id=1) is False


# ---------------------------------------------------------------------------
# Expiry (Part: focus expires after TTL)
# ---------------------------------------------------------------------------

def test_focus_expires_after_ttl(monkeypatch):
    monkeypatch.setenv("PRIMARY_RISK_FOCUS_TTL_MS", "1000")
    now = [prf._now_ms()]
    monkeypatch.setattr(prf, "_now_ms", lambda: now[0])
    prf.update("s1", hazard_type="object_near_edge", risk_level="YELLOW", confidence=0.7, track_id=17)
    assert prf.lookup("s1", hazard_type="object_near_edge", track_id=17) is not None
    now[0] += 2000  # past TTL
    assert prf.lookup("s1", hazard_type="object_near_edge", track_id=17) is None
    assert prf.cache_size("s1") == 0


# ---------------------------------------------------------------------------
# Current-frame sync gate (Part 5)
# ---------------------------------------------------------------------------

def test_primary_valid_when_current_frame_supports():
    prf.update("s1", hazard_type="object_near_edge", risk_level="YELLOW", confidence=0.7, track_id=17)
    entry = prf.lookup("s1", hazard_type="object_near_edge", track_id=17)
    assert prf.validate_focus_entry(
        entry, current_entities=[_entity(17)],
        current_det_risks=[_det("object_near_edge", 17)], session_id="s1") is True


def test_cached_yellow_invalid_without_current_det_support(monkeypatch):
    monkeypatch.setenv("PRIMARY_RISK_FOCUS_REQUIRE_CURRENT_DETERMINISTIC_SUPPORT", "true")
    prf.update("s1", hazard_type="object_near_edge", risk_level="YELLOW", confidence=0.7, track_id=17)
    entry = prf.lookup("s1", hazard_type="object_near_edge", track_id=17)
    # Entity visible but NO matching deterministic risk this frame -> invalid.
    assert prf.validate_focus_entry(
        entry, current_entities=[_entity(17)], current_det_risks=[], session_id="s1") is False
    # Entity no longer visible -> invalid even with det support.
    assert prf.validate_focus_entry(
        entry, current_entities=[_entity(99)],
        current_det_risks=[_det("object_near_edge", 17)], session_id="s1") is False


# ---------------------------------------------------------------------------
# apply_focus_filter -- shadow mode (default)
# ---------------------------------------------------------------------------

def test_shadow_mode_logs_would_suppress_without_mutation():
    prf.update("s1", hazard_type="object_near_edge", risk_level="YELLOW", confidence=0.7, track_id=17)
    resp = {
        "entities": [_entity(17)],
        "scene_risks": [
            _risk("prim", "object_near_edge", "YELLOW", 17),
            _risk("ctx", "object_near_edge", "YELLOW", 99),
        ],
        "risks": [_det("object_near_edge", 17)],
    }
    before = copy.deepcopy(resp["scene_risks"])
    stats = prf.apply_focus_filter(resp, session_id="s1", current_det_risks=resp["risks"])
    assert stats["primary_risk_focus_shadow_mode"] is True
    assert stats["primary_risk_context_would_suppress_count"] == 1
    assert stats["primary_risk_context_suppressed_count"] == 0
    # scene_risks unchanged -- no candidate_only added.
    assert resp["scene_risks"] == before
    assert resp["scene_risks"][1].get("candidate_only") is None


def test_shadow_mode_reports_cache_and_update_counts():
    prf.update("s1", hazard_type="object_near_edge", risk_level="YELLOW", confidence=0.7, track_id=17)
    resp = {"entities": [_entity(17)], "scene_risks": [], "risks": [_det("object_near_edge", 17)]}
    stats = prf.apply_focus_filter(resp, session_id="s1", current_det_risks=resp["risks"])
    assert stats["primary_risk_focus_cache_size"] == 1
    assert stats["primary_risk_focus_update_count"] == 1
    assert stats["primary_risk_focus_valid_count"] == 1
    assert stats["reasoner_linked_risk_ttl_ms"] >= 0


# ---------------------------------------------------------------------------
# apply_focus_filter -- apply mode
# ---------------------------------------------------------------------------

def test_apply_mode_suppresses_context_keeps_primary(monkeypatch):
    _apply_env(monkeypatch)
    prf.update("s1", hazard_type="object_near_edge", risk_level="YELLOW", confidence=0.7, track_id=17)
    resp = {
        "entities": [_entity(17)],
        "scene_risks": [
            _risk("prim", "object_near_edge", "YELLOW", 17),
            _risk("ctx", "object_near_edge", "YELLOW", 99),
        ],
        "risks": [_det("object_near_edge", 17)],
    }
    stats = prf.apply_focus_filter(resp, session_id="s1", current_det_risks=resp["risks"])
    assert stats["primary_risk_focus_shadow_mode"] is False
    assert stats["primary_risk_context_suppressed_count"] == 1
    # Primary stays stampable (no candidate_only); context is suppressed.
    assert resp["scene_risks"][0].get("candidate_only") is None
    assert resp["scene_risks"][1].get("candidate_only") is True
    assert resp["scene_risks"][1].get("suppressed_by_primary_focus") is True
    assert resp["scene_risks"][1].get("primary_focus_track_id") == "17"


def test_apply_mode_no_change_without_current_support(monkeypatch):
    _apply_env(monkeypatch)
    prf.update("s1", hazard_type="object_near_edge", risk_level="YELLOW", confidence=0.7, track_id=17)
    resp = {
        "entities": [_entity(17)],
        "scene_risks": [_risk("ctx", "object_near_edge", "YELLOW", 99)],
        "risks": [],  # no deterministic support this frame
    }
    stats = prf.apply_focus_filter(resp, session_id="s1", current_det_risks=[])
    assert stats["primary_risk_context_suppressed_count"] == 0
    assert resp["scene_risks"][0].get("candidate_only") is None


def test_apply_mode_does_not_suppress_red_or_orange_independent(monkeypatch):
    _apply_env(monkeypatch)
    prf.update("s1", hazard_type="object_near_edge", risk_level="YELLOW", confidence=0.7, track_id=17)
    resp = {
        "entities": [_entity(17)],
        "scene_risks": [
            _risk("prim", "object_near_edge", "YELLOW", 17),
            _risk("red", "object_near_edge", "RED", 55),
            _risk("orange", "object_near_edge", "ORANGE", 66),
        ],
        "risks": [_det("object_near_edge", 17)],
    }
    prf.apply_focus_filter(resp, session_id="s1", current_det_risks=resp["risks"])
    assert resp["scene_risks"][1].get("candidate_only") is None  # RED kept
    assert resp["scene_risks"][2].get("candidate_only") is None  # ORANGE kept


def test_apply_mode_does_not_suppress_excluded_hazards(monkeypatch):
    _apply_env(monkeypatch)
    prf.update("s1", hazard_type="object_near_edge", risk_level="YELLOW", confidence=0.7, track_id=17)
    resp = {
        "entities": [_entity(17)],
        "scene_risks": [
            _risk("prim", "object_near_edge", "YELLOW", 17),
            _risk("ppe", "ppe_missing", "YELLOW", 99),
            _risk("veh", "worker_near_vehicle", "YELLOW", 88),
        ],
        "risks": [_det("object_near_edge", 17)],
    }
    stats = prf.apply_focus_filter(resp, session_id="s1", current_det_risks=resp["risks"])
    assert resp["scene_risks"][1].get("candidate_only") is None  # PPE untouched
    assert resp["scene_risks"][2].get("candidate_only") is None  # vehicle untouched
    assert stats["primary_risk_context_suppressed_count"] == 0


def test_focus_never_creates_risk_or_keeps_box_yellow(monkeypatch):
    # Focus only suppresses; it must never ADD a scene_risk or stamp a risk_level.
    _apply_env(monkeypatch)
    prf.update("s1", hazard_type="object_near_edge", risk_level="YELLOW", confidence=0.7, track_id=17)
    resp = {"entities": [_entity(17)], "scene_risks": [], "risks": [_det("object_near_edge", 17)]}
    prf.apply_focus_filter(resp, session_id="s1", current_det_risks=resp["risks"])
    assert resp["scene_risks"] == []
    assert "risk_level" not in resp["entities"][0]
    assert "risk_color" not in resp["entities"][0]


# ---------------------------------------------------------------------------
# Per-session isolation + no-blob storage
# ---------------------------------------------------------------------------

def test_cache_is_per_session():
    prf.update("sA", hazard_type="object_near_edge", risk_level="YELLOW", confidence=0.7, track_id=1)
    prf.update("sB", hazard_type="object_near_edge", risk_level="YELLOW", confidence=0.7, track_id=1)
    assert prf.lookup("sA", hazard_type="object_near_edge", track_id=1) is not None
    prf.expire_session("sA")
    assert prf.lookup("sA", hazard_type="object_near_edge", track_id=1) is None
    assert prf.lookup("sB", hazard_type="object_near_edge", track_id=1) is not None


def test_no_image_or_blob_stored():
    prf.update("s1", hazard_type="object_near_edge", risk_level="YELLOW", confidence=0.7,
               track_id=17, detection_id=3)
    entry = prf.lookup("s1", hazard_type="object_near_edge", track_id=17)
    assert entry is not None
    allowed = {"primary_track_id", "primary_detection_id", "hazard_type", "risk_level",
               "confidence", "last_seen_ms", "expires_at_ms", "source"}
    assert set(entry).issubset(allowed)
    for key, val in entry.items():
        assert not isinstance(val, bytes), f"{key} holds bytes"
        if isinstance(val, str):
            assert len(val) <= 64, f"{key} looks like a blob"


# ---------------------------------------------------------------------------
# GeminiBoxDecision risk_role / context_for_box_id (Part 2)
# ---------------------------------------------------------------------------

def test_gemini_box_decision_risk_role_optional():
    from risk.gemini_reasoner import GeminiBoxDecision
    bd = GeminiBoxDecision(box_id="A", hazard_type="object_near_edge", severity=2, likelihood=2)
    assert bd.risk_role is None and bd.context_for_box_id is None


def test_gemini_box_decision_accepts_risk_role():
    from risk.gemini_reasoner import GeminiBoxDecision
    bd = GeminiBoxDecision(box_id="A", hazard_type="object_near_edge", severity=2,
                           likelihood=2, risk_role="primary")
    assert bd.risk_role == "primary"
    ctx = GeminiBoxDecision(box_id="B", hazard_type="object_near_edge", severity=1,
                            likelihood=1, risk_role="context", context_for_box_id="A")
    assert ctx.risk_role == "context" and ctx.context_for_box_id == "A"
