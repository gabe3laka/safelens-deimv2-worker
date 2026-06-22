"""
tests/test_semantic_memory.py -- Unit tests for risk/semantic_memory.py

Covers:
  [x] semantic label stores by session_id + track_id, not label text
  [x] multiple identical labels remain separate by track_id
  [x] same semantic label is not applied to all same-label objects
  [x] semantic label expires after TTL
  [x] semantic label does not overwrite YOLO label
  [x] semantic label does not create risk_level
  [x] risk words are rejected
  [x] no image/crop/base64/full Gemini text is stored
  [x] cache is per-session, not global
  [x] bbox fallback is skipped when ambiguous
  [x] SEMANTIC_LABEL_APPLY_ENABLED=false -> shadow mode: logs but no entity mutation
  [x] SEMANTIC_LABEL_APPLY_ENABLED=true -> adds semantic_label/display_label only
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

import pytest

import risk.semantic_memory as sem


@pytest.fixture(autouse=True)
def _reset_cache():
    sem.reset()
    yield
    sem.reset()


# ---------------------------------------------------------------------------
# Sanitizer tests
# ---------------------------------------------------------------------------

def test_sanitizer_allows_clean_label():
    assert sem.sanitize_label("plant pot") == "plant pot"
    assert sem.sanitize_label("office chair") == "office chair"
    assert sem.sanitize_label("metal cabinet") == "metal cabinet"
    assert sem.sanitize_label("fire extinguisher") == "fire extinguisher"


def test_sanitizer_lowercase_and_trim():
    assert sem.sanitize_label("  Plant Pot  ") == "plant pot"


def test_sanitizer_rejects_empty():
    assert sem.sanitize_label("") is None
    assert sem.sanitize_label("   ") is None
    assert sem.sanitize_label(None) is None


def test_sanitizer_rejects_risk_words():
    for word in ("danger", "dangerous", "hazard", "hazardous", "unsafe",
                 "risk", "yellow", "orange", "red", "fall", "falling",
                 "edge", "warning", "alert"):
        result = sem.sanitize_label(word)
        assert result is None, f"Expected '{word}' to be rejected"
    # Risk word embedded in phrase.
    assert sem.sanitize_label("near edge plant") is None
    assert sem.sanitize_label("dangerous box") is None


def test_sanitizer_rejects_sentences():
    # Too many words.
    assert sem.sanitize_label("this is a very long sentence about something") is None
    # Contains punctuation.
    assert sem.sanitize_label("plant pot, on floor") is None
    assert sem.sanitize_label("plant (pot)") is None


def test_sanitizer_rejects_disallowed_chars():
    assert sem.sanitize_label("plant@pot") is None
    assert sem.sanitize_label("box!") is None


def test_sanitizer_max_length():
    long_label = "a" * 49
    result = sem.sanitize_label(long_label)
    assert result is not None
    assert len(result) <= 48


def test_sanitizer_allows_hyphens_and_slashes():
    assert sem.sanitize_label("hi-vis vest") == "hi-vis vest"
    assert sem.sanitize_label("a/b panel") == "a/b panel"


# ---------------------------------------------------------------------------
# Store by session_id + track_id
# ---------------------------------------------------------------------------

def test_stores_by_session_and_track_id(monkeypatch):
    monkeypatch.setenv("SEMANTIC_LABEL_CACHE_ENABLED", "true")
    ok = sem.update("sess1", track_id=17, semantic_label="plant pot", confidence=0.8)
    assert ok is True
    entry = sem.lookup("sess1", track_id=17)
    assert entry is not None
    assert entry["semantic_label"] == "plant pot"


def test_different_sessions_are_isolated(monkeypatch):
    monkeypatch.setenv("SEMANTIC_LABEL_CACHE_ENABLED", "true")
    sem.update("sess1", track_id=17, semantic_label="plant pot", confidence=0.8)
    sem.update("sess2", track_id=17, semantic_label="office chair", confidence=0.8)

    e1 = sem.lookup("sess1", track_id=17)
    e2 = sem.lookup("sess2", track_id=17)
    assert e1["semantic_label"] == "plant pot"
    assert e2["semantic_label"] == "office chair"


def test_multiple_identical_labels_separate_by_track_id(monkeypatch):
    """Multiple objects with same semantic label must be keyed by track, not by label."""
    monkeypatch.setenv("SEMANTIC_LABEL_CACHE_ENABLED", "true")
    sem.update("sess1", track_id=17, semantic_label="plant pot", confidence=0.8)
    sem.update("sess1", track_id=22, semantic_label="plant pot", confidence=0.8)
    sem.update("sess1", track_id=35, semantic_label="plant pot", confidence=0.8)

    # All three entries exist independently.
    assert sem.cache_size("sess1") == 3
    assert sem.lookup("sess1", track_id=17)["semantic_label"] == "plant pot"
    assert sem.lookup("sess1", track_id=22)["semantic_label"] == "plant pot"
    assert sem.lookup("sess1", track_id=35)["semantic_label"] == "plant pot"


# ---------------------------------------------------------------------------
# Expiry
# ---------------------------------------------------------------------------

def test_label_expires_after_ttl(monkeypatch):
    monkeypatch.setenv("SEMANTIC_LABEL_CACHE_TTL_MS", "1000")
    monkeypatch.setenv("SEMANTIC_LABEL_CACHE_ENABLED", "true")

    now_ref = [sem._now_ms()]

    def fake_now():
        return now_ref[0]

    monkeypatch.setattr(sem, "_now_ms", fake_now)

    sem.update("sess1", track_id=1, semantic_label="cup", confidence=0.9)
    assert sem.lookup("sess1", track_id=1) is not None

    # Advance time past the TTL.
    now_ref[0] += 2000
    assert sem.lookup("sess1", track_id=1) is None


def test_label_not_expired_within_ttl(monkeypatch):
    monkeypatch.setenv("SEMANTIC_LABEL_CACHE_TTL_MS", "5000")
    monkeypatch.setenv("SEMANTIC_LABEL_CACHE_ENABLED", "true")
    sem.update("sess1", track_id=1, semantic_label="cup", confidence=0.9)
    assert sem.lookup("sess1", track_id=1) is not None


# ---------------------------------------------------------------------------
# Confidence threshold
# ---------------------------------------------------------------------------

def test_low_confidence_rejected(monkeypatch):
    monkeypatch.setenv("SEMANTIC_LABEL_MIN_CONFIDENCE", "0.55")
    monkeypatch.setenv("SEMANTIC_LABEL_CACHE_ENABLED", "true")
    ok = sem.update("sess1", track_id=1, semantic_label="cup", confidence=0.4)
    assert ok is False
    assert sem.lookup("sess1", track_id=1) is None


def test_exact_min_confidence_accepted(monkeypatch):
    monkeypatch.setenv("SEMANTIC_LABEL_MIN_CONFIDENCE", "0.55")
    monkeypatch.setenv("SEMANTIC_LABEL_CACHE_ENABLED", "true")
    ok = sem.update("sess1", track_id=1, semantic_label="cup", confidence=0.55)
    assert ok is True


# ---------------------------------------------------------------------------
# Cache disabled
# ---------------------------------------------------------------------------

def test_cache_disabled_rejects_all(monkeypatch):
    monkeypatch.setenv("SEMANTIC_LABEL_CACHE_ENABLED", "false")
    ok = sem.update("sess1", track_id=1, semantic_label="cup", confidence=0.9)
    assert ok is False
    assert sem.lookup("sess1", track_id=1) is None


# ---------------------------------------------------------------------------
# apply_to_entities -- shadow mode (APPLY_ENABLED=false)
# ---------------------------------------------------------------------------

def _make_resp_dict(entities):
    return {"entities": entities, "tracks": []}


def test_shadow_mode_does_not_mutate_entities(monkeypatch):
    """Shadow mode: labels stored but entities are NOT mutated."""
    monkeypatch.setenv("SEMANTIC_LABEL_CACHE_ENABLED", "true")
    sem.update("sess1", track_id=5, semantic_label="plant pot", confidence=0.8)

    entity = {"track_id": 5, "label": "object", "confidence": 0.9}
    resp = _make_resp_dict([entity])
    stats = sem.apply_to_entities(resp, session_id="sess1", tracks=[], apply_enabled=False)

    # Entity must NOT be mutated.
    assert "semantic_label" not in entity
    assert "display_label" not in entity
    # Applied count must be 0 in shadow mode.
    assert stats["semantic_label_applied_count"] == 0
    # Track match count still increments for logging.
    assert stats["semantic_label_track_match_count"] == 1


def test_shadow_mode_cache_size_logged(monkeypatch):
    monkeypatch.setenv("SEMANTIC_LABEL_CACHE_ENABLED", "true")
    sem.update("sess1", track_id=5, semantic_label="plant pot", confidence=0.8)
    sem.update("sess1", track_id=6, semantic_label="office chair", confidence=0.8)

    resp = _make_resp_dict([])
    stats = sem.apply_to_entities(resp, session_id="sess1", tracks=[], apply_enabled=False)
    assert stats["semantic_label_cache_size"] == 2


# ---------------------------------------------------------------------------
# apply_to_entities -- apply mode (APPLY_ENABLED=true)
# ---------------------------------------------------------------------------

def test_apply_mode_adds_semantic_label_to_entity(monkeypatch):
    monkeypatch.setenv("SEMANTIC_LABEL_CACHE_ENABLED", "true")
    sem.update("sess1", track_id=5, semantic_label="plant pot", confidence=0.8)

    entity = {"track_id": 5, "label": "object", "confidence": 0.9}
    resp = _make_resp_dict([entity])
    stats = sem.apply_to_entities(resp, session_id="sess1", tracks=[], apply_enabled=True)

    assert entity.get("semantic_label") == "plant pot"
    assert entity.get("display_label") == "plant pot"
    assert entity.get("semantic_label_confidence") == pytest.approx(0.8, abs=0.01)
    assert entity.get("semantic_label_source") == "gemini_cache"
    assert "semantic_label_age_ms" in entity
    assert stats["semantic_label_applied_count"] == 1
    assert stats["semantic_label_track_match_count"] == 1


def test_apply_mode_does_not_overwrite_yolo_label(monkeypatch):
    """YOLO label field must never be overwritten."""
    monkeypatch.setenv("SEMANTIC_LABEL_CACHE_ENABLED", "true")
    sem.update("sess1", track_id=5, semantic_label="plant pot", confidence=0.8)

    entity = {"track_id": 5, "label": "person", "confidence": 0.9}
    resp = _make_resp_dict([entity])
    sem.apply_to_entities(resp, session_id="sess1", tracks=[], apply_enabled=True)

    # Original YOLO label must be preserved.
    assert entity["label"] == "person"
    # Semantic label is added as a separate field.
    assert entity.get("semantic_label") == "plant pot"


def test_apply_mode_does_not_create_risk_level(monkeypatch):
    """Semantic label must never create or modify risk_level."""
    monkeypatch.setenv("SEMANTIC_LABEL_CACHE_ENABLED", "true")
    sem.update("sess1", track_id=5, semantic_label="plant pot", confidence=0.8)

    entity = {"track_id": 5, "label": "object"}
    resp = _make_resp_dict([entity])
    sem.apply_to_entities(resp, session_id="sess1", tracks=[], apply_enabled=True)

    assert "risk_level" not in entity
    assert "risk_score" not in entity
    assert "hazard_type" not in entity
    assert "should_alert" not in entity


def test_apply_mode_risk_level_not_overwritten(monkeypatch):
    """Existing risk_level must not be modified."""
    monkeypatch.setenv("SEMANTIC_LABEL_CACHE_ENABLED", "true")
    sem.update("sess1", track_id=5, semantic_label="plant pot", confidence=0.8)

    entity = {"track_id": 5, "label": "person", "risk_level": "ORANGE",
              "risk_score": 9, "should_alert": True}
    resp = _make_resp_dict([entity])
    sem.apply_to_entities(resp, session_id="sess1", tracks=[], apply_enabled=True)

    assert entity["risk_level"] == "ORANGE"
    assert entity["risk_score"] == 9
    assert entity["should_alert"] is True


# ---------------------------------------------------------------------------
# Duplicate-label safety: same label not applied to all same-label objects
# ---------------------------------------------------------------------------

def test_same_label_not_applied_to_all_objects(monkeypatch):
    """When two entities have the same label in cache, each track maps to its own entry."""
    monkeypatch.setenv("SEMANTIC_LABEL_CACHE_ENABLED", "true")
    sem.update("sess1", track_id=17, semantic_label="plant pot", confidence=0.8)
    sem.update("sess1", track_id=22, semantic_label="plant pot", confidence=0.75)

    entity_17 = {"track_id": 17, "label": "object"}
    entity_22 = {"track_id": 22, "label": "object"}
    entity_99 = {"track_id": 99, "label": "object"}  # no cached label

    resp = _make_resp_dict([entity_17, entity_22, entity_99])
    stats = sem.apply_to_entities(resp, session_id="sess1", tracks=[], apply_enabled=True)

    # Each entity gets its own cached label by track identity, not by label text.
    assert entity_17["semantic_label"] == "plant pot"
    assert entity_22["semantic_label"] == "plant pot"
    assert "semantic_label" not in entity_99
    assert stats["semantic_label_applied_count"] == 2


# ---------------------------------------------------------------------------
# bbox fallback ambiguity detection
# ---------------------------------------------------------------------------

def test_bbox_fallback_ambiguous_skipped(monkeypatch):
    """When two entities could match the same cached bbox, skip as ambiguous."""
    monkeypatch.setenv("SEMANTIC_LABEL_CACHE_ENABLED", "true")
    monkeypatch.setenv("SEMANTIC_LABEL_BBOX_FALLBACK_MAX_AGE_MS", "5000")
    monkeypatch.setenv("SEMANTIC_LABEL_BBOX_FALLBACK_IOU_MIN", "0.5")

    # Manually insert a cache entry with a stored bbox.
    now = sem._now_ms()
    with sem._LOCK:
        sem._CACHE["sess1"] = {
            "track_7": {
                "semantic_label": "plant pot",
                "confidence": 0.8,
                "last_seen_ms": now,
                "expires_at_ms": now + 15000,
                "source": "gemini",
                "bbox": {"x": 0.1, "y": 0.1, "w": 0.2, "h": 0.2},
            }
        }

    # Two entities with similar bboxes (both would match by IoU).
    e1 = {"label": "object", "bbox": {"x": 0.1, "y": 0.1, "w": 0.2, "h": 0.2}}
    e2 = {"label": "object", "bbox": {"x": 0.12, "y": 0.12, "w": 0.18, "h": 0.18}}

    resp = _make_resp_dict([e1, e2])
    stats = sem.apply_to_entities(resp, session_id="sess1", tracks=[], apply_enabled=True)

    # Neither should be applied due to ambiguity.
    assert "semantic_label" not in e1
    assert "semantic_label" not in e2
    assert stats["semantic_label_ambiguous_skip_count"] >= 1


# ---------------------------------------------------------------------------
# per-session isolation
# ---------------------------------------------------------------------------

def test_cache_is_per_session(monkeypatch):
    monkeypatch.setenv("SEMANTIC_LABEL_CACHE_ENABLED", "true")
    sem.update("sess_A", track_id=1, semantic_label="cup", confidence=0.8)
    sem.update("sess_B", track_id=1, semantic_label="vase", confidence=0.8)

    assert sem.lookup("sess_A", track_id=1)["semantic_label"] == "cup"
    assert sem.lookup("sess_B", track_id=1)["semantic_label"] == "vase"

    # Expiring session A does not affect session B.
    sem.expire_session("sess_A")
    assert sem.lookup("sess_A", track_id=1) is None
    assert sem.lookup("sess_B", track_id=1) is not None


def test_reset_clears_all_sessions(monkeypatch):
    monkeypatch.setenv("SEMANTIC_LABEL_CACHE_ENABLED", "true")
    sem.update("sess_A", track_id=1, semantic_label="cup", confidence=0.8)
    sem.update("sess_B", track_id=2, semantic_label="vase", confidence=0.8)
    sem.reset()
    assert sem.lookup("sess_A", track_id=1) is None
    assert sem.lookup("sess_B", track_id=2) is None


# ---------------------------------------------------------------------------
# Max per session eviction
# ---------------------------------------------------------------------------

def test_max_per_session_evicts_oldest(monkeypatch):
    monkeypatch.setenv("SEMANTIC_LABEL_CACHE_ENABLED", "true")
    monkeypatch.setenv("SEMANTIC_LABEL_MAX_PER_SESSION", "3")

    sem.update("sess1", track_id=1, semantic_label="cup", confidence=0.8)
    sem.update("sess1", track_id=2, semantic_label="vase", confidence=0.8)
    sem.update("sess1", track_id=3, semantic_label="plant pot", confidence=0.8)
    sem.update("sess1", track_id=4, semantic_label="chair", confidence=0.8)

    # Cache should not exceed max.
    assert sem.cache_size("sess1") <= 3


# ---------------------------------------------------------------------------
# No images/crops/base64/full Gemini responses stored
# ---------------------------------------------------------------------------

def test_no_image_or_base64_stored(monkeypatch):
    monkeypatch.setenv("SEMANTIC_LABEL_CACHE_ENABLED", "true")
    sem.update("sess1", track_id=1, semantic_label="cup", confidence=0.8)

    entry = sem.lookup("sess1", track_id=1)
    assert entry is not None
    for key in entry:
        val = entry[key]
        # No binary/image data stored.
        assert not isinstance(val, bytes), f"Field {key} contains bytes"
        if isinstance(val, str):
            # Should not be a base64 blob (no long strings of base64 chars).
            assert len(val) <= 128, f"Field {key} looks like a large blob: {val[:50]}"


# ---------------------------------------------------------------------------
# GeminiBoxDecision schema -- semantic_label field
# ---------------------------------------------------------------------------

def test_gemini_box_decision_has_semantic_label_field():
    from risk.gemini_reasoner import GeminiBoxDecision
    # semantic_label is optional and defaults to None.
    bd = GeminiBoxDecision(
        box_id="A",
        hazard_type="other",
        severity=2,
        likelihood=2,
    )
    assert bd.semantic_label is None


def test_gemini_box_decision_accepts_semantic_label():
    from risk.gemini_reasoner import GeminiBoxDecision
    bd = GeminiBoxDecision(
        box_id="A",
        hazard_type="other",
        severity=2,
        likelihood=2,
        semantic_label="plant pot",
    )
    assert bd.semantic_label == "plant pot"


def test_gemini_box_decision_rejects_long_semantic_label():
    from risk.gemini_reasoner import GeminiBoxDecision
    import pytest as _pytest
    with _pytest.raises(Exception):
        GeminiBoxDecision(
            box_id="A",
            hazard_type="other",
            severity=2,
            likelihood=2,
            semantic_label="x" * 49,  # exceeds max_length=48
        )
