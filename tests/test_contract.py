"""
tests/test_contract.py -- golden API-contract fixtures (B10).

These fixtures are the shared contract the app repo can also test against:
  detect_old.json      -- legacy /detect shape (must keep parsing; no schema_version)
  detect_risk.json     -- risk-aware /detect (schema_version risk.v1 + risk block)
  detect_degraded.json -- risk failed but detection preserved (no 500)
  reason_draft.json    -- /reason AI draft (reason.v1; human-review; no alert)

Each is validated against the REAL Pydantic models so a contract drift fails CI.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

import pytest

pytest.importorskip("pydantic")

FIX = Path(__file__).parent / "fixtures"

# App-compatible fields every /detect response must carry.
_DETECT_BASE = ("entities", "poses", "backend", "model", "tasks",
                "inference_ms", "img_w", "img_h")


def _load(name):
    return json.loads((FIX / name).read_text())


def test_detect_old_is_legacy_shape():
    body = _load("detect_old.json")
    for k in _DETECT_BASE:
        assert k in body, k
    # Legacy shape carries NO risk additions.
    assert "schema_version" not in body
    assert "risks" not in body
    # entities validate against the Entity model.
    from schema import Entity
    for e in body["entities"]:
        Entity(**e)


def test_detect_risk_shape_and_schema_version():
    body = _load("detect_risk.json")
    for k in _DETECT_BASE:
        assert k in body, k
    assert body["schema_version"] == "risk.v1"
    assert body["degraded"] is False and body["degradation_mode"] == "full"
    assert isinstance(body["risks"], list) and body["risks"]
    # risk items validate against the real RiskItem model.
    from risk.risk_schema import RiskItem
    for r in body["risks"]:
        item = RiskItem(**r)
        assert item.produced_by == "risk_engine"
        assert item.requires_human_review is False
    assert body["risk_engine"]["enabled"] is True


def test_detect_degraded_preserves_detection():
    body = _load("detect_degraded.json")
    for k in _DETECT_BASE:
        assert k in body, k                       # detection preserved
    assert body["schema_version"] == "risk.v1"
    assert body["degraded"] is True
    assert body["degradation_mode"] == "no_risk"
    assert body["warning"] and "risk_engine_error" in body["warning"]
    assert body["risk_engine"]["degraded"] is True


def test_reason_draft_is_ai_draft():
    body = _load("reason_draft.json")
    from risk.reason_schema import ReasonResponse
    parsed = ReasonResponse(**body)
    assert parsed.schema_version == "reason.v1"
    assert parsed.produced_by == "vlm_reasoner"
    assert parsed.requires_human_review is True
    assert parsed.should_alert is False
    for r in parsed.risks:
        assert r.requires_human_review is True
        assert r.should_alert is False


def test_all_new_shapes_have_schema_version():
    for name, version in (("detect_risk.json", "risk.v1"),
                          ("detect_degraded.json", "risk.v1"),
                          ("reason_draft.json", "reason.v1")):
        assert _load(name)["schema_version"] == version
