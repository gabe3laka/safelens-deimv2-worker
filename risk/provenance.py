"""
risk/provenance.py -- stamp every risk item with who/what/when produced it.

Provenance is mandatory for a regulated HSE signal: each risk records the
producer (deterministic engine vs VLM draft), the model/rule version, the
rule_id that fired, and a timestamp. This supports audit trails and keeps the
deterministic engine distinguishable from AI drafts that require human review.
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict


def model_version() -> str:
    return os.getenv("RISK_MODEL_VERSION", "risk_engine.v1")


def enabled() -> bool:
    return os.getenv("RISK_PROVENANCE_ENABLED", "true").strip().lower() in (
        "1", "true", "yes", "on")


def now_ms() -> int:
    return int(time.time() * 1000)


def stamp(item: Dict[str, Any], *, rule_id: str, ts_ms: int | None = None,
          produced_by: str = "risk_engine",
          requires_human_review: bool = False) -> Dict[str, Any]:
    """Attach provenance fields to a risk dict (in place) and return it.

    The deterministic engine is the safety signal -> requires_human_review
    defaults False. VLM/open-vocab producers pass requires_human_review=True.
    """
    item["produced_by"] = produced_by
    item["rule_id"] = rule_id
    item["model_version"] = model_version()
    item["requires_human_review"] = requires_human_review
    item["timestamp_ms"] = ts_ms if ts_ms is not None else now_ms()
    return item
