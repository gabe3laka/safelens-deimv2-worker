"""
risk/ -- deterministic, additive risk-aware perception for the SafeLens worker.

Public API (import-light; importing this package pulls no torch/cv2):

    from risk import attach_risk, evaluate, config, enabled, SCHEMA_VERSION

`attach_risk(resp_dict, session_id=...)` is the one-liner both /detect and
/ws/vision call to merge the additive risk block when RISK_ENGINE_ENABLED is on.
It never raises; when disabled the response is byte-for-byte the legacy shape.

The deterministic engine is the safety signal. The event-driven VLM reasoner and
GroundingDINO scanner are a SEPARATE later path and only enrich `scene_risks`
as human-review AI drafts -- they are intentionally not part of this package.
"""

from __future__ import annotations

from .risk_engine import attach_risk, config, enabled, evaluate
from .risk_schema import SCHEMA_VERSION

__all__ = ["attach_risk", "config", "enabled", "evaluate", "SCHEMA_VERSION"]
