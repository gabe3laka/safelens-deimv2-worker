"""Thin client for the RunPod Senior-QHSE-Manager reasoning engine.

httpx is imported lazily so this module stays importable on hosts where the
reasoning deps are not installed; the calling node degrades to a detector-only
event when the service is unreachable.
"""
from __future__ import annotations

from typing import Any


async def reason_over_hazard(
    reasoning_url: str,
    payload: dict[str, Any],
    *,
    timeout: float = 30.0,
) -> dict[str, Any]:
    """POST detections + scene context to the reasoning engine; return typed JSON."""
    import httpx  # lazy: only needed when a reasoning call actually fires
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(f"{reasoning_url.rstrip('/')}/reason", json=payload)
        resp.raise_for_status()
        from .models import ReasoningRecord
        return ReasoningRecord.model_validate(resp.json()).model_dump()
