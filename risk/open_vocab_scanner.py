"""
risk/open_vocab_scanner.py -- orchestrator for open-vocabulary scanners.

Thin wrapper that selects the backend (GroundingDINO today; future scanners
behind the same API), enforces a periodic/on-demand throttle (never per-frame),
and guarantees candidate-only output. Disabled by default.

Allowed uses (never to approve an HSE risk): user-requested unknown-object
scan, low-confidence/high-risk scenes, rare-object discovery, dataset-candidate
creation, periodic scanner mode.
"""

from __future__ import annotations

import os
import threading
import time
from typing import Any, Dict, Optional

from . import grounding_dino_scanner as gdino
from .reason_schema import OpenVocabResult

_LOCK = threading.RLock()
_LAST_SCAN_MS: Dict[str, int] = {}


def enabled() -> bool:
    return os.getenv("OPEN_VOCAB_SCANNER_ENABLED", "false").strip().lower() in (
        "1", "true", "yes", "on")


def backend() -> str:
    return os.getenv("OPEN_VOCAB_SCANNER_MODE", "grounding_dino").strip().lower()


def _interval_ms() -> int:
    try:
        return int(os.getenv("OPEN_VOCAB_SCAN_INTERVAL_MS", "30000"))
    except (TypeError, ValueError):
        return 30000


def _now_ms() -> int:
    return int(time.time() * 1000)


def scan(frame_b64: Optional[str], *, prompt: Optional[str] = None,
         session_id: Optional[str] = None, frame_id: Optional[str] = None,
         entities: Optional[list] = None, force: bool = False) -> Dict[str, Any]:
    """Run an open-vocab scan (throttled per session unless force=True).

    force=True is for explicit user-requested scans (bypasses the interval, not
    the enabled flag). `entities` lets the privacy egress guard blur persons.
    Always returns a candidate-only result dict.
    """
    if not enabled():
        return OpenVocabResult(status="disabled", session_id=session_id,
                               frame_id=frame_id).enforce_candidate_contract().model_dump()
    sid = session_id or "__default__"
    now = _now_ms()
    if not force:
        with _LOCK:
            last = _LAST_SCAN_MS.get(sid, 0)
            if now - last < _interval_ms():
                return OpenVocabResult(status="throttled", session_id=session_id,
                                       frame_id=frame_id).enforce_candidate_contract().model_dump()
            _LAST_SCAN_MS[sid] = now
    else:
        with _LOCK:
            _LAST_SCAN_MS[sid] = now

    if backend() == "grounding_dino":
        return gdino.scan(frame_b64, prompt=prompt, session_id=session_id,
                          frame_id=frame_id, entities=entities)
    return OpenVocabResult(status="unavailable", session_id=session_id, frame_id=frame_id,
                           error=f"unknown backend {backend()}").enforce_candidate_contract().model_dump()


def config() -> Dict[str, Any]:
    return {
        "enabled": enabled(),
        "backend": backend(),
        "scan_interval_ms": _interval_ms(),
        "available": gdino.available() if backend() == "grounding_dino" else False,
        "model_id": gdino._model_id() if backend() == "grounding_dino" else None,
        "candidate_only": True,
        "note": "Open-vocab candidates are low-authority, require human review, "
                "and never trigger official HSE alerts directly.",
    }


def reset() -> None:
    with _LOCK:
        _LAST_SCAN_MS.clear()
    gdino.reset()
