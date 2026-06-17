"""
agentic_cpu/action_log.py -- append-only audit log of agent actions.

Backends (CPU_AGENT_ACTION_LOG_BACKEND):
  * memory   -- bounded in-process deque. MVP default. NOT durable: lost on
                restart. Fine for a demo, not for production approval trails.
  * supabase / postgres -- documented external backends. Best-effort: if the
                client/connection is not configured at runtime we fall back to
                memory and log a warning (we never crash the request path and we
                never require a DB for tests).

No secrets/frames are ever logged here -- only the structured action record
(which itself carries no imagery).
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from typing import Any, Deque, Dict, List

from . import config

log = logging.getLogger("safelens-vision-worker.agentic.action_log")

_LOCK = threading.RLock()
_MEM: Deque[Dict[str, Any]] = deque(maxlen=2000)
_warned_backend = {"v": False}


def _now_ms() -> int:
    return int(time.time() * 1000)


def _metric(name: str) -> None:
    try:
        import worker_runtime as runtime
        runtime.inc(name)
    except Exception:  # noqa: BLE001
        pass


def append(record: Dict[str, Any]) -> Dict[str, Any]:
    """Append one action record. Returns the stored record. Never raises."""
    rec = dict(record or {})
    rec.setdefault("logged_at_ms", _now_ms())
    backend = config.action_log_backend()
    try:
        if backend in ("supabase", "postgres"):
            ok = _append_external(backend, rec)
            if not ok:
                if not _warned_backend["v"]:
                    log.warning("action_log backend '%s' unavailable; using memory "
                                "(not durable). Configure the backend for production.", backend)
                    _warned_backend["v"] = True
                _append_memory(rec)
        else:
            _append_memory(rec)
    except Exception as exc:  # noqa: BLE001 -- logging must never break a request
        log.warning("action_log append failed: %s", exc)
    _metric("cpu_agent_action_logged_total")
    return rec


def _append_memory(rec: Dict[str, Any]) -> None:
    with _LOCK:
        _MEM.append(rec)


def _append_external(backend: str, rec: Dict[str, Any]) -> bool:
    """Hook for supabase/postgres. Returns False (fall back to memory) until a
    client is wired at deploy time -- intentionally not requiring a DB driver in
    the image or tests. See docs/agentic_cpu_inside_runpod.md."""
    return False


def recent(limit: int = 50) -> List[Dict[str, Any]]:
    with _LOCK:
        items = list(_MEM)
    return items[-limit:]


def count() -> int:
    with _LOCK:
        return len(_MEM)


def reset() -> None:
    with _LOCK:
        _MEM.clear()
    _warned_backend["v"] = False
