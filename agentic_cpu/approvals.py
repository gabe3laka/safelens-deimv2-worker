"""
agentic_cpu/approvals.py -- the human-approval gate.

Flow: Preview -> Human approval -> Execute -> Log. The CPU agent NEVER finalizes
a serious record on its own. execute() rejects any approval-required action that
was not explicitly approved (returns {"ok": false, "error": "approval_required"}).
"""

from __future__ import annotations

import threading
import time
from typing import Any, Dict, Optional

from . import action_log, agents, config
from .schemas import APPROVAL_REQUIRED_ACTIONS

_LOCK = threading.RLock()
_STORE: Dict[str, Dict[str, Any]] = {}
_MAX = 2000


def _now_ms() -> int:
    return int(time.time() * 1000)


def _metric(name: str) -> None:
    try:
        import worker_runtime as runtime
        runtime.inc(name)
    except Exception:  # noqa: BLE001
        pass


def register(action: Dict[str, Any]) -> Dict[str, Any]:
    """Store a draft so it can later be approved/executed by action_id."""
    with _LOCK:
        _STORE[action["action_id"]] = action
        while len(_STORE) > _MAX:
            oldest = min(_STORE.items(), key=lambda kv: kv[1].get("created_at_ms", 0))[0]
            _STORE.pop(oldest, None)
    if action.get("requires_human_approval"):
        _metric("cpu_agent_approval_required_total")
    return action


def get(action_id: str) -> Optional[Dict[str, Any]]:
    with _LOCK:
        a = _STORE.get(action_id)
        return dict(a) if a else None


def preview(action_type: str, payload: Dict[str, Any],
            source: Dict[str, Any]) -> Dict[str, Any]:
    """Produce + register a draft for review (does not execute)."""
    req = {"detection_context": payload.get("detection_context", payload),
           "company_profile": payload.get("company_profile", {}),
           "notes": payload.get("notes"), "payload": payload, "source": source}
    action = agents.dispatch(action_type, req)
    return register(action)


def _requires_approval(action: Dict[str, Any]) -> bool:
    if not config.require_approval():
        return False
    return (action.get("action_type") in APPROVAL_REQUIRED_ACTIONS
            or bool(action.get("requires_human_approval")))


def execute(action_id: str, approved: bool, approved_by: Optional[str]) -> Dict[str, Any]:
    """Finalize an action IFF approved. Rejects unapproved approval-required actions."""
    with _LOCK:
        action = _STORE.get(action_id)
        if action is None:
            return {"ok": False, "error": "action_not_found", "action_id": action_id}
        if _requires_approval(action) and not approved:
            _metric("cpu_agent_approval_required_total")
            return {"ok": False, "error": "approval_required",
                    "action_id": action_id, "status": action.get("status")}
        if _requires_approval(action) and not approved_by:
            return {"ok": False, "error": "approver_required",
                    "action_id": action_id, "status": action.get("status")}
        action["status"] = "executed"
        action["executed_at_ms"] = _now_ms()
        action["approved_by"] = approved_by
        action["result"] = {"finalized": True, "action_type": action.get("action_type")}
        stored = dict(action)
    action_log.append({
        "event": "action_executed",
        "action_id": action_id,
        "action_type": stored.get("action_type"),
        "approved_by": approved_by,
        "source": stored.get("source"),
        "ts_ms": _now_ms(),
    })
    _metric("cpu_agent_actions_executed_total")
    return {"ok": True, "action_id": action_id, "status": "executed", "action": stored}


def reset() -> None:
    with _LOCK:
        _STORE.clear()
