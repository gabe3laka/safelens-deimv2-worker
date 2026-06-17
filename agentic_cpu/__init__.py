"""
agentic_cpu/ -- CPU-only agentic orchestration mounted inside the same RunPod
worker, under /agent/*.

Design rules (single-worker MVP):
  * Runs in the SAME FastAPI app + process as the GPU vision routes, but on a
    SEPARATE bounded job queue with its own concurrency limit -- it can never
    block /detect.
  * CPU-only: imports NO torch / cv2 / ultralytics / transformers / vision
    loaders (enforced by tests/test_agent_import_guard.py). It consumes the
    structured detection JSON the app already has.
  * Mock by default: no real LLM key or DB required (CPU_AGENT_MODE=mock).
  * Human-in-the-loop: serious actions are DRAFTS (pending_approval) and only
    finalize via an approved execute call. Approvals/logs should be persisted
    externally for production (memory backend is not durable).

Public API:
    from agentic_cpu import get_router, status_snapshot, ready, enabled
"""

from __future__ import annotations

from typing import Any, Dict

from . import config


def enabled() -> bool:
    return config.agentic_enabled()


def ready() -> bool:
    """Agent layer readiness (independent of GPU vision readiness)."""
    from . import llm
    return config.agentic_enabled() and llm.available() and not config.under_gpu_pressure()


def get_router():
    """Return the FastAPI APIRouter (lazy import so importing this package does
    not require FastAPI to be installed, e.g. in pure-unit tests)."""
    from .router import router
    return router


def status_snapshot() -> Dict[str, Any]:
    """Non-sensitive snapshot for GET /debug/state and /ready."""
    from . import action_log, graph, jobs
    snap = config.snapshot()
    snap.update({
        "ready": ready(),
        "jobs_inflight": jobs.queue_depth(),
        "queue_depth": jobs.queue_depth(),
        "action_log_count": action_log.count(),
        "graph": graph.snapshot(),
    })
    return snap


def reset_all() -> None:
    """Test helper: clear jobs, approvals, and the action log."""
    from . import action_log, approvals, jobs
    jobs.reset()
    approvals.reset()
    action_log.reset()


__all__ = ["enabled", "ready", "get_router", "status_snapshot", "reset_all"]
