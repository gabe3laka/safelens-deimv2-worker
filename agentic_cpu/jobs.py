"""
agentic_cpu/jobs.py -- bounded background job queue for CPU agent work.

Guarantees the single-worker architecture needs:
  * CPU agent jobs run on a bounded pool (CPU_AGENT_MAX_INFLIGHT) -- SEPARATE
    from the GPU reasoner's limits, so agent work cannot starve /detect.
  * Total queued+running is capped at CPU_AGENT_QUEUE_MAX; submitting past that
    raises QueueFull (the router returns HTTP 429 / structured queue_full).
  * Each job has a hard timeout (CPU_AGENT_JOB_TIMEOUT_MS) -> structured error,
    never a hang on the request path.
  * Results are stored (memory MVP) and fetchable via GET /agent/jobs/{job_id}.

No GPU deps; pure stdlib + worker_runtime metrics (best-effort).
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, Optional

from . import config

log = logging.getLogger("safelens-vision-worker.agentic.jobs")


class QueueFull(Exception):
    """Raised by submit() when queued+running has reached CPU_AGENT_QUEUE_MAX."""


_LOCK = threading.RLock()
_JOBS: Dict[str, Dict[str, Any]] = {}
_PENDING = {"n": 0}          # queued + running
_EXECUTOR: Optional[ThreadPoolExecutor] = None
_JOBS_MAX = 1000


def _now_ms() -> int:
    return int(time.time() * 1000)


def _executor() -> ThreadPoolExecutor:
    global _EXECUTOR
    if _EXECUTOR is None:
        _EXECUTOR = ThreadPoolExecutor(max_workers=config.max_inflight(),
                                       thread_name_prefix="cpu-agent")
    return _EXECUTOR


def _gauge(name: str, value: float) -> None:
    try:
        import worker_runtime as runtime
        runtime.set_gauge(name, value)
    except Exception:  # noqa: BLE001
        pass


def _inc(name: str) -> None:
    try:
        import worker_runtime as runtime
        runtime.inc(name)
    except Exception:  # noqa: BLE001
        pass


def _publish_depth() -> None:
    with _LOCK:
        n = _PENDING["n"]
    _gauge("cpu_agent_jobs_inflight", float(n))
    _gauge("cpu_agent_queue_depth", float(n))


def queue_depth() -> int:
    with _LOCK:
        return _PENDING["n"]


def _run_with_timeout(fn: Callable[..., Any], args, kwargs, timeout_s: float):
    """Run fn in a watched thread; return (value, error). On timeout the thread
    is abandoned (daemon) and the job is marked timed out -- threads cannot be
    force-killed in CPython; this is a documented MVP limitation."""
    box: Dict[str, Any] = {}

    def _target():
        try:
            box["value"] = fn(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001
            box["error"] = f"{type(exc).__name__}: {exc}"

    th = threading.Thread(target=_target, daemon=True)
    th.start()
    th.join(timeout_s)
    if th.is_alive():
        return None, "job_timeout"
    if "error" in box:
        return None, box["error"]
    return box.get("value"), None


def _worker(job_id: str, fn: Callable[..., Any], args, kwargs) -> None:
    with _LOCK:
        job = _JOBS.get(job_id)
        if job:
            job["status"] = "running"
            job["updated_at_ms"] = _now_ms()
    _publish_depth()
    timeout_s = max(0.01, config.job_timeout_ms() / 1000.0)
    value, error = _run_with_timeout(fn, args, kwargs, timeout_s)
    with _LOCK:
        job = _JOBS.get(job_id)
        if job:
            if error:
                job["status"] = "error"
                job["error"] = error
            else:
                job["status"] = "done"
                job["result"] = value
            job["updated_at_ms"] = _now_ms()
        _PENDING["n"] = max(0, _PENDING["n"] - 1)
    _publish_depth()
    if error:
        _inc("cpu_agent_jobs_failed_total")
    else:
        _inc("cpu_agent_jobs_completed_total")


def submit(action_type: str, fn: Callable[..., Any], *args, **kwargs) -> str:
    """Submit a job; returns job_id. Raises QueueFull at capacity. Never blocks."""
    with _LOCK:
        if _PENDING["n"] >= config.queue_max():
            _inc("cpu_agent_jobs_rejected_total")
            raise QueueFull(f"queue full ({_PENDING['n']}/{config.queue_max()})")
        job_id = "job_" + uuid.uuid4().hex[:16]
        _JOBS[job_id] = {
            "job_id": job_id, "action_type": action_type, "status": "queued",
            "result": None, "error": None,
            "created_at_ms": _now_ms(), "updated_at_ms": _now_ms(),
        }
        _PENDING["n"] += 1
        # bound the job store
        while len(_JOBS) > _JOBS_MAX:
            oldest = min(_JOBS.items(), key=lambda kv: kv[1]["created_at_ms"])[0]
            if oldest == job_id:
                break
            _JOBS.pop(oldest, None)
    _publish_depth()
    try:
        _executor().submit(_worker, job_id, fn, args, kwargs)
    except Exception as exc:  # noqa: BLE001
        with _LOCK:
            _PENDING["n"] = max(0, _PENDING["n"] - 1)
            j = _JOBS.get(job_id)
            if j:
                j["status"] = "error"
                j["error"] = f"submit_failed: {exc}"
        _publish_depth()
        raise QueueFull(f"executor refused job: {exc}") from exc
    return job_id


def submit_and_wait(action_type: str, fn: Callable[..., Any], *args,
                    wait_ms: Optional[int] = None, **kwargs) -> str:
    """Submit then wait up to wait_ms for completion (bounded). Returns job_id.

    For fast mock jobs the result is ready inline; the router returns it 200.
    For long jobs the router returns 202 + job_id and the caller polls
    GET /agent/jobs/{job_id}. Raises QueueFull at capacity.
    """
    job_id = submit(action_type, fn, *args, **kwargs)
    budget = wait_ms if wait_ms is not None else min(2000, config.job_timeout_ms())
    deadline = time.monotonic() + max(0.0, budget / 1000.0)
    while time.monotonic() < deadline:
        with _LOCK:
            status = _JOBS.get(job_id, {}).get("status")
        if status in ("done", "error"):
            break
        time.sleep(0.01)
    return job_id


def get(job_id: str) -> Optional[Dict[str, Any]]:
    with _LOCK:
        job = _JOBS.get(job_id)
        return dict(job) if job else None


def reset() -> None:
    """Test helper: clear jobs + pending counter (does not kill live threads)."""
    with _LOCK:
        _JOBS.clear()
        _PENDING["n"] = 0
    _publish_depth()
