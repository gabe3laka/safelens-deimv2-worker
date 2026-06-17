"""
gpu_vision/concurrency.py -- bounded GPU reasoner concurrency + a GPU-pressure
signal, separate from the CPU agent's limits.

Why: in a single worker the latency-sensitive GPU path and the human-paced CPU
agent path must not contend without bound. GPU reasoner jobs run through a
bounded semaphore (GPU_REASONER_MAX_INFLIGHT); a job that cannot get a slot is
DROPPED (counted), never queued unboundedly, so /detect never backs up. The CPU
agent reads gpu_busy_ratio() and degrades first when the GPU is busy
(CPU_AGENT_DISABLE_ON_GPU_PRESSURE).

Pure stdlib (no torch) so importing it can never break boot. An optional,
best-effort CUDA memory probe is used ONLY if torch is already importable.
"""

from __future__ import annotations

import os
import threading
from contextlib import contextmanager
from typing import Any, Dict, Iterator

try:
    import worker_runtime as runtime
except Exception:  # noqa: BLE001 -- metrics are best-effort
    runtime = None  # type: ignore

_LOCK = threading.Lock()
_INFLIGHT = {"n": 0}
_DROPPED = {"n": 0}


def gpu_reasoner_max_inflight() -> int:
    try:
        return max(1, int(os.getenv("GPU_REASONER_MAX_INFLIGHT", "1")))
    except (TypeError, ValueError):
        return 1


def inflight() -> int:
    with _LOCK:
        return _INFLIGHT["n"]


def _set_gauge(name: str, value: float) -> None:
    if runtime is not None:
        try:
            runtime.set_gauge(name, value)
        except Exception:  # noqa: BLE001
            pass


def _inc(name: str) -> None:
    if runtime is not None:
        try:
            runtime.inc(name)
        except Exception:  # noqa: BLE001
            pass


@contextmanager
def gpu_reasoner_slot() -> Iterator[bool]:
    """Acquire a GPU reasoner slot if one is free.

    Yields True if acquired (caller should run the GPU job), False if the worker
    is already at GPU_REASONER_MAX_INFLIGHT (caller should DROP the job, not
    queue it). Never blocks.
    """
    acquired = False
    with _LOCK:
        if _INFLIGHT["n"] < gpu_reasoner_max_inflight():
            _INFLIGHT["n"] += 1
            acquired = True
        else:
            _DROPPED["n"] += 1
    if acquired:
        _set_gauge("gpu_reasoner_jobs_inflight", float(_INFLIGHT["n"]))
    else:
        _inc("gpu_reasoner_jobs_dropped_total")
    try:
        yield acquired
    finally:
        if acquired:
            with _LOCK:
                _INFLIGHT["n"] = max(0, _INFLIGHT["n"] - 1)
            _set_gauge("gpu_reasoner_jobs_inflight", float(_INFLIGHT["n"]))


def _cuda_mem_busy_ratio() -> float:
    """Best-effort CUDA memory pressure (0..1). 0.0 if torch/CUDA unavailable.

    Only used if torch is ALREADY importable; never imports lazily in a way that
    could be slow on the hot path (the import is cheap once torch is loaded).
    """
    try:
        import torch  # noqa: PLC0415
        if not torch.cuda.is_available():
            return 0.0
        free, total = torch.cuda.mem_get_info(0)
        if total <= 0:
            return 0.0
        return max(0.0, min(1.0, 1.0 - (free / total)))
    except Exception:  # noqa: BLE001
        return 0.0


def gpu_busy_ratio() -> float:
    """A 0..1 GPU-pressure estimate: max(inflight saturation, CUDA mem usage).

    The CPU agent uses this to decide whether to degrade
    (CPU_AGENT_DISABLE_ON_GPU_PRESSURE / CPU_AGENT_MAX_GPU_BUSY_RATIO).
    """
    with _LOCK:
        sat = _INFLIGHT["n"] / float(max(1, gpu_reasoner_max_inflight()))
    ratio = max(min(1.0, sat), _cuda_mem_busy_ratio())
    _set_gauge("gpu_busy_ratio", float(round(ratio, 4)))
    return ratio


def snapshot() -> Dict[str, Any]:
    with _LOCK:
        n, dropped = _INFLIGHT["n"], _DROPPED["n"]
    return {
        "gpu_reasoner_max_inflight": gpu_reasoner_max_inflight(),
        "gpu_reasoner_jobs_inflight": n,
        "gpu_reasoner_jobs_dropped_total": dropped,
        "gpu_busy_ratio": round(gpu_busy_ratio(), 4),
    }


def reset() -> None:
    """Test helper."""
    with _LOCK:
        _INFLIGHT["n"] = 0
        _DROPPED["n"] = 0
