"""
agentic_cpu/config.py -- env-driven configuration for the CPU agentic layer.

All knobs are env vars (12-factor); none require a real LLM key or DB. Defaults
favour SAFE + LOCAL: mock LLM, memory action log + checkpointer, approval
required. AGENTIC_CPU_ENABLED defaults False in code (additive / off) and is set
True in the Dockerfile for production.
"""

from __future__ import annotations

import os
from typing import Any, Dict


def _bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def agentic_enabled() -> bool:
    return _bool("AGENTIC_CPU_ENABLED", False)


def mode() -> str:
    return os.getenv("CPU_AGENT_MODE", "mock").strip().lower()


def max_inflight() -> int:
    return max(1, _int("CPU_AGENT_MAX_INFLIGHT", 2))


def queue_max() -> int:
    return max(1, _int("CPU_AGENT_QUEUE_MAX", 16))


def job_timeout_ms() -> int:
    return max(1, _int("CPU_AGENT_JOB_TIMEOUT_MS", 30000))


def require_approval() -> bool:
    return _bool("CPU_AGENT_REQUIRE_APPROVAL", True)


def action_log_backend() -> str:
    return os.getenv("CPU_AGENT_ACTION_LOG_BACKEND", "memory").strip().lower()


def checkpointer_backend() -> str:
    return os.getenv("CHECKPOINTER_BACKEND", "memory").strip().lower()


def llm_provider() -> str:
    return os.getenv("CPU_AGENT_LLM_PROVIDER", "mock").strip().lower()


def llm_model() -> str:
    return os.getenv("CPU_AGENT_LLM_MODEL", "mock").strip()


def disable_on_gpu_pressure() -> bool:
    return _bool("CPU_AGENT_DISABLE_ON_GPU_PRESSURE", True)


def max_gpu_busy_ratio() -> float:
    return _float("CPU_AGENT_MAX_GPU_BUSY_RATIO", 0.85)


def under_gpu_pressure() -> bool:
    """True if the CPU agent should degrade so the GPU path keeps priority.

    Lazy-imports gpu_vision so importing agentic_cpu stays GPU-dep-free (the
    import guard checks this). gpu_vision itself never imports torch at module
    load, so this never pulls a heavy dep onto the agent path.
    """
    if not disable_on_gpu_pressure():
        return False
    try:
        import gpu_vision  # noqa: PLC0415
        return gpu_vision.gpu_busy_ratio() > max_gpu_busy_ratio()
    except Exception:  # noqa: BLE001
        return False


def snapshot() -> Dict[str, Any]:
    """Non-sensitive config snapshot for /debug/state and /agent/ready."""
    return {
        "enabled": agentic_enabled(),
        "mode": mode(),
        "max_inflight": max_inflight(),
        "queue_max": queue_max(),
        "job_timeout_ms": job_timeout_ms(),
        "require_approval": require_approval(),
        "action_log_backend": action_log_backend(),
        "checkpointer_backend": checkpointer_backend(),
        "llm_provider": llm_provider(),
        "llm_model": llm_model(),
        "disable_on_gpu_pressure": disable_on_gpu_pressure(),
        "max_gpu_busy_ratio": max_gpu_busy_ratio(),
    }
