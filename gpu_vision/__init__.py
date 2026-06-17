"""
gpu_vision/ -- thin wiring around the GPU perception layer's concurrency.

The detector + VLM reasoner already live in vision_backend / risk.vlm_reasoner.
This package does NOT re-implement them; it adds the *bounded GPU concurrency*
the single-worker architecture needs so GPU reasoner jobs and CPU agent jobs have
SEPARATE limits and the CPU layer can back off under GPU pressure.

Import-light (stdlib only); importing it never pulls torch.
"""

from __future__ import annotations

from .concurrency import (
    gpu_busy_ratio,
    gpu_reasoner_max_inflight,
    gpu_reasoner_slot,
    inflight,
    snapshot,
)

__all__ = [
    "gpu_busy_ratio",
    "gpu_reasoner_max_inflight",
    "gpu_reasoner_slot",
    "inflight",
    "snapshot",
]
