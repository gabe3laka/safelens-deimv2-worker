"""
shared/ -- cross-layer schemas, prompts, and wire contracts for the SafeLens worker.

Imported by BOTH the GPU perception layer (temporal_reasoning) and the CPU
agentic layer (agentic_cpu). Deliberately import-light: pydantic + stdlib only,
NO torch / cv2 / ultralytics / transformers. This keeps the CPU agent free of
GPU deps (enforced by tests/test_agent_import_guard.py) and means importing
`shared` can never break server boot.
"""

from __future__ import annotations

__all__ = ["schemas"]
