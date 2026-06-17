"""
agentic_cpu/tools/ -- pure-Python tools the agents call.

Every tool consumes STRUCTURED JSON (detection results, risk blocks, profile
text) and returns structured JSON. No tool imports a GPU loader, torch, cv2, or
transformers -- if an agent needs vision data it receives the detection JSON the
app already has (enforced by tests/test_agent_import_guard.py).
"""

from __future__ import annotations

__all__ = ["vision_tools", "document_tools", "risk_tools",
           "audit_tools", "capa_tools", "training_tools"]
