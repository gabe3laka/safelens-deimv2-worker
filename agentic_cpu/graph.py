"""
agentic_cpu/graph.py -- the agent orchestration graph (dependency-free).

Describes how the agents compose (company setup -> observation -> assessment ->
CAPA -> training) and which steps are approval-gated. A LangGraph/LangChain
backend can replace this behind the same shape later; the CHECKPOINTER_BACKEND
env documents where long-running graph state would persist for production
(memory for MVP, postgres for durability). No GPU deps.
"""

from __future__ import annotations

from typing import Any, Dict, List

from . import agents, config
from .schemas import APPROVAL_REQUIRED_ACTIONS

# Conceptual graph edges (for docs + /debug/state). Each node maps to an agent
# action_type; approval-gated nodes must pass through approvals.execute.
NODES: List[Dict[str, Any]] = [
    {"node": "company_setup", "action_type": "company_profile_extract", "approval": False},
    {"node": "safety_observation", "action_type": "incident_create", "approval": True},
    {"node": "risk_assessment", "action_type": "risk_assessment_approve", "approval": True},
    {"node": "audit", "action_type": "audit_finding_send", "approval": True},
    {"node": "capa", "action_type": "capa_create", "approval": True},
    {"node": "training", "action_type": "training_record_create", "approval": True},
    {"node": "vision_improvement", "action_type": "dataset_candidate_approve", "approval": True},
]


def plan() -> List[Dict[str, Any]]:
    return list(NODES)


def checkpointer() -> Dict[str, Any]:
    return {"backend": config.checkpointer_backend(),
            "durable": config.checkpointer_backend() != "memory",
            "note": "memory is not durable across worker restarts (MVP)."}


def known_action_types() -> List[str]:
    return [n["action_type"] for n in NODES]


def is_registered(action_type: str) -> bool:
    try:
        return action_type in agents._registry() or action_type in APPROVAL_REQUIRED_ACTIONS
    except Exception:  # noqa: BLE001
        return False


def snapshot() -> Dict[str, Any]:
    return {"nodes": [n["node"] for n in NODES], "checkpointer": checkpointer()}
