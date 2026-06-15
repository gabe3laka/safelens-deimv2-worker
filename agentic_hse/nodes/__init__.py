"""LangGraph node callables for the six SafeLens HSE agents.

Each node takes the ``AgenticHSEState`` dict and returns a partial-state update.
Nodes import only the pure ``approval`` helpers at module load (and lazily import
the reasoning client) so the graph module stays importable without heavy deps.
"""
from .audit_agent import run_audit_agent
from .observation_agent import run_observation_agent
from .risk_assessment_agent import run_risk_assessment_agent
from .setup_agent import run_setup_agent
from .training_agent import run_training_agent
from .vision_improvement_agent import run_vision_improvement_agent

__all__ = [
    "run_setup_agent",
    "run_observation_agent",
    "run_risk_assessment_agent",
    "run_audit_agent",
    "run_training_agent",
    "run_vision_improvement_agent",
]
