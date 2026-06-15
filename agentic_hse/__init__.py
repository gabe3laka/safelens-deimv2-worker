"""SafeLens agentic HSE layer (LangGraph orchestration + Pydantic AI typed output).

Kept import-light: only the dependency-free ``approval`` helpers are re-exported
at package load so importing ``agentic_hse`` never pulls in pydantic/langgraph.
Import ``agentic_hse.graph.build_graph`` or ``agentic_hse.models`` explicitly when
those heavier deps are available.
"""
from .approval import band_for_score, requires_approval, should_halt

__all__ = ["requires_approval", "band_for_score", "should_halt"]
