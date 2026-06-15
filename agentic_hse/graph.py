"""LangGraph orchestration for the SafeLens agentic HSE layer.

Topology realizes the Preview -> Approval -> Execute -> Log control cycle:

    START -> setup -> observation -> risk_assessment -> audit -> training
          -> vision_improvement -> approval (interrupt)
                 approve -> execute -> log -> END
                 reject  -> log -> END
                 revise  -> revised preview -> approval (interrupt)

``langgraph`` is imported lazily inside ``build_graph`` so this module stays
importable on hosts where langgraph is absent (the pure ``route_after_assessment``
router and the node callables can still be imported and unit-tested). The
checkpointer (self-hosted Postgres, never Supabase) is injected by the caller;
see ``checkpointer/langgraph_checkpointer.py``.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from .approval import requires_approval
from .nodes import (
    run_audit_agent,
    run_observation_agent,
    run_risk_assessment_agent,
    run_setup_agent,
    run_training_agent,
    run_vision_improvement_agent,
)
from .state import AgenticHSEState


def _top_score(state: dict[str, Any]) -> int:
    return max((int(e.get("score", 0)) for e in state.get("events") or []), default=0)


def _log_entry(agent: str, status: str, summary: str, **extra: Any) -> dict[str, Any]:
    return {
        "agent": agent,
        "status": status,
        "summary": summary,
        "logged_at": datetime.now(timezone.utc).isoformat(),
        **extra,
    }


def route_after_assessment(state: dict[str, Any]) -> str:
    """Risk drafts always require approval; high scores must never bypass it."""
    if state.get("pending_approval"):
        return "approval"
    return "approval" if requires_approval(_top_score(state)) else "log"


def _approval_node(state: dict[str, Any]) -> dict[str, Any]:
    """HITL gate. Pauses the graph via ``interrupt()`` until a human resumes with
    ``Command(resume={"decision": "approve"|"reject"|"revise", ...})``. The full
    state is persisted by the checkpointer, so a pause survives shift handovers
    and process crashes."""
    from langgraph.types import interrupt
    request = state.get("pending_approval") or {"action_type": "review", "score": _top_score(state)}
    decision = interrupt(request)
    decision_data = decision if isinstance(decision, dict) else {"decision": str(decision)}
    verdict = decision_data.get("decision", "held")
    status = {
        "approve": "approved",
        "reject": "rejected",
        "revise": "pending_approval",
    }.get(verdict, "held")
    return {
        "approvals": (state.get("approvals") or []) + [{"request": request, "decision": decision_data}],
        "action_log": (state.get("action_log") or [])
        + [_log_entry(
            "approval",
            status,
            f"human decision recorded: {verdict}",
            approver_id=decision_data.get("approver_id"),
            notes=decision_data.get("notes", ""),
        )],
    }


def route_after_approval(state: dict[str, Any]) -> str:
    approvals = state.get("approvals") or []
    decision = (approvals[-1].get("decision") if approvals else {}) or {}
    verdict = decision.get("decision") if isinstance(decision, dict) else decision
    if verdict == "approve":
        return "execute"
    if verdict == "revise":
        return "revise"
    return "log"


def _revision_node(state: dict[str, Any]) -> dict[str, Any]:
    approvals = state.get("approvals") or []
    decision = (approvals[-1].get("decision") if approvals else {}) or {}
    revised = decision.get("revised_payload") if isinstance(decision, dict) else None
    if not isinstance(revised, dict):
        revised = {}
    draft = dict(state.get("risk_assessment") or {})
    draft.update(revised)
    score = int(draft.get("likelihood", 1)) * int(draft.get("severity", 1))
    draft["score"] = score
    from .approval import band_for_score, build_approval_request

    draft["matrix_band"] = band_for_score(score)
    draft["requires_human_approval"] = True
    return {
        "risk_assessment": draft,
        "pending_approval": build_approval_request("risk_assessment", draft, score),
        "action_log": (state.get("action_log") or [])
        + [_log_entry("revision", "preview", "revised risk assessment awaiting approval")],
    }


def _execute_node(state: dict[str, Any]) -> dict[str, Any]:
    approvals = state.get("approvals") or []
    last = approvals[-1] if approvals else {}
    decision = last.get("decision")
    verdict = decision.get("decision") if isinstance(decision, dict) else decision
    approved = verdict == "approve"
    return {
        "pending_approval": {},
        "action_log": (state.get("action_log") or [])
        + [_log_entry(
            "execute",
            "executed" if approved else "held",
            (
                "approved preview recorded for downstream execution"
                if approved
                else "actions held because approval was not granted"
            ),
        )],
    }


def _log_node(state: dict[str, Any]) -> dict[str, Any]:
    return {
        "action_log": (state.get("action_log") or [])
        + [_log_entry(
            "log",
            "logged",
            f"{len(state.get('events') or [])} event(s) logged; top score {_top_score(state)}",
        )]
    }


def build_graph(checkpointer: Any = None):
    """Compile the agentic HSE graph. ``checkpointer`` should be a self-hosted
    Postgres AsyncPostgresSaver (NOT Supabase). Raises a clear ImportError if
    langgraph is not installed."""
    try:
        from langgraph.graph import END, START, StateGraph
    except Exception as exc:  # noqa: BLE001
        raise ImportError(
            "langgraph is required to build the agentic HSE graph. "
            "Install: pip install langgraph langgraph-checkpoint-postgres"
        ) from exc

    g = StateGraph(AgenticHSEState)
    g.add_node("setup", run_setup_agent)
    g.add_node("observation", run_observation_agent)
    g.add_node("risk_assessment", run_risk_assessment_agent)
    g.add_node("audit", run_audit_agent)
    g.add_node("training", run_training_agent)
    g.add_node("vision_improvement", run_vision_improvement_agent)
    g.add_node("approval", _approval_node)
    g.add_node("revise", _revision_node)
    g.add_node("execute", _execute_node)
    g.add_node("log", _log_node)

    g.add_edge(START, "setup")
    g.add_edge("setup", "observation")
    g.add_edge("observation", "risk_assessment")
    g.add_edge("risk_assessment", "audit")
    g.add_edge("audit", "training")
    g.add_edge("training", "vision_improvement")
    g.add_conditional_edges(
        "vision_improvement",
        route_after_assessment,
        {"approval": "approval", "log": "log"},
    )
    g.add_conditional_edges(
        "approval",
        route_after_approval,
        {"execute": "execute", "revise": "revise", "log": "log"},
    )
    g.add_edge("revise", "approval")
    g.add_edge("execute", "log")
    g.add_edge("log", END)

    return g.compile(checkpointer=checkpointer)
