try:
    from langgraph.graph import StateGraph, END
except Exception:
    StateGraph = None
    END = 'END'

from .state import AgenticHSEState


def build_graph():
    if StateGraph is None:
        return None
    graph = StateGraph(AgenticHSEState)
    return graph.compile()
