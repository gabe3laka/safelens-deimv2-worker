"""Application-lifetime LangGraph runtime with optional durable checkpointing."""
from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from typing import Any


@dataclass
class GraphRuntime:
    graph: Any
    durable: bool
    backend: str
    pool: Any = None

    async def close(self) -> None:
        if self.pool is not None:
            await self.pool.close()


_runtime: GraphRuntime | None = None
_runtime_lock = asyncio.Lock()


async def get_runtime() -> GraphRuntime:
    global _runtime
    if _runtime is not None:
        return _runtime
    async with _runtime_lock:
        if _runtime is not None:
            return _runtime

        from .graph import build_graph

        dsn = os.getenv("LANGGRAPH_POSTGRES_DSN", "").strip()
        if dsn:
            from checkpointer.langgraph_checkpointer import build_checkpointer_from_pool

            saver, pool = await build_checkpointer_from_pool(dsn)
            _runtime = GraphRuntime(
                graph=build_graph(checkpointer=saver),
                durable=True,
                backend="postgres",
                pool=pool,
            )
        else:
            from langgraph.checkpoint.memory import InMemorySaver

            _runtime = GraphRuntime(
                graph=build_graph(checkpointer=InMemorySaver()),
                durable=False,
                backend="memory",
            )
        return _runtime


async def close_runtime() -> None:
    global _runtime
    if _runtime is not None:
        await _runtime.close()
        _runtime = None


def graph_config(thread_id: str) -> dict[str, dict[str, str]]:
    return {"configurable": {"thread_id": thread_id}}


def summarize_result(result: Any) -> dict[str, Any]:
    if not isinstance(result, dict):
        return {"state": {}, "status": "unknown", "interrupts": []}
    interrupts = result.get("__interrupt__") or []
    state = {key: value for key, value in result.items() if key != "__interrupt__"}
    return {
        "state": state,
        "status": "awaiting_approval" if interrupts else "completed",
        "interrupts": [
            getattr(item, "value", item) for item in interrupts
        ],
    }
