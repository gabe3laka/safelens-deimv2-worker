"""Self-hosted Postgres checkpointer for the agentic HSE LangGraph.

Uses langgraph's ``AsyncPostgresSaver`` against a SELF-HOSTED Postgres (local on
the Hostinger VPS or co-located on RunPod) -- NOT the SafeLens Supabase project,
which OpenClaw cannot access. Migrating checkpoints into Supabase later is a
user-run option (see ``checkpointer/README.md``).

Important: ``AsyncPostgresSaver.from_conn_string()`` returns an *async context
manager*, not an awaitable saver -- so the saver must be used inside ``async
with`` (preferred, scoped to the app lifetime) or built from a long-lived
connection pool. Both patterns are provided.
"""
from __future__ import annotations

import contextlib
import os
from typing import Any, AsyncIterator


def _dsn(dsn: str | None = None) -> str:
    dsn = dsn or os.environ.get("LANGGRAPH_POSTGRES_DSN", "")
    if not dsn:
        raise RuntimeError(
            "LANGGRAPH_POSTGRES_DSN is not set (see checkpointer/.env.example). "
            "Use a plain postgresql:// DSN -- psycopg3 rejects the +asyncpg suffix."
        )
    return dsn


@contextlib.asynccontextmanager
async def open_checkpointer(dsn: str | None = None) -> AsyncIterator[Any]:
    """Preferred: scope the saver to the application lifetime.

        from agentic_hse.graph import build_graph
        async with open_checkpointer() as saver:
            graph = build_graph(checkpointer=saver)
            await graph.ainvoke(state, config={"configurable": {"thread_id": "site-1"}})
    """
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
    async with AsyncPostgresSaver.from_conn_string(_dsn(dsn)) as saver:
        await saver.setup()  # idempotent: creates checkpoint tables if absent
        yield saver


async def build_checkpointer_from_pool(dsn: str | None = None):
    """Alternative: build from a long-lived async connection pool. The caller owns
    the pool lifetime and must ``await pool.close()`` on shutdown.

    Returns ``(saver, pool)``."""
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
    from psycopg_pool import AsyncConnectionPool

    pool = AsyncConnectionPool(
        conninfo=_dsn(dsn),
        max_size=10,
        open=False,
        kwargs={"autocommit": True, "prepare_threshold": 0},
    )
    await pool.open()
    saver = AsyncPostgresSaver(pool)
    await saver.setup()
    return saver, pool
