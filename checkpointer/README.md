# LangGraph checkpointer (self-hosted Postgres)

The agentic HSE graph persists its state with LangGraph's `AsyncPostgresSaver`
so HITL approval pauses (`interrupt()`) survive shift handovers and process
crashes. Per the build prompt, the checkpointer runs on a **self-hosted
Postgres** — local on the Hostinger VPS or co-located on RunPod next to training
and the reasoning engine — **never the SafeLens Supabase project**, which
OpenClaw cannot access.

## Run

```bash
docker compose -f docker-compose.postgres.yml up -d
cp .env.example .env          # adjust the password before any non-local use
export $(grep -v '^#' .env | xargs)
```

## Wire into the graph

```python
from agentic_hse.graph import build_graph
from langgraph_checkpointer import open_checkpointer

async with open_checkpointer() as saver:           # scoped to app lifetime
    graph = build_graph(checkpointer=saver)
    cfg = {"configurable": {"thread_id": "site-1"}}
    await graph.ainvoke(initial_state, config=cfg)  # pauses at interrupt() if score>=10
    # resume after human decision:
    from langgraph.types import Command
    await graph.ainvoke(Command(resume={"decision": "approve"}), config=cfg)
```

For a long-lived server, use `build_checkpointer_from_pool()` instead and close
the pool on shutdown.

## Optional: mirror checkpoints into Supabase later

This is a **user-run** step — OpenClaw never connects to Supabase. If you later
want approval/checkpoint history inside Supabase, point a one-off ETL at the
`checkpoints*` tables this saver creates and load them into the
`agent_actions_log` / `approval_records` tables defined in
`../db/supabase_schema_proposal.sql`.
