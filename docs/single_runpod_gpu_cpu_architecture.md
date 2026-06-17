# Single RunPod worker: GPU vision + CPU agent (MVP)

This worker runs BOTH GPU vision/perception and lightweight CPU agentic
orchestration in **one repo, one Docker image, one FastAPI app, one uvicorn
process, one public port (8000)**.

```
App -> Cloudflare Gateway -> one RunPod worker on :8000
                              ├── GPU vision/perception routes
                              │     /detect /reason /scan /warmup /ready
                              │     /metrics /debug/* /ws/vision /build/*
                              └── CPU agentic orchestration routes
                                    /agent/*
```

This is an **MVP** choice: keep everything in one worker for simplicity, but keep
the layers internally separated so the CPU agent can later be split into its own
(cheaper, CPU-only) RunPod endpoint without an app rewrite.

## Why one worker (for now)

- One image, one deploy, one URL behind Cloudflare -> least moving parts.
- The CPU agent is lightweight and human-paced; co-locating it avoids a second
  service while we prove the contract.
- The internal seams (`agentic_cpu` as a mounted router on its own bounded queue,
  consuming structured JSON only) mean "split later" is a deployment change, not a
  redesign.

## Hard rule: one worker, NOT one blocking loop

GPU routes are latency-sensitive; CPU agent routes may be slow. They never share a
blocking path:

| Concern | GPU vision | CPU agent |
| --- | --- | --- |
| Runs every frame | detector + tracking + scene graph + deterministic risk | never |
| VLM | event-triggered, non-blocking, bounded GPU slot | never imports it |
| Concurrency limit | `GPU_REASONER_MAX_INFLIGHT` (`gpu_vision`) | `CPU_AGENT_MAX_INFLIGHT` (`agentic_cpu.jobs`) |
| Queue | drop-if-busy (no unbounded queue) | bounded `CPU_AGENT_QUEUE_MAX` -> 429 |
| Blocks `/detect`? | no (reasoner is async) | **never** (separate queue + thread pool) |

Concretely:

- `/detect` never waits on a CPU agent task and never waits indefinitely on the
  VLM (the reasoner is triggered async and `/detect` attaches the latest cached
  result).
- GPU reasoner jobs and CPU agent jobs have **separate** concurrency limits and
  thread pools.
- Approvals/action logs are returned as draft payloads and meant to be persisted
  **externally** for production (the in-memory backend is not durable).

## Degrade the CPU agent first, never `/detect`

When the GPU is under pressure (`gpu_vision.gpu_busy_ratio()` above
`CPU_AGENT_MAX_GPU_BUSY_RATIO`, default 0.85, with
`CPU_AGENT_DISABLE_ON_GPU_PRESSURE=true`), the CPU agent routes return a
structured `503 degraded_gpu_pressure` so GPU vision keeps priority. `/detect` is
never throttled by the agent layer.

## Components

| Package | Role | GPU deps? |
| --- | --- | --- |
| `server.py` | the one FastAPI app; existing routes preserved | yes (vision) |
| `gpu_vision/` | bounded GPU reasoner concurrency + GPU-pressure signal | no (torch probed lazily) |
| `temporal_reasoning/` | event-triggered temporal VLM perception (additive `/detect` fields) | no at import (VLM lazy) |
| `agentic_cpu/` | CPU agent layer, mounted at `/agent/*` | **none** (enforced by test) |
| `shared/` | cross-layer pydantic schemas, prompts, wire contracts | no |

## Readiness

`GET /ready` reports both layers but overall readiness is the **GPU vision worker
only** — a disabled/not-ready CPU agent never fails vision readiness:

```json
{ "ok": true, "gpu_vision_ready": true,
  "agentic_cpu_ready": false, "agentic_cpu_enabled": false, "degraded_mode": null }
```

## Disable the CPU agent

Set `AGENTIC_CPU_ENABLED=false`. The `/agent/*` routes still exist (health/ready
report `enabled:false`); draft routes return `503 agentic_cpu_disabled`. Vision is
completely unaffected.

## Splitting the CPU agent into its own RunPod endpoint later

Because `agentic_cpu` only consumes structured detection JSON and imports no GPU
deps:

1. Build a CPU-only image whose entrypoint mounts `agentic_cpu.get_router()`.
2. Point the app's `/agent/*` calls at that endpoint's URL.
3. Forward the same detection JSON the app already has.

No change to the GPU worker or the agent code is required.

See also: `temporal_reasoning_integration.md`, `agentic_cpu_inside_runpod.md`,
`app_cloudflare_contract.md`, `runpod_env_reference.md`.
