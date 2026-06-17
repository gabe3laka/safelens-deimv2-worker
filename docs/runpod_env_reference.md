# RunPod env reference — temporal perception + CPU agent

All knobs are env vars (set in the Dockerfile defaults; override at deploy time).
None require a real LLM key or DB to run in mock mode. Secrets are NEVER baked
into the image.

## Temporal VLM perception (GPU side)

| Env | Default | Meaning |
| --- | --- | --- |
| `TEMPORAL_REASONING_ENABLED` | `true` (code default `false`) | master switch; off = legacy `/detect` shape |
| `TEMPORAL_MEMORY_WINDOW_FRAMES` | `45` | per-track history depth |
| `TEMPORAL_MEMORY_TTL_MS` | `30000` | session sub-record TTL (falls back to `SESSION_TTL_MS`) |
| `TEMPORAL_MAX_ACTIVE_SESSIONS` | `64` | bounded active temporal sessions |
| `TEMPORAL_STORE_KEYFRAMES` | `false` | never persist raw frames (keep false) |
| `TEMPORAL_REASONING_TRIGGER_MIN_INTERVAL_MS` | `5000` | min interval between VLM triggers per session |
| `TEMPORAL_REASONING_MAX_ASYNC_JOBS` | `2` | global cap on concurrent temporal reasoning jobs |
| `TEMPORAL_LABEL_FLIP_WINDOW_FRAMES` | `8` | window for label-instability detection |
| `SCENE_CONTEXT_ENABLED` | `true` | enable scene-context refresh |
| `SCENE_CONTEXT_REFRESH_MS` | `15000` | periodic scene-context refresh interval |
| `SCENE_HINT_ENABLED` | `true` | honor `scene_hint`/`site_context` from the request |
| `CONTEXTUAL_SUPPRESSION_ENABLED` | `true` | allow indoor suppression of vehicle FPs |
| `SEMANTIC_CORRECTION_ENABLED` | `true` | enable perception corrections |
| `SEMANTIC_CORRECTION_LOW_CONF_THRESHOLD` | `0.35` | low-confidence threshold |
| `OBJECT_EDGE_RISK_ENABLED` | `true` | enable deterministic object-near-edge risk |
| `OBJECT_EDGE_DISTANCE_THRESHOLD` | `0.10` | normalized near-edge distance |
| `OBJECT_EDGE_HISTORY_FRAMES` | `6` | frames used for edge-motion |
| `REASONER_RESULT_STALE_MS` | `12000` | age after which a cached reasoner result is `stale` |
| `REASONER_HUMAN_REVIEW_SCORE` | `10` | risk score at/above which escalation favours human review |
| `GPU_REASONER_MAX_INFLIGHT` | `1` | bounded GPU reasoner slots (drop-if-busy) |

The temporal VLM reuses the `risk.vlm_reasoner` model knobs: `VLM_REASONER_ENABLED`
(default `false`), `REASONER_MODE` (`qwen_vl`|`deepseek_vl2`|`mock`|`disabled`),
`REASONER_MIN_INTERVAL_MS`, `REASONER_CACHE_DIR`, `PRIVACY_BLUR_ENABLED`, etc.
(unchanged — see the Dockerfile / `docs/runbook.md`).

## CPU agent

| Env | Default | Meaning |
| --- | --- | --- |
| `AGENTIC_CPU_ENABLED` | `true` (code default `false`) | master switch; routes always exist, behaviour gated |
| `CPU_AGENT_MODE` | `mock` | `mock` = deterministic, no LLM key |
| `CPU_AGENT_MAX_INFLIGHT` | `2` | CPU agent thread-pool workers (separate from GPU) |
| `CPU_AGENT_QUEUE_MAX` | `16` | max queued+running; exceeding -> HTTP 429 `queue_full` |
| `CPU_AGENT_JOB_TIMEOUT_MS` | `30000` | per-job hard timeout -> structured error |
| `CPU_AGENT_REQUIRE_APPROVAL` | `true` | serious actions require explicit approval to execute |
| `CPU_AGENT_ACTION_LOG_BACKEND` | `memory` | `memory` (not durable) \| `postgres` \| `supabase` |
| `CHECKPOINTER_BACKEND` | `memory` | graph/checkpoint state backend (`memory` not durable) |
| `CPU_AGENT_LLM_PROVIDER` | `mock` | `mock` \| `openai` \| `azure-openai` \| `anthropic` |
| `CPU_AGENT_LLM_MODEL` | `mock` | model id for a real provider |
| `CPU_AGENT_DISABLE_ON_GPU_PRESSURE` | `true` | CPU agent degrades first under GPU pressure |
| `CPU_AGENT_MAX_GPU_BUSY_RATIO` | `0.85` | GPU-busy ratio above which the agent degrades |

A real LLM provider also needs a key in the environment (e.g. `OPENAI_API_KEY` /
`ANTHROPIC_API_KEY`) — set at deploy time, never baked into the image.

## Production notes

- Set `WORKER_SHARED_SECRET` at deploy time (auth is disabled when unset).
- Switch `CPU_AGENT_ACTION_LOG_BACKEND` and `CHECKPOINTER_BACKEND` off `memory`
  for a durable approval trail (the container is ephemeral).
- Weights for the VLM resolve at runtime into the model cache / volume; they are
  never downloaded at build or baked into the image.
- To turn the CPU agent off entirely: `AGENTIC_CPU_ENABLED=false` (vision
  unaffected). To turn temporal perception off: `TEMPORAL_REASONING_ENABLED=false`
  (legacy `/detect` shape).

## New `/metrics` series

```
safelens_gpu_detect_latency_ms
safelens_gpu_reasoner_jobs_inflight
safelens_gpu_reasoner_jobs_dropped_total
safelens_temporal_triggers_total
safelens_cpu_agent_jobs_inflight
safelens_cpu_agent_queue_depth
safelens_cpu_agent_jobs_completed_total
safelens_cpu_agent_jobs_failed_total
safelens_cpu_agent_approval_required_total
```
