# CPU agentic layer inside the RunPod worker (`/agent/*`)

`agentic_cpu/` is a CPU-only agentic orchestration layer mounted into the same
FastAPI app under `/agent/*`. It drafts QHSE artefacts (risk assessments, audit
findings, CAPAs, training, safety observations, dataset candidates) from the
**structured detection JSON** the app already has — it never runs inference and
imports no GPU dependency.

## Routes

| Method | Path | Purpose | Approval |
| --- | --- | --- | --- |
| GET | `/agent/health` | liveness + enabled/mode/queue | — |
| GET | `/agent/ready` | agent-layer readiness | — |
| POST | `/agent/company/profile/extract` | structured company profile from text | none (informational) |
| POST | `/agent/safety-observation/draft` | observation -> incident draft | required |
| POST | `/agent/risk-assessment/draft` | 5x5 risk assessment draft | required |
| POST | `/agent/audit/draft` | internal HSE audit findings draft | required |
| POST | `/agent/capa/draft` | CAPA draft | required |
| POST | `/agent/training/draft` | toolbox talk + completion-record draft | required |
| POST | `/agent/vision-improvement/candidate` | dataset candidate proposal | required |
| POST | `/agent/approvals/preview` | render a draft for review | required |
| POST | `/agent/approvals/execute` | finalize an approved action | gate |
| GET | `/agent/jobs/{job_id}` | background job status/result | — |

## Approval model: Preview -> Human approval -> Execute -> Log

The agent NEVER finalizes a serious record on its own. Every serious draft is
`status="pending_approval"`, `requires_human_approval=true`. Approval-required
action types:

```
incident_create, incident_close, capa_create, risk_register_write,
risk_assessment_approve, audit_finding_send, training_record_create,
dataset_candidate_approve, model_deployment_approve
```

`POST /agent/approvals/execute` rejects an unapproved action:

```json
{ "ok": false, "error": "approval_required" }   // HTTP 403
```

With `approved: true` and an `approved_by` actor it finalizes and writes to the
action log:

```json
{ "ok": true, "action_id": "act_…", "status": "executed", "action": { … } }
```

## Bounded background jobs (never blocks `/detect`)

Draft routes run through `agentic_cpu.jobs`, a bounded thread pool **separate**
from the GPU path:

- `CPU_AGENT_MAX_INFLIGHT` — pool workers (default 2)
- `CPU_AGENT_QUEUE_MAX` — max queued+running; exceeding it returns
  `429 {"error": "queue_full"}`
- `CPU_AGENT_JOB_TIMEOUT_MS` — per-job hard timeout -> `{"status": "error",
  "error": "job_timeout"}`

Fast mock jobs complete inline and the route returns the draft (200) with a
`job_id`; long jobs return `202 {"status": "accepted", "job_id": …}` and the
caller polls `GET /agent/jobs/{job_id}`.

## Mock by default — no LLM key, no DB

- `CPU_AGENT_MODE=mock`, `CPU_AGENT_LLM_PROVIDER=mock`: agents build structured
  drafts deterministically; no API key required. An external provider (OpenAI/
  Anthropic-compatible) can be configured at deploy time via env + a key in the
  environment (never baked into the image). See `agentic_cpu/llm.py`.

## Durability (read this before production)

- `CPU_AGENT_ACTION_LOG_BACKEND=memory` and `CHECKPOINTER_BACKEND=memory` are
  **MVP defaults and NOT durable** — they are lost on worker restart.
- For production set them to `postgres` / `supabase` so the approval trail and any
  long-running graph state survive restarts. The `supabase`/`postgres` action-log
  paths are documented hooks in `agentic_cpu/action_log.py`; until a client is
  wired at deploy time they fall back to memory with a warning.
- Long approval workflows must NOT live only in RunPod memory — persist them
  externally (the worker container is ephemeral).

## Degrade first under GPU pressure

With `CPU_AGENT_DISABLE_ON_GPU_PRESSURE=true`, when
`gpu_vision.gpu_busy_ratio()` exceeds `CPU_AGENT_MAX_GPU_BUSY_RATIO` (0.85) the
agent routes return `503 degraded_gpu_pressure`. GPU vision keeps priority;
`/detect` is never throttled by the agent.

## CPU-only guarantee

`agentic_cpu` imports **no** `torch`, `torchvision`, `ultralytics`, `cv2`,
`transformers`, or vision loader. This is enforced by
`tests/test_agent_import_guard.py` (a clean-subprocess check) and re-checked in
the Docker build smoke test. If the agent needs vision data, it consumes the
detection JSON — it never imports a GPU module.

## Disable

`AGENTIC_CPU_ENABLED=false` — routes still exist; health/ready report
`enabled:false`; draft routes return `503 agentic_cpu_disabled`; vision unaffected.

Env reference: `runpod_env_reference.md`.
