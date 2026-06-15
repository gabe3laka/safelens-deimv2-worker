# Worker Integration Plan

## Goal

Add the agentic HSE layer beside the existing worker without adding VLM latency
to `/detect`, `/ws/vision`, or `/build/*`.

## Current Draft State

`agentic_hse/` contains the canonical graph, six node
implementations, typed models, reasoning client, approval helpers, and FastAPI
router. `server.py` mounts the router under `/agentic`.

This is a local draft only. It is not pushed, deployed, or connected to
production data.

## Runtime Separation

- `/detect` and `/ws/vision`: fast detector paths.
- `/build/*`: selected-crop Build/Plan processing.
- `/agentic/reason`: contextual reasoning through the server-configured
  `SAFELENS_REASONING_URL`; callers cannot select an origin.
- LangGraph workflow: asynchronous business workflow with durable checkpointing.

Detector inference must return without waiting for a VLM. The application or a
background workflow decides whether a detection requires agentic reasoning.

## Risk-Sensitive Trigger

Call the reasoning service for labels such as:

`open_hole`, `open_panel`, `suspended_load`, `ladder`, `scaffold`, `forklift`,
`fire`, `smoke`, `spill`, `trailing_cable`, `blocked_exit`, `gas_cylinder`, and
negative PPE classes.

The trigger can also come from zone rules, user intent, an incident workflow, or
low detector confidence. A label is an invitation to reason, not proof of a
violation.

## Configuration Artifacts

The worker or deployment pipeline consumes reviewed versions of:

- `datasets/dataset_manifest.json`: source/license policy
- `schemas/model_registry.json`: approved model records and metrics
- `schemas/hazard_taxonomy.json`: hazard mapping and semantics
- `schemas/model_classes.json`: canonical 25 detector classes
- `schemas/reasoning_schema.json`: typed RunPod response
- approved labels/configs from the dataset preparation output
- a restricted model-weights URL only after human promotion approval

Configuration should be content-addressed or versioned. The worker should log
the model/config version used for each decision.

## Reasoning Call

1. Worker receives detections and context.
2. Filter to risk-sensitive detections.
3. Send only the needed selected frame/crop reference and structured context
   through the signed internal route.
4. Validate the response with `ReasoningRecord`.
5. Recompute `score`, band, and approval requirement deterministically.
6. Store evidence references rather than large image payloads in graph state.

If reasoning is unavailable, the node emits a fail-safe high-risk record,
requires human approval, and records the outage. It never returns an implicit
safe result.

## Human Approval

Every generated risk assessment requires approval. In addition, any reasoning
score `>= 10` must route to LangGraph `interrupt()` even outside that workflow.

- `approve`: execute only the already-previewed allowed action.
- `reject`: hold the action and log the reason.
- `revise`: update the preview, recompute score/band, and interrupt again.
- `score >= 17`: recommend halt/isolation pending authorized human action.

No route may deploy models, close incidents, or issue external reports merely
because an AI response recommends it.

## Checkpointing

Use the self-hosted Postgres checkpointer in `checkpointer/`. Configure one
stable `thread_id` per workflow/session and keep the saver alive for the
application lifetime. Do not point the LangGraph saver at Supabase.

## Model Promotion

The training job emits candidate weights and an evaluation report. A candidate
must outperform or deliberately trade off against the incumbent `yolo26` on:

- aggregate mAP
- per-class AP/recall
- false-negative rate on high-consequence classes
- site/camera/lighting slices
- latency and memory

After approval, a user-controlled process updates `model_registry.json` and
publishes a restricted weights URL. The worker does not discover or promote
unapproved outputs automatically.

## Rollout

1. Run the graph locally with deterministic reasoning records.
2. Connect a private staging RunPod reasoning service.
3. Verify pause/resume with self-hosted Postgres.
4. Add application persistence through user-reviewed Supabase migrations.
5. Shadow agentic decisions beside production detection.
6. Review false positives/negatives and approval workload.
7. Enable selected workflows for authorized users.

## Rollback

The router is additive. Disable `/agentic/*` at the edge or remove the optional
router mount while leaving detection and Build Mode untouched. Model rollback
uses the previous approved registry entry.
