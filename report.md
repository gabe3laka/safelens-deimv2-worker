# SafeLens Agentic HSE System Report

## 1. Executive Summary

SafeLens should evolve from a vision detector into a guarded vertical HSE
system: it observes a scene, reasons about context, drafts safety work products,
pauses for human approval where risk is material, and logs every decision.

This bundle contains a locally runnable draft of that orchestration beside the
existing FastAPI worker. Six agent nodes are wired through LangGraph, every risk
assessment routes to an `interrupt()` approval gate, typed reasoning is enforced
by Pydantic, and checkpoint support targets self-hosted Postgres. The bundle also
contains user-run SQL, a local RAG builder, dataset preparation tooling,
reasoning prompts/examples, API plans, and training templates.

The complete runtime and artifact bundle is maintained only on
`feat/agentic-hse-draft` in `gabe3laka/safelens-deimv2-worker`; `main` was not
changed. Nothing was deployed, connected to Supabase, or used to train a model.
The commercial detector direction is DEIM/DEIMv2 or RT-DETR under
permissive licensing. The present worker still defaults to `yolo26` with
EdgeCrafter fallback, so detector migration remains a reviewed future change.

## 2. Product Vision

The product vision is an auditable safety co-pilot for supervisors, HSE teams,
and operations leaders. SafeLens combines live visual evidence, company rules,
site context, risk matrices, and human decisions to help teams identify hazards
earlier and produce consistent follow-up records.

The product is not an autonomous safety authority. It proposes, explains, and
prepares. Accountable people decide and execute.

## 3. Why A Vertical Safety Agent

A generic chatbot can summarize text but lacks the controlled state, domain
contracts, visual evidence, risk scoring, approval boundaries, and audit trail
required for HSE work. A vertical system can encode:

- company-specific PPE and permit rules
- 3x3, 4x4, or 5x5 risk matrices
- relational hazards such as a load above people
- standards and internal-document retrieval
- mandatory approval for high-risk actions
- structured audit, training, and model-improvement outputs

That combination makes the system useful in operations without pretending that
language generation is a substitute for competent-person judgment.

## 4. Source Context

The bundle was prepared from:

- app repo `gabe3laka/HSE-eagle-vision-2` at `6b9f241`
- worker repo `gabe3laka/safelens-deimv2-worker` at `3a839d1`
- local draft branch `feat/agentic-hse-draft`

The existing worker routes for detection, warmup, debug, Build Mode, and vision
streaming are retained. The agentic layer is mounted under `/agentic/*`.

Primary implementation and research references include:

- LangGraph: https://github.com/langchain-ai/langgraph
- Pydantic AI: https://github.com/pydantic/pydantic-ai
- pgvector: https://github.com/pgvector/pgvector
- DEIM: https://github.com/ShihuaHuang95/DEIM
- Roboflow Universe: https://universe.roboflow.com/
- Hugging Face Datasets: https://huggingface.co/datasets
- SHEL5K: https://data.mendeley.com/datasets/9rcv8mm682/4

## 5. Hostinger And OpenClaw Role

Hostinger/OpenClaw is an authoring and orchestration workspace. Its allowed role
in this handoff is to prepare code, prompts, schemas, plans, SQL files, and local
artifacts.

It does not:

- connect to or write to Supabase
- deploy the app or worker
- run RunPod training
- promote model weights
- publish customer images
- close incidents or send regulatory reports

## 6. Delivery Model

Delivery is a local folder and ZIP bundle. Any later transfer should use a
restricted, expiring link controlled by the user. Private customer images must
never be placed in a public repository, public dataset, or world-readable link.

The SQL files are proposals for the user to review and run. The RAG process
builds a local index and export file; it does not upload embeddings.

## 7. RunPod And GPU Role

RunPod has two distinct roles:

1. Train and evaluate candidate vision models from approved datasets.
2. Host the Senior-QHSE-Manager reasoning service for contextual scene analysis.

The reasoning service should be called only when detections or workflow context
justify deeper analysis. Normal `/detect` and Build Mode latency should not
depend on a large VLM call.

## 8. Framework Comparison

LangGraph is the preferred top-level orchestrator because its explicit graph
state, conditional edges, `interrupt()` mechanism, resume semantics, and
checkpoint support match a safety workflow.

Pydantic AI is useful inside nodes for typed model interactions and validation.
The OpenAI Agents SDK is strong for tool-driven agent applications but this
design benefits from LangGraph's explicit durable workflow. CrewAI is convenient
for role-based collaboration but provides less direct control over this approval
state machine. Google ADK and Microsoft Agent Framework are credible ecosystem
options, but would add a platform decision without improving the present
workflow. AutoGen is not recommended as the primary direction for this build.

## 9. Framework Recommendation

Use LangGraph for state, routing, pausing, resuming, and audit flow. Use Pydantic
models to validate reasoning and generated records. Keep detector inference and
reasoning service calls behind small, testable clients.

## 10. Multi-Agent Architecture

The implemented graph is:

`START -> setup -> observation -> risk_assessment -> audit -> training -> vision_improvement`

After the six domain nodes, a generated risk assessment always pauses:

- approve: `approval(interrupt) -> execute -> log -> END`
- reject: `approval(interrupt) -> log -> END`
- revise: `approval(interrupt) -> revise preview -> approval(interrupt)`

This gives the system a deterministic control path even when an LLM or VLM
produces the underlying reasoning record.

## 11. Agent Roles

1. Company Setup Agent normalizes company rules, site rules, and matrix settings.
2. Safety Observation Agent converts detections into contextual hazard events.
3. Risk Assessment Agent prepares the risk record and approval request.
4. Audit Agent drafts objective evidence and corrective/preventive actions.
5. Vision Improvement Agent queues uncertain frames and a guarded retraining plan.
6. Training Agent creates nine anonymized training-module previews.

The names are roles in one controlled graph, not independent autonomous actors.

## 12. Preview, Approval, Execute, Log

The governing cycle is:

`Preview -> Human approval -> Execute -> Log`

The draft implementation never treats generation as approval. Rejection holds
the action; revision recomputes the preview and interrupts again. Approval
records and action logs are first-class data
contracts, and critical recommendations still require the site's authorized
person to act.

## 13. Component Responsibility Split

| Component | Responsibility |
|---|---|
| Frontend | Capture user intent, display evidence and previews, collect approval |
| Cloudflare | Authenticate, rate-limit, route tenants, hide internal origins |
| Supabase | User-managed application records, RLS, incidents, risk and audit data |
| Hostinger/OpenClaw | Author and package local artifacts; no direct Supabase writes |
| Vision worker | Fast detection, Build Mode, agentic HTTP routes |
| RunPod reasoning | Contextual VLM inference returning typed JSON |
| RunPod training | Candidate model training and held-out evaluation |
| Self-hosted Postgres | LangGraph checkpoints and approval-resume durability |

## 14. HSE Intelligence Layer

The intelligence layer sits between raw detections and business actions. It
combines:

- visual detections and frame references
- zone geometry and people exposure
- company rules and site-specific controls
- retrieved policies, SOPs, and risk records
- a deterministic likelihood/severity matrix
- standards-aware recommendations

This layer prevents a detector label from being treated as a complete safety
judgment.

## 15. Senior-QHSE Reasoning Engine

The RunPod reasoning engine receives detections, scene context, company profile,
and zone context. It returns one validated `ReasoningRecord` containing hazard,
location, people exposed, latent/active state, trigger, score, controls,
reasoning, standard reference, and approval requirement.

The engine should use an open-weight VLM with terms suitable for the product.
The current service file is a deterministic contract stub so integration can be
tested before a selected model is loaded.

## 16. Relational Risk

Risk is relational rather than object-only:

- a cup in the middle of a stable table is not a material hazard
- the same hot cup at an edge above a worker can be high risk
- a spill in a locked room is different from oil at a stair base
- a loose bracket at ground level differs from one above an occupied walkway

The reasoning dimensions are object, geometry, height/drop path, people
exposure, dynamics, existing controls, and latent/active state with a named
trigger.

## 17. Risk Matrix Logic

The default 5x5 matrix is:

`score = likelihood * severity`

| Score | Band | Default control |
|---|---|---|
| 1-4 | Low | Log and manage locally |
| 5-9 | Medium | Review and assign controls |
| 10-16 | High | Mandatory human approval |
| 17-25 | Critical | Recommend halt/isolation pending approval |

Every score of 10 or more routes through LangGraph `interrupt()`. Company
profiles may select 3x3 or 4x4 matrices later, but the approval policy must be
mapped explicitly rather than silently changed.

## 18. Company Setup And RAG

Company Setup normalizes supplied matrix settings, PPE rules, restrictions,
permit controls, and document/RAG references into a Company Safety Profile. It
does not parse documents by itself. The separate local RAG builder walks
supplied documents, chunks text, creates 384-dimensional embeddings, builds a
local FAISS index, and emits matching document/chunk JSONL and CSV exports.

OpenClaw does not upload that export. A user may load it through the supplied
pgvector staging SQL after reviewing ownership and tenant fields.

## 19. Safety Observation

The observation node filters risk-sensitive detections and calls only the
server-configured private reasoning service. If the service is unavailable, it
emits a fail-safe score-16 record requiring review instead of crashing or
returning an implicit safe result.

Remote reasoning output is Pydantic-validated before entering graph state.

## 20. Risk Assessment Generation

The risk node selects the highest-scoring event, matches its own reasoning
record, and drafts initial/residual scores, people exposed, controls,
responsible person, due date, and standard reference. Every draft creates a
pending approval payload. Residual values remain proposed until the site confirms
that controls are installed and effective.

## 21. Audit Generation

The audit node drafts:

- observation and classification
- objective evidence reference
- checklist question violated
- possible root cause for confirmation
- risk band
- corrective and preventive action
- responsible role
- due date
- verification method
- standard reference

The draft is editable evidence, not a final finding until an authorized auditor
reviews it.

## 22. Training Generation

The training node prepares anonymized previews for toolbox talks,
micro-learning, quizzes, worker/supervisor briefings, training records,
before/after explanations, method-statement summaries, and refresher training.
Templates use placeholders and prohibit personal or customer-identifying
evidence.

Generated learning material must be checked against approved site procedures
before issue.

## 23. Vision Improvement Loop

The vision-improvement node queues uncertain detections as dataset candidates.
Each candidate is marked for privacy review and `auto_deploy=false`.

The loop is:

`candidate -> anonymize -> label/QA -> train -> evaluate -> human approve -> registry`

No model is automatically promoted.

## 24. Computer Vision Pipeline

The recommended commercial direction is DEIM/DEIMv2 (Apache-2.0), with RT-DETR
as an alternative. Training should:

1. select commercial-safe public sources
2. add approved private candidates
3. normalize the 25-class taxonomy
4. deduplicate and split by source/site
5. export COCO and YOLO representations
6. train on RunPod
7. evaluate per class and by hazard scenario
8. compare against the incumbent `yolo26`
9. request human promotion approval

## 25. Hazard Dataset Research

`datasets/hazard_dataset_research.md` covers PPE, falls, ladder/scaffold, open
holes, slips/trips, electrical, fire/hot work, lifting, confined space, manual
handling, forklift interaction, guarding, chemicals, cylinders, excavation,
blocked exits, lighting, and lone/restricted work.

Commercial-safe seed data is strongest for PPE and general construction.
Relational and rare hazards need approved private data and synthetic generation.

## 26. Data Licensing

`datasets/dataset_manifest.json` contains commercial-safe production sources
only. NC, AGPL, unverified, and platform-term references are isolated in
`datasets/dataset_reference_catalog.json` and cannot be selected by the
preparation script.

Every acquired source should retain URL, version, license text, attribution,
hash, and acquisition date.

## 27. Privacy Strategy

Private/customer frames remain in restricted storage with expiring access.
Faces, badges, screens, vehicle plates, and other identifiers should be blurred
or removed before labeling where they are not required for safety analysis.

Dataset candidates need consent/purpose, retention, reviewer, and deletion
metadata. Public links and public repositories are prohibited.

## 28. Human Approval And Audit Guardrails

The system must never:

- auto-deploy a model
- auto-close an incident
- send an external report without approval
- declare a permit, isolation, or atmosphere safe from imagery
- bypass the score threshold

Graph state records previews, approval requests, human decisions/notes, and
execution or hold status. Production persistence into application audit tables
remains a user-integrated next phase.

## 29. Supabase Schema Proposal

The user-run SQL adds `org_id` to the ten existing tables without redesigning
them. New owner-scoped tables cover company/site profiles, documents/chunks,
agent memory, action logs, approvals, dataset candidates, model versions, and
evaluations.

RLS is enabled on every new table with owner policies. Organization membership
policies are intentionally deferred until the application defines a trusted
membership/role table.

## 30. Worker API

The mounted draft routes are:

- `GET /agentic/health`
- `POST /agentic/session/start`
- `POST /agentic/reason`
- `POST /agentic/risk-assessment/draft`
- `POST /agentic/risk-assessment/approve`
- `POST /agentic/audit/draft`
- `POST /agentic/training/draft`

The full schema is documented in
`integration/openapi-agentic-hse.yaml`. The reasoning origin is
configured server-side and cannot be supplied by API callers.

## 31. Cloudflare Routing And Security

Cloudflare should validate user JWTs or signed service tokens, derive tenant
context server-side, apply stricter limits to reasoning than detection, and
forward to private worker/RunPod origins. Raw pod URLs and credentials must not
reach the browser.

High and critical calls should be logged with request IDs and evidence
references, excluding image payloads and secrets.

## 32. Durable Checkpointing

When `LANGGRAPH_POSTGRES_DSN` is configured, LangGraph uses self-hosted Postgres
through `AsyncPostgresSaver`, not Supabase. Without that variable, the local
runtime uses an in-memory saver for development. Postgres mode allows an
interrupted graph to resume after process restart or shift handover; memory mode
does not.

## 33. Artifact Bundle

Key locations:

- `agentic_hse/`: canonical runtime implementation
- `agents/`: public Agent 1-6 re-exports
- `schemas/`: canonical JSON contracts
- `db/`: user-run Supabase and pgvector SQL
- `integration/`: OpenAPI and routing plans
- `checkpointer/`: self-hosted checkpoint support
- `runpod_reasoning/`: service, prompt, examples, schema
- `runpod_training/`: preparation script, config, plan
- `datasets/`: source manifest and research
- `rag/`: local indexing and export
- `training-materials/`: anonymized templates

## 34. Implementation Roadmap

1. Completed locally: typed graph, routes, fail-safe reasoning, and approval resume.
2. Completed locally: schemas, SQL proposals, RAG/data tooling, and artifact packaging.
3. Staging: connect signed RunPod reasoning and self-hosted Postgres.
4. Application integration: persist drafts, approvals, and logs through reviewed migrations.
5. Product workflows: editable audit/training interfaces and verification evidence.
6. Vision lifecycle: acquire approved data, train, evaluate, and approve candidates.
7. Enterprise: organization roles, multi-site analytics, and governance.

## 35. MVP Build Plan

The first production slice should support one company, one site, one matrix,
selected risk-sensitive classes, a self-hosted checkpointer, and explicit human
approval. Success means:

- existing detection and Build Mode remain stable
- reasoning output validates
- score 10+ pauses reliably
- approval resumes the same thread
- every outcome is logged
- no private data is publicly exposed

## 36. Future Enterprise Features

Future work may add organization membership/RBAC, evidence retention policies,
permit integrations, calibrated depth, multi-camera tracking, multilingual
training, regulatory mapping, model-drift dashboards, and signed model
provenance.

Each addition should preserve tenant isolation and accountable human decisions.

## 37. Open Questions And Risks

- Which company documents and jurisdictions govern the first deployment?
- Who is authorized to approve high and critical actions?
- What image retention and anonymization rules apply?
- Which RunPod VLM is approved after license, cost, and accuracy evaluation?
- How will `org_id` membership and service roles be modeled?
- What incumbent `yolo26` baseline and held-out data define promotion?
- Which hazards require nonvisual sensors or permit-system integrations?

The largest technical risk is false confidence from incomplete context. The
largest governance risk is allowing generated output to look like authorized
action.

## 38. Final Recommendation

Proceed with the LangGraph plus Pydantic architecture as a guarded HSE workflow,
not an autonomous safety system. Keep fast vision separate from deep reasoning,
use self-hosted checkpointing, adopt only verified commercial-safe data, and
measure candidate DEIMv2 models against the incumbent before promotion.

The next operational milestone is a user-controlled staged deployment where one
risk-sensitive observation uses the private RunPod model and self-hosted
Postgres, becomes a validated reasoning record and risk draft, pauses for a
durable human decision, and is persisted by the application audit layer.
