# SafeLens Agentic HSE Bundle Index

Prepared on June 15, 2026. The complete runtime and artifact bundle is stored
only on `feat/agentic-hse-draft` in
`gabe3laka/safelens-deimv2-worker`; `main` was not changed. Nothing was deployed,
trained on a GPU, uploaded to Google Drive, or connected to Supabase.

## Delivery Status

- Git branch: `feat/agentic-hse-draft`
- Runtime: locally runnable after installing dependencies
- Checkpointing: in-memory by default; durable self-hosted Postgres when configured
- RunPod reasoning: typed deterministic scaffold; VLM execution deferred
- RAG: builder and matching user-run imports ready; no private documents supplied
- Datasets: acquisition deferred; manifest and preparation pipeline ready
- ZIP: sanitized local handoff created after commit and branch push

## Required Layout

| Path | Purpose |
|---|---|
| `report.md` | 38-section architecture, governance, implementation, and rollout report |
| `DATASET_COLLECTION_PROMPT.md` | Future Google Drive/plugin dataset acquisition prompt |
| `agents/` | Public Agent 1-6 imports backed by the canonical runtime package |
| `agentic_hse/` | LangGraph runtime, typed models, routes, and node implementations |
| `schemas/` | Canonical JSON schemas and production/reference dataset catalogs |
| `integration/` | OpenAPI, worker, and Cloudflare plans |
| `db/` | User-run additive Supabase proposal and 384-dimension pgvector loader |
| `checkpointer/` | Self-hosted Postgres saver, Compose file, and environment example |
| `runpod_reasoning/` | Prompt, 14 examples, schema, service, Dockerfile, plan, and 12 eval pairs |
| `runpod_training/` | Hardened dataset preparation, train config, and GPU/evaluation plan |
| `datasets/` | Safe-only production manifest, reference catalog, research, and private intake README |
| `rag/` | Local document parser, FAISS builder, loader-ready document/chunk exports |
| `training-materials/` | Nine anonymized training-output templates |

## Runtime Contracts

The canonical implementation is `agentic_hse/`.
`agents/` contains thin re-exports so there is no duplicate agent
logic.

The API surface is:

- `GET /agentic/health`
- `POST /agentic/session/start`
- `POST /agentic/reason`
- `POST /agentic/risk-assessment/draft`
- `POST /agentic/risk-assessment/approve`
- `POST /agentic/audit/draft`
- `POST /agentic/training/draft`

Callers cannot provide a reasoning-service URL. The worker reads
`SAFELENS_REASONING_URL` from server configuration. Risk assessments always
interrupt for approval. Approve executes the preview, reject holds it, and
revise creates a new preview and interrupts again.

## Data Contracts

`datasets/dataset_manifest.json` and its branch mirror contain only verified
commercial-safe source/tool entries. NC, AGPL, unverified, and platform-term
references live in `dataset_reference_catalog.json` and are excluded from
preparation.

The RAG builder uses `sentence-transformers/all-MiniLM-L6-v2` at its native 384
dimensions and emits:

- `documents_export.jsonl` / `documents_export.csv`
- `chunks_export.jsonl` / `chunks_export.csv`

The CSV columns match `db/pgvector_loader.sql`, which inserts parent
documents before foreign-keyed chunks.

## Validation

Focused tests cover score boundaries, fail-safe reasoning, model validation,
multi-hazard isolation, zero confidence, graph interrupt/approve/revise,
grouped dataset splits, RAG/SQL compatibility, JSON parsing, and OpenAPI
security fields.

Packaging excludes `.git`, caches, compiled Python, environments, credentials,
raw/private media, generated training sets, model weights, and checkpointer
database files. The final ZIP and SHA-256 are reported after creation.
