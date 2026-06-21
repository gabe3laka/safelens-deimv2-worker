# SafeLens worker runbook

Operational guide for the SafeLens vision worker (RunPod load-balancing HTTP
endpoint). Covers rollback, readiness/state, tuning, feature toggles, secret/
weight hygiene, and the top failure modes.

> Architecture recap: a long-running FastAPI/uvicorn server. `/health` and
> `/ping` are **liveness** (always 200). `/ready` is **readiness** (200 only when
> the model + config + risk matrix are ready). Weights resolve at **runtime**
> (volume / HF cache / registry) — never baked into the image.

---

## 1. Rollback

The image is published to `ghcr.io/gabe3laka/safelens-deimv2-worker:latest` on
push to `main` (and tagged by commit SHA when CI passes `BUILD_SHA`).

1. Find the last-good image tag (commit SHA) from the GHCR package or the
   `docker-publish` workflow run.
2. In the RunPod endpoint, pin the container image to that SHA tag instead of
   `latest` and redeploy. (Avoid `latest` for production — pin a SHA.)
3. Confirm `GET /debug/state` → `runtime.build_sha` matches the rolled-back SHA.
4. Confirm `GET /ready` returns 200 before sending live traffic.
5. Feature-flag rollback (no redeploy needed): most new behavior is env-gated —
   set `RISK_ENGINE_ENABLED=false` and/or `VLM_REASONER_ENABLED=false` to revert
   to plain detection instantly.

## 2. Reading `GET /debug/state`

Non-sensitive snapshot (no secrets/tokens). Key blocks:

- `backend_status` — requested vs **active** backend, fallback state, model load.
- `effective_config` / `last_detect_effective_config` — resolved conf/img_size/
  iou/max_det and where each came from (`payload` / `env:...` / `default`).
- `risk_engine` — flags, matrix profile/version, active sessions, last-eval
  risk/alert counts + highest level.
- `reasoner` — VLM enabled/mode/model id, trigger level, timeout, active sessions.
- `open_vocab_scanner` — GroundingDINO enabled/backend/interval (candidate-only).
- `runtime` — `build_sha`, `uptime_s`, `degraded`, `degradation_mode`,
  `degradation_ladder`, `accepting_frames`, `auth` (enabled? header? — never the
  secret), `input_guards` (max bytes / megapixels), `active_sessions`.

## 3. Checking `GET /ready`

```bash
curl -fsS -H "x-worker-secret: $WORKER_SHARED_SECRET" https://<worker>/ready
```

- `200` → model loaded, config valid, risk matrix valid (or risk disabled),
  accepting frames.
- `503` → not ready. Inspect the body: `model_loaded`, `matrix_valid`,
  `accepting_frames`, `checks.risk_matrix`. The gateway should poll `/ready`
  before routing live traffic and back off on a `/detect` 503.

## 4. Stronger YOLO settings

Generic `YOLO_*` names take precedence; legacy `YOLO26_*` still work. For better
recall (demo/testing — verify YOLO license before commercial use):

```env
VISION_BACKEND=ultralytics
YOLO_DET_MODEL_ID=yolo11s.pt
YOLO_IMG_SIZE=960
YOLO_CONF=0.10
YOLO_IOU=0.60
YOLO_MAX_DETECTIONS=300
```

Verify what actually took effect via `/debug/state.effective_config.active_detector`.
Payload `conf`/`img_size` override env per request (HSE profiles).

## 5. Enable / disable the risk engine

```env
RISK_ENGINE_ENABLED=true     # adds tracks/scene_graph/risks + schema_version risk.v1
RISK_TRACKING_ENABLED=true
RISK_SCENE_GRAPH_ENABLED=true
RISK_MATRIX_PROFILE=/app/risk/risk_matrix_profile.json
SESSION_TTL_MS=30000
SESSION_MAX_ACTIVE=64
```

Disable (`RISK_ENGINE_ENABLED=false`) → `/detect` returns the legacy shape
(byte-for-byte), no risk fields. A malformed `RISK_MATRIX_PROFILE` fails `/ready`
(it does not silently fall back mid-stream).

## 6. Enable / disable the Gemini reasoner

```env
VLM_REASONER_ENABLED=true    # event-driven; NEVER per-frame; never blocks /detect
REASONER_MODE=gemini        # gemini | mock | disabled (removed transformer modes return unavailable)
QWEN_VL_DEEP_MODEL_ID=Qwen/Qwen2.5-VL-7B-Instruct
QWEN_VL_DEEP_ENABLED=false
REASONER_MAX_IMAGE_SIDE=512
REASONER_TRIGGER_LEVEL=YELLOW
REASONER_MIN_INTERVAL_MS=1500
REASONER_TIMEOUT_MS=2500
```

- `REASONER_MODE=mock` → CPU, weight-free draft contract (integration without GPU).
- VLM output is always an **AI draft** (`requires_human_review=true`,
  `should_alert=false`); it never becomes the safety authority.
  degrades gracefully to full precision with diagnostics in `/debug/state`.
- Open-vocab scanner (separate, optional): `OPEN_VOCAB_SCANNER_ENABLED=true`
  (GroundingDINO; candidate-only). Both degrade to a clear status if weights/deps
  are missing.

## 7. Confirm no weights / secrets are committed

```bash
# No weights/datasets tracked in git:
git ls-files | grep -E '\.(pt|pth|onnx|safetensors|bin|gguf|engine|ckpt)$' || echo "OK: no weights tracked"
git ls-files | grep -E '\.(jpg|jpeg|png|mp4|mov)$' || echo "OK: no media tracked"
# No secrets in the image build context (.dockerignore excludes them):
grep -E '\.env|\*secret\*|\*token\*' .dockerignore
# Image carries no model weights (build does NOT download or bake them):
grep -n "NO model weights are baked" Dockerfile
# Secrets never appear in logs (redaction): structured logs redact
# image_b64/frame_b64/authorization/hf_token/secret/api_key.
```

Weights resolve at runtime only: `/runpod-volume/models/*`, the HF cache, or an
approved registry (`model_registry.example.json`, all `commit_weights:false`).
For **air-gapped / no-egress** deployments, pre-populate `/runpod-volume/models/`
on the volume (the container runs as uid 10001 — the volume must be writable by it).

## 8. Authentication

Set a shared secret so only the Supabase proxy can reach the worker:

```env
WORKER_SHARED_SECRET=<random-long-string>
WORKER_AUTH_HEADER=x-worker-secret   # or send "Authorization: Bearer <secret>"
```

Every route except `/health` and `/ping` then requires the secret; `/ws/vision`
authenticates on connect (header or `?token=`). Unset = compatibility/testing
mode (auth disabled) — never run production without it.

## 9. Top failure modes & fixes

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `/ready` 503, `model_loaded:false` | model still loading / load failed | check `/debug/state.state.error_traceback`; `POST /warmup?wait=true`; verify weights reachable on the volume / HF cache |
| `/ready` 503, `matrix_valid:false` | malformed `RISK_MATRIX_PROFILE` | fix the profile (bands monotonic/contiguous/full-coverage) or unset to use the bundled default |
| `/detect` 401 | missing/invalid worker secret | send `WORKER_SHARED_SECRET` via `x-worker-secret` (proxy config) |
| `/detect` 413 `payload_too_large` / `image_too_large` | body > `MAX_REQUEST_BYTES` or image > `MAX_IMAGE_MEGAPIXELS` | downscale client-side or raise the caps deliberately |
| `/detect` 503 `model_not_ready` | cold worker | gateway should retry with backoff; `/warmup` |
| `/detect` 503 `shutting_down` | graceful shutdown in progress | expected during redeploy; gateway drains/retries elsewhere |
| `warning: risk_engine_error...`, `degradation_mode:no_risk` | risk layer failed | detection is preserved; inspect logs; risk auto-degrades (never 500) |
| `reasoner_status:unavailable` | Gemini API key/deps absent or removed mode selected | provide `GEMINI_API_KEY`, use `mock`, or set `disabled`; YOLO detection continues |
| backend shows `edgecrafter` when `yolo26` requested | YOLO load failed → auto-fallback | see `backend_status.fallback_reason`; fix YOLO weights/licensing |
| volume write errors at runtime | `/runpod-volume` not writable by uid 10001 | chown the volume / mount writable (non-root container) |

## 10. Observability

- `GET /metrics` (Prometheus text): `safelens_model_ready`, `safelens_ready`,
  `safelens_active_sessions`, `safelens_detect_requests_total`,
  `safelens_detect_errors_total`, `safelens_detect_latency_ms{quantile=...}`,
  `safelens_risk_level_total{level=...}`, `safelens_reasoner_status_total{status=...}`,
  `safelens_degradation_rank`, `safelens_ws_dropped_frames_total`. Scrapers must
  send the worker secret (or add `/metrics` to `WORKER_PUBLIC_PATHS`).
- Structured JSON logs carry `event`, `build_sha`, `session_id`, `frame_id` and
  **redact** imagery/secrets. System-health alerts (model load failed, latency
  budget, drop-rate) are separate from HSE risk alerts.
