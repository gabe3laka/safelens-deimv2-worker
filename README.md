# safelens-deimv2-worker

SafeLens vision worker for RunPod -- live HTTP server mode.

## Vision backends (YOLO26 default, EdgeCrafter fallback)

| Backend | Role | Selected by |
|---------|------|-------------|
| **YOLO26** (Ultralytics) | **default** -- boxes + poses (+ optional seg) | `VISION_BACKEND=yolo26` (default) |
| EdgeCrafter (ECDet-S / ECPose-S) | fallback | `VISION_BACKEND=edgecrafter` or automatic |
| DEIMv2 | legacy / debug only | `VISION_BACKEND=deimv2` |

If the requested backend fails to **load** and `AUTO_BACKEND_FALLBACK=true`
(default), the worker automatically serves `FALLBACK_VISION_BACKEND`
(default `edgecrafter`). The actually-serving backend is visible in
`GET /debug/state` (`backend_status`) and in the `/detect` response
(`backend` field + a `backend_fallback: ...` `warning`). The app-facing
contract is unchanged: `entities` (normalized 0..1 `bbox` x/y/w/h), `poses`
(COCO-17 keypoints), `backend`, `tasks`, `model`, `inference_ms`,
`img_w`/`img_h` -- plus an optional additive `segments` list
(`{maskContour, source: "yolo26-seg"}`) when the YOLO26 seg task is enabled.

**YOLO26 runs in task-based modes** (never det+seg+pose on every frame):

| Mode | Used by | Tasks (env, default) |
|------|---------|----------------------|
| live | `POST /detect` HSE loop | `YOLO26_LIVE_TASKS=det` (fast boxes only) |
| build | `/build/session/frame`, `workflowMode=build` | `YOLO26_BUILD_TASKS=det,seg` (selected crop only) |
| plan | `/build/session/frame`, `workflowMode=plan` | `YOLO26_PLAN_TASKS=det,seg` (crop grounding for planOverlays) |

Pose is **opt-in** (`YOLO26_POSE_ENABLED=true` or an explicit `pose` in a task
list) and never runs on every frame by default. Warmup loads only the det
model; the seg model lazy-loads the first time a Build/Plan crop needs it (and
re-runs only on extraction frames / crop changes / every `YOLO26_SEG_EVERY_N`
frames). Crop segmentation produces `maskSource: "yolo26-seg"` +
`maskContour` (used as the blueprint `outline`); on any failure the existing
fallback contour takes over and the frame is still returned. Plan Mode uses the
crop detections for visual grounding (`highlight`/`callout`/`arrow`/`target`/
`ghost-position`/`warning-zone`/`step-marker` overlays, all normalized 0..1).

Other YOLO26 env: `YOLO26_DET_MODEL_ID` / `YOLO26_SEG_MODEL_ID` /
`YOLO26_POSE_MODEL_ID` (default `yolo26n*.pt`; legacy `YOLO26_MODEL_ID` still
honored), `YOLO26_DEVICE`, `YOLO26_IMG_SIZE`, `YOLO26_CONF`, `YOLO26_CACHE_DIR`
(default `/runpod-volume/models/yolo26`). det+seg weights are best-effort
pre-baked into `/app/models/yolo26` and otherwise resolved from the cache dir
or auto-downloaded by Ultralytics into it. Model-load status per task is in
`GET /debug/state` under `backend_status.yolo26`.

### Generic detector config + stronger demo profile (A1)

The detector config also accepts **generic `YOLO_*` env names** that take
**precedence** over the legacy `YOLO26_*` names (which still work unchanged when
the generic ones are unset). `VISION_BACKEND=ultralytics` routes the YOLO family
(YOLO11 / YOLO26 / YOLOE) through `ultralytics_loader.py` (a thin adapter over
`yolo26_loader`). This lets you test a stronger detector for better recall
**without editing the image**:

```env
VISION_BACKEND=ultralytics
YOLO_DET_MODEL_ID=yolo11s.pt
YOLO_IMG_SIZE=960
YOLO_CONF=0.10
YOLO_IOU=0.60
YOLO_MAX_DETECTIONS=300
```

The resolved active backend/model/knobs are visible in `GET /debug/state` under
`effective_config.active_detector` (`active_backend`, `active_model_id`,
`img_size`, `conf`, `iou`, `max_detections`, `weights_source`). `/detect` **and**
`/ws/vision` both use this resolved active-backend config (streaming no longer
falls back to stale EdgeCrafter defaults). Precedence per value is
**payload → `YOLO_*` → `YOLO26_*` → default**.

> **Licensing (do not skip):** `yolo11s.pt` / Ultralytics weights are **AGPL-3.0**
> — fine for testing/demo, but **not** a silent commercial production default
> without an Ultralytics Enterprise License. For commercial production prefer
> **DEIM / DEIMv2** or **RT-DETR** (Apache-2.0). Every candidate is recorded in
> `model_registry.example.json` with `license_status` + `commit_weights:false`;
> weights are never committed or baked into the image (runtime-resolved).

## Architecture

This worker runs as a **long-running FastAPI/uvicorn HTTP server** on a RunPod
load-balancing endpoint (not a serverless queue worker).

Pattern adapted from [Kingo333/fluxrt-serverless](https://github.com/Kingo333/fluxrt-serverless).

### Why live-server mode?

The previous queue-based RunPod serverless handler was fine for static-image
validation but caused repeated start/stop cycles and queued-but-never-processed
jobs because the health probe had no route to answer while the model was loading.

The live-server architecture solves this by:

1. Starting the HTTP server **immediately** before any model is loaded.
2. Returning 200 from `/health` and `/ping` at all times (no model dependency).
3. Loading the DEIMv2 model in a **background thread** so the server stays responsive.
4. Using `bootstrap.py` as a failsafe launcher -- if `server.py` crashes during
   import, a minimal fallback FastAPI app is started on the same port so RunPod's
   health probe can still surface the real error.

## Model loading (official DEIMv2, not transformers Auto classes)

DEIMv2 is **not** a standard Transformers `AutoImageProcessor` /
`AutoModelForObjectDetection` model. The official checkpoints
(e.g. `Intellindust/DEIMv2_DINOv3_S_COCO`) are pushed to the Hub via
`huggingface_hub.PyTorchModelHubMixin` and contain only `config.json` +
`model.safetensors` -- there is **no** `preprocessor_config.json` and no custom
HF modeling code on the Hub.

The architecture lives in the upstream GitHub repo
([Intellindust-AI-Lab/DEIMv2](https://github.com/Intellindust-AI-Lab/DEIMv2)),
cloned into `/opt/DEIMv2` and placed on `PYTHONPATH` by the Dockerfile. The
worker loads the model with the official custom class:

```python
from official_deimv2_loader import load_official_deimv2
model, device, cls = load_official_deimv2("Intellindust/DEIMv2_DINOv3_S_COCO")
```

See `official_deimv2_loader.py` for the loader, preprocessing
(`Resize((640, 640))` + `ToTensor()`), and postprocessing (normalized boxes).

## Routes

| Method | Path | Notes |
|--------|------|-------|
| GET | `/health` | Returns immediately, **no model required** |
| GET | `/ping` | Alias for `/health` |
| GET | `/debug/startup` | Env info, disk, startup log. Add `?deep=true` for torch/CUDA + import diagnostics |
| POST | `/debug/model-load` | Attempt model load only (official loader). Returns structured `ok`/`backend`/`model_class` or traceback |
| POST | `/warmup` | Trigger background model load. Add `?wait=true` to block until ready |
| POST | `/detect` | Run DEIMv2 inference. Returns 503 if model is not ready yet |

### `/detect` request

```json
{
  "image_b64": "<base64-encoded JPEG or PNG>",
  "conf": 0.35,
  "img_size": 640,
  "classes": [0, 2, 7]
}
```

## Build Mode (lightweight blueprint processing)

Build Mode is a **CPU-only**, additive feature, **fully separate** from the
EdgeCrafter `/detect` pipeline. It never loads a model, never triggers warmup,
and never touches the GPU. MediaPipe hand tracking runs **client-side in the
app**; the worker *consumes* the app's `handLandmarks` + `gesture` (it does not
re-run hand tracking) and turns a selected-crop image into a lightweight,
replayable blueprint:

```
selected crop -> grayscale -> blur -> Canny edges -> contours ->
largest contour -> approxPolyDP outline -> normalized 0..1 points ->
anchors -> hand landmarks (mapped to crop-local) -> step markers -> JSON frames
```

State is **in-memory JSON keyframes only** (no images, no video), with a
per-session cap (240 frames) and TTL cleanup (~45 min). The image work runs in a
worker thread, so it never blocks `/detect` or `/health`.

### Build Mode routes

| Method | Path | Notes |
|--------|------|-------|
| POST | `/build/session/start` | Open a session, returns `session_id` |
| POST | `/build/session/lock` | Lock the selected region for the session |
| POST | `/build/session/frame` | Process one selected-crop keyframe → `blueprint_frame` |
| POST | `/build/session/finish` | Close the session, returns a replay id |
| GET | `/build/session/{session_id}/replay` | Return the stored JSON keyframes |

### `/build/session/frame` request (selected crop only)

```json
{
  "sessionId": "build_...",
  "frameId": "f-0",
  "timestampMs": 1234,
  "selectedRegion": { "x": 0.1, "y": 0.2, "w": 0.4, "h": 0.3 },
  "image_b64": "<base64 JPEG/PNG of the selected crop>",
  "handLandmarks": [{ "x": 0.42, "y": 0.55, "role": "index-tip" }],
  "gesture": { "type": "pinch", "active": true, "strength": 0.8 }
}
```

The response wraps a camelCase `blueprint_frame` (`outline`, `anchors`,
`sparsePoints`, `handLandmarks`, `stepMarkers`, `gesture`). Hand landmarks and
step markers are mapped into **selected-crop-local** coordinates
(`local = (landmark - region_origin) / region_size`) so they line up with the
outline. An active pinch/grab gesture adds a step marker at the index fingertip.

Structured errors look like `{ "ok": false, "error": "<code>" }` — e.g.
`missing_image_b64`, `invalid_base64`, `decode_failure`, `unknown_session`,
`missing_session_id`, `invalid_selected_region`, `too_many_frames`,
`payload_too_large`, `processing_failure`.

Build Mode is **not** Gaussian splatting, a dense point cloud, a full 3D scan,
server-side MediaPipe, or video storage (those are explicitly out of scope).

### BlueprintFrame v2 (Build / Plan workflow)

`/build/session/frame` returns a **BlueprintFrame v2**. All v1 fields are kept
and every v2 field is optional, so the old app contract still works. New fields:
`version`, `workflowMode` (`"build"` | `"plan"`, default `"build"`),
`maskContour` + `maskSource`, `sourceAssetId`, rule-based `aiNotes` /
`instruction` / `nextAction` / `safetyWarning` / `qualityCheck` /
`activityLabel` / `importance`, and (Plan mode) `planSteps` +
`currentPlanStepIndex`. `workflowMode` resolves from the payload, then the
session (set at `start` / `lock`), defaulting to `"build"`:

- **build** — the user is doing the work; the worker documents activity into notes.
- **plan** — the user wants guidance; the worker returns suggested steps / next actions.

**Plan Mode is intent-driven.** Send `userIntent` (on `start` / `lock` or per frame):

```json
{ "workflowMode": "plan", "userIntent": { "taskType": "build", "text": "...", "confirmed": true } }
```

If `confirmed` is false (or `userIntent` is missing) the worker asks the user to
confirm rather than guessing the task. Once confirmed it returns `detectedIntent`,
task-specific `planSteps` / `instruction` / `nextAction` / `safetyWarning` /
`qualityCheck`, cautious `aiNotes`, and **`planOverlays`** — visual guidance the
app can render: `arrow` (with `from`/`to`), `target`, `highlight`,
`ghost-position`, `warning-zone`. `taskType`
(`identify` | `inspect` | `build`/`assemble` | `repair`/`troubleshoot`) controls
the guidance, and electrical / high-risk wording triggers safety-first warnings
(isolate / de-energize, qualified handling) — never live-electrical instructions.
This is rule-based (no VLM/LLM): mask center = main object, anchors = inspection
points, hand/pinch = active work point, low confidence = ask for clarification.

Segmentation is config-driven and CPU-only by default:

| Env | Default | Notes |
|-----|---------|-------|
| `BUILD_SEGMENTATION_BACKEND` | `fallback` | `none` \| `fallback` (Canny contour) \| `sam2` (optional, lazy) |
| `BUILD_MASK_OUTPUT` | `contour` | `contour` \| `mask_thumbnail` \| `none` |
| `BUILD_SEGMENT_ON_EXTRACT` | `true` | segment on extraction frames |
| `BUILD_SEGMENT_EVERY_N` | `3` | otherwise segment every Nth frame and reuse the mask in between |

SAM2 is **optional and never a hard dependency**: when `BUILD_SEGMENTATION_BACKEND=sam2`
and SAM2 is unavailable, the worker logs and falls back to the contour pipeline.
Keep SAM2 disabled until the fallback is proven stable.

### Plan Mode selected-crop context (geometry first, reasoning on the app side)

For `/build/session/frame` with `workflowMode: "plan"`, the worker returns
richer **selected-crop** context so the app + its Supabase/DeepSeek Edge Function
can reason (the worker never calls DeepSeek). All fields are optional/additive:
`selectedLabel`, `cropEntities`, `cropSegments`, `suggestedGoals`,
`virtualBlueprintPoints` (rule-based 2D points -- roles `anchor` /
`alignment-point` / `target-position` / `connection-point` / `inspection-point`
/ `warning-point`, capped at 12, all x/y clamped 0..1), `planContext`
(`selectedLabel`/`objectCount`/`hasMultipleParts`/`likelyUse`/`contextSource`/
`warnings`), and optional `depthPoints` / `knownPartPose` / `assemblyState`.
Electronics/PCB scenes get connector/edge points + an "ensure unpowered" safety
hint. `/detect` stays det-only and fast; segmentation runs only on selected
Build/Plan crops (lazy-loaded). Build Mode is unchanged (only a cheap
`selectedLabel`).

Optional adapters are **disabled safe stubs** by default and degrade to a clear
signal if enabled without a backend (no extra image deps; Point-E is never run):

| Env | Default | Role |
|-----|---------|------|
| `PLAN_CONTEXT_ENABLED` | `true` | rule-based crop context (above) |
| `PLAN_DEPTH_ENABLED` / `PLAN_DEPTH_BACKEND` | `false` / `none` | Depth Anything pseudo-depth on the crop only |
| `PLAN_OPEN_VOCAB_ENABLED` | `false` | GroundingDINO/Grounded-SAM inspection path |
| `PLAN_KNOWN_PART_POSE_ENABLED` | `false` | FoundationPose/MegaPose for known parts |
| `PLAN_ASSEMBLY_STATE_ENABLED` | `false` | IndustReal/ASDF step-state |

`/debug/state` reports a `plan_context` block with all of these.

## Risk-aware perception (deterministic engine + tracking)

The `risk/` package adds a **deterministic, additive** risk layer on the live
path (`/detect` HSE loop + `/ws/vision`). It is the **safety signal**: pure
Python rules over the detector's entities, a per-session tracker, and a
deterministic scene graph — **no GPU, no weights, no VLM**. It is gated by
`RISK_ENGINE_ENABLED` (**default off**): when disabled, `/detect` and
`/ws/vision` responses are byte-for-byte the legacy shape. When enabled, the
response gains a `schema_version: "risk.v1"` plus additive `risk_engine`,
`tracks`, `scene_graph`, `risks`, and `scene_risks` fields. A risk failure
degrades to the normal detection result with a `warning` — never a 500.

```
detector entities → per-session tracker (IoU/centroid) → scene graph (geometry)
  → 15 deterministic rules → 5×5 risk matrix (severity×likelihood) → controls
  (hierarchy of controls) → provenance stamp → scored RiskItem[]
```

- **15 rules** (`R01..R15`): PPE (hardhat/vest/gloves), pedestrian↔vehicle /
  ↔forklift separation, fire, smoke, blocked exit, object-near-edge,
  working-at-height (ladder), scaffold, spill/slip, exposed electrical,
  overhead suspended load, open-hole/fall — each fires only for the classes the
  detector actually provides.
- **Risk matrix** is a versioned JSON profile (`RISK_MATRIX_PROFILE`,
  default `risk/risk_matrix_profile.json`): `score = severity × likelihood`,
  banded GREEN/YELLOW/ORANGE/RED. The profile is **validated on load** (bands
  monotonic, contiguous, full coverage) — a malformed matrix raises.
- **Per-session tracking** is keyed by `session_id` (`/detect`) / `camera_id`
  (`/ws/vision`) — never global, so two camera streams never cross-contaminate
  `track_id`s. State has TTL eviction + a bounded active-session count, reusing
  the Build Mode session sweep pattern.
- **Provenance**: every risk carries `produced_by="risk_engine"`,
  `rule_id`, `model_version`, `timestamp_ms`, and `requires_human_review=false`
  (the deterministic engine is authoritative; AI drafts will set it true).
- **Privacy** (`risk/privacy.py`): blurs persons/faces before any frame is
  persisted or sent to a VLM (`PRIVACY_BLUR_ENABLED`, default off). Hazards and
  conditions only — never emotion/identity/biometric inference.
- **Validation gate** (`validation/run_validation.py`): runs the engine over
  synthetic hazard scenarios (no GPU/weights) and exits non-zero if
  critical-hazard recall drops below `VALIDATION_MIN_RECALL_CRITICAL` (0.90).

| Env | Default | Notes |
|-----|---------|-------|
| `RISK_ENGINE_ENABLED` | `false` | master switch; off = legacy response shape |
| `RISK_TRACKING_ENABLED` / `RISK_SCENE_GRAPH_ENABLED` | `true` | sub-stage toggles |
| `RISK_MATRIX_PROFILE` | `/app/risk/risk_matrix_profile.json` | versioned 5×5 matrix |
| `SESSION_TTL_MS` / `SESSION_MAX_ACTIVE` | `30000` / `64` | per-session tracker memory |
| `RISK_NEAR_THRESHOLD` | `0.12` | normalized centroid distance for "near" |
| `PRIVACY_BLUR_ENABLED` | `false` | blur persons before VLM/persist (later PR) |

`/debug/state` reports a `risk_engine` block (flags, matrix profile/version,
active sessions, and the last evaluation's risk/alert counts + highest level).
## Event-driven reasoning (Qwen-VL) + open-vocab scanner (GroundingDINO)

`risk/vlm_reasoner.py` is a **real** Qwen-VL (and optional DeepSeek-VL2) reasoner
behind `POST /reason`, plus a non-blocking trigger from `/detect`. It is **not**
the safety authority — the deterministic engine is. The VLM only **explains /
verifies / drafts** *after* the deterministic engine produces a candidate, and
its output is always an **AI draft**: `produced_by="vlm_reasoner"`,
`requires_human_review=true`, `should_alert=false` (enforced by the schema, not
trusted from the model).

- **Never per-frame.** `/detect` calls `maybe_trigger()`: rate-limited
  (`REASONER_MIN_INTERVAL_MS`), fired only at/above `REASONER_TRIGGER_LEVEL`
  (default `ORANGE`), run on a bounded background executor. `/detect` **never
  waits** — it attaches the most recent cached draft (if any) as `scene_risks`
  plus a `reasoner_status` and returns. If the VLM is slow/unavailable the live
  loop is unaffected.
- **Real but lazy.** `torch`/`transformers` import only on first model use;
  Qwen weights resolve at runtime into `REASONER_CACHE_DIR`/the HF cache and are
  **never baked into the image or downloaded at Docker build**. Missing
  deps/weights → `reasoner_status="unavailable"`; over-budget → `"timeout"`;
  disabled → `"disabled"` — it never raises into the request path.
- **`REASONER_MODE=mock`** gives a CPU, weight-free implementation of the full
  `/reason` contract so the app can integrate before a GPU/Qwen deployment.
- **Privacy.** When `PRIVACY_BLUR_ENABLED`, the frame is blurred (persons)
  **before** it is passed to the model — no un-blurred frame reaches the VLM.

`POST /reason` validates strictly against `risk/reason_schema.py`
(`schema_version: "reason.v1"`). `POST /scan` is the optional open-vocabulary
**GroundingDINO** scanner (`risk/grounding_dino_scanner.py`): disabled by
default, throttled (never per-frame), output is **candidate-only**
(`produced_by="open_vocab_scanner"`, `candidate_only=true`,
`requires_human_review=true`) and can **never** trigger an official HSE alert.

| Env | Default | Notes |
|-----|---------|-------|
| `VLM_REASONER_ENABLED` | `false` | master switch for `/reason` + `/detect` trigger |
| `REASONER_MODE` | `qwen_vl` | `qwen_vl` \| `deepseek_vl2` \| `mock` |
| `QWEN_VL_MODEL_ID` | `Qwen/Qwen2.5-VL-7B-Instruct` | runtime-resolved; 3B for low VRAM |
| `REASONER_TRIGGER_LEVEL` / `REASONER_MIN_INTERVAL_MS` | `ORANGE` / `5000` | when/how often to trigger |
| `REASONER_TIMEOUT_MS` | `8000` | hard cap; over-budget → `reasoner_status:"timeout"` |
| `REASONER_QUANTIZATION` | `4bit` | `none` \| `8bit` \| `4bit` (GPU memory) |
| `OPEN_VOCAB_SCANNER_ENABLED` | `false` | GroundingDINO scanner (candidate-only) |

`/debug/state` reports `reasoner` and `open_vocab_scanner` blocks. **Deferred to
a later PR:** fine-tuning, running the VLM every frame, and using the VLM as the
alert authority (all explicitly out of scope).

## Docker image

Built and pushed to GitHub Container Registry on every push to `main`:

```
ghcr.io/gabe3laka/safelens-deimv2-worker:latest
```

## RunPod configuration

### Endpoint type

Use **HTTP service** (load-balancing), **not** serverless queue.

Set the container port to match the `PORT` env var (default: `8000`).

### Health probe

Configure RunPod to probe `GET /health` (or `GET /ping`).

### Key environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `8000` | Uvicorn listen port |
| `AUTO_WARMUP` | `true` | Start model load immediately on container start |
| `SKIP_WARMUP` | `false` | Skip model load entirely (diagnostic mode) |
| `WARMUP_TIMEOUT_S` | `600` | Seconds before warmup is marked failed |
| `DEIMV2_BACKEND` | `official-deimv2-hf` | `official-deimv2-hf` (default) or `transformers-fallback` (DETR, pipeline validation only) |
| `DEIMV2_MODEL_ID` | `Intellindust/DEIMv2_DINOv3_S_COCO` | Official DEIMv2-S HuggingFace model ID |
| `DEIMV2_DEVICE` | `cuda` | Inference device (`cuda` or `cpu`) |
| `DEIMV2_CONF` | `0.35` | Default confidence threshold |
| `DEIMV2_IMG_SIZE` | `640` | Square resize before inference (DEIMv2-S eval size) |
| `HF_HOME` | `/runpod-volume/.cache/huggingface` | HuggingFace cache (mount volume here) |
| `HF_TOKEN` | _(unset)_ | Optional; only for private/gated mirrors. Never logged or exposed. |

### RunPod env block (copy/paste)

```
PORT=8000
PORT_HEALTH=8000
SKIP_WARMUP=true
AUTO_WARMUP=false
DEIMV2_DEVICE=cuda
DEIMV2_BACKEND=official-deimv2-hf
DEIMV2_MODEL_ID=Intellindust/DEIMv2_DINOv3_S_COCO
DEIMV2_CONF=0.35
DEIMV2_IMG_SIZE=640
HF_HOME=/runpod-volume/.cache/huggingface
TRANSFORMERS_CACHE=/runpod-volume/.cache/huggingface
```

## Files

| File | Purpose |
|------|---------|
| `server.py` | FastAPI app -- all routes, warmup logic, state |
| `bootstrap.py` | Failsafe launcher -- starts server.py, falls back to minimal app on error |
| `deimv2_infer.py` | Backend dispatch + inference (official DEIMv2 / transformers fallback) |
| `official_deimv2_loader.py` | Official DEIMv2 PyTorchModelHubMixin loader + pre/post-processing |
| `handler.py` | Legacy RunPod serverless handler (kept for reference) |
| `schema.py` | Pydantic request/response models |
| `risk/` | Deterministic risk engine (`tracking`, `scene_graph`, `risk_matrix`, `controls`, `provenance`, `privacy`, `risk_engine`, `risk_schema` + `risk_matrix_profile.json`) **and** the event-driven reasoning layer (`vlm_reasoner`, `grounding_dino_scanner`, `open_vocab_scanner`, `reason_schema`) |
| `validation/` | CPU-only risk-engine quality gate (`run_validation.py` + synthetic `scenarios/`) |

## Diagnostic mode

Set `SKIP_WARMUP=true` to start the server without loading the model.
`/health` and `/debug/startup` work immediately. Useful for smoke-testing
the container boot without waiting for model download.

## Building locally

```bash
docker build -t safelens-deimv2-worker:local .
docker run --rm -p 8000:8000 -e SKIP_WARMUP=true safelens-deimv2-worker:local
curl http://localhost:8000/health
curl http://localhost:8000/debug/startup
```
