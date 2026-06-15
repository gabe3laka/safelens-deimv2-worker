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
