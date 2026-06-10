# safelens-deimv2-worker

SafeLens DEIMv2 detection worker for RunPod -- live HTTP server mode.

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
