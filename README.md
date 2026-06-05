# safelens-deimv2-worker

SafeLens DEIMv2 detection worker for RunPod — live HTTP server mode.

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
4. Using `bootstrap.py` as a failsafe launcher — if `server.py` crashes during
   import, a minimal fallback FastAPI app is started on the same port so RunPod's
   health probe can still surface the real error.

## Routes

| Method | Path | Notes |
|--------|------|-------|
| GET | `/health` | Returns immediately, **no model required** |
| GET | `/ping` | Alias for `/health` |
| GET | `/debug/startup` | Env info, disk, startup log. Add `?deep=true` for torch/CUDA details |
| POST | `/warmup` | Trigger background model load. Add `?wait=true` to block until ready |
| POST | `/detect` | Run DEIMv2 inference. Returns 503 if model is not ready yet |

### `/health` response (model not yet loaded)

```json
{
  "ok": true,
  "worker": "safelens-deimv2-worker",
  "mode": "live-server",
  "version": "0.2.0-live-server",
  "status": "cold",
  "model_loaded": false,
  "error": null
}
```

### `/detect` request

```json
{
  "image_b64": "<base64-encoded JPEG or PNG>",
  "conf": 0.35,
  "img_size": 640,
  "classes": [0, 2, 7]
}
```

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
| `DEIMV2_MODEL_ID` | `Intellindust-AI-Lab/DEIMv2-S` | HuggingFace model ID |
| `DEIMV2_DEVICE` | `cuda` | Inference device (`cuda` or `cpu`) |
| `DEIMV2_CONF` | `0.35` | Default confidence threshold |
| `DEIMV2_IMG_SIZE` | `640` | Shorter-side resize before inference |
| `HF_HOME` | `/runpod-volume/.cache/huggingface` | HuggingFace cache (mount volume here) |

## Files

| File | Purpose |
|------|---------|
| `server.py` | FastAPI app — all routes, warmup logic, state |
| `bootstrap.py` | Failsafe launcher — starts server.py, falls back to minimal app on error |
| `deimv2_infer.py` | DEIMv2 model loading and inference |
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
