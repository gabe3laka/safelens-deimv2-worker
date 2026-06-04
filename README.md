# safelens-deimv2-worker

> **Sprint 4A-DEIMv2** — Standalone Dockerized DEIMv2 RunPod serverless worker  
> Part of the **Eagle Vision 2 / SafeLens** HSE object-detection pipeline.

---

## What this repo is

This is the **RunPod serverless worker** that runs [DEIMv2](https://github.com/Intellindust-AI-Lab/DEIMv2) GPU inference for Eagle Vision 2.

It is completely separate from the Eagle Vision 2 frontend repo (`gabe3laka/HSE-eagle-vision-2`).  
RunPod builds and runs containers from **this** repo.

### Architecture

```
Browser camera frame
  -> Eagle Vision 2 BackendVisionDetector (frontend, dry-run mode)
  -> Supabase Edge Function proxy          (hides RunPod key from browser)
  -> RunPod DEIMv2 worker  (this repo)
  -> DEIMv2 returns normalised entity boxes
  -> Eagle Vision 2 displays entities in dev/debug mode
  -> No DEIMv2 safety alerts yet (Sprint 4A is dry-run)
```

---

## Repo structure

```
safelens-deimv2-worker/
  README.md              This file
  Dockerfile             CUDA/PyTorch base; clones DEIMv2 at build time
  requirements.txt       Python deps (runpod, Pillow, pydantic, torch, ...)
  handler.py             RunPod serverless entry point
  deimv2_infer.py        DEIMv2 model loading + inference wrapper
  schema.py              Pydantic request / response models
  .gitignore             Excludes weights, checkpoints, cache
  scripts/
    smoke_test.py        Local + live endpoint smoke test
  examples/
    request.example.json  Sample RunPod request payload
    response.example.json Sample RunPod response payload
```

---

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `DEIMV2_MODEL_ID` | `Intellindust-AI-Lab/DEIMv2-S` | HuggingFace model id |
| `DEIMV2_DEVICE` | `cuda` | `cuda` or `cpu` |
| `DEIMV2_CONF` | `0.35` | Confidence threshold (0..1) |
| `DEIMV2_IMG_SIZE` | `640` | Shorter-side resize before inference |
| `HF_HOME` | `/runpod-volume/.cache/huggingface` | HuggingFace cache dir |

### Available model sizes

| Model | AP (COCO) | Params | Use case |
|---|---|---|---|
| `DEIMv2-N` | 43.0 | 3.6 M | Ultra-light / CPU |
| `DEIMv2-S` | **50.9** | 9.7 M | **Recommended default** |
| `DEIMv2-M` | 53.0 | 18.1 M | Higher accuracy |
| `DEIMv2-L` | 56.0 | 32.2 M | Best accuracy |

---

## Request / response schema

### Request (sent by Supabase Edge Function proxy)

```json
{
  "input": {
    "image_b64": "<base64-encoded JPEG or PNG>",
    "conf":      0.35,
    "img_size":  640,
    "classes":   null
  }
}
```

### Response

```json
{
  "entities": [
    {
      "label":      "person",
      "class_id":   0,
      "confidence": 0.91,
      "bbox": { "x": 0.12, "y": 0.05, "w": 0.18, "h": 0.72 }
    }
  ],
  "inference_ms": 48.3,
  "model":        "deimv2-s",
  "img_w":        1280,
  "img_h":        720
}
```

All bounding boxes are normalised to **0..1** relative to the original image.

---

## Building and deploying

### 1. Build locally

```bash
docker build -t safelens-deimv2-worker:latest .
```

### 2. Test locally (CPU, no GPU required)

```bash
docker run --rm -e DEIMV2_DEVICE=cpu -e DEIMV2_MODEL_ID=Intellindust-AI-Lab/DEIMv2-N \
    safelens-deimv2-worker:latest python scripts/smoke_test.py --image examples/test.jpg
```

### 3. Push to Docker Hub

```bash
docker tag safelens-deimv2-worker:latest <your-dockerhub>/safelens-deimv2-worker:latest
docker push <your-dockerhub>/safelens-deimv2-worker:latest
```

### 4. Deploy to RunPod Serverless

1. Go to [RunPod Serverless](https://www.runpod.io/console/serverless)
2. Create a new endpoint → "Custom" → paste your Docker image URL
3. Set environment variables (see table above)
4. Set a GPU tier (RTX 3090 or A4000 recommended for DEIMv2-S)
5. Mount a network volume at `/runpod-volume` to cache model weights

---

## Smoke testing

```bash
# Local test (requires Python deps installed)
pip install -r requirements.txt
python scripts/smoke_test.py --image path/to/test.jpg

# Against a live RunPod endpoint
export RUNPOD_ENDPOINT_ID=<your-endpoint-id>
export RUNPOD_API_KEY=<your-api-key>    # Never commit this!
python scripts/smoke_test.py --endpoint --image path/to/test.jpg
```

---

## Model weights

**Model weights are never committed to this repo.**

Weights are downloaded automatically from HuggingFace hub on first run and cached in `HF_HOME`.  
Mount a RunPod network volume at `/runpod-volume` to persist the cache across container restarts.

---

## Sprint 4A notes (dry-run)

- DEIMv2 returns **entities** (normalised boxes + labels) only
- Eagle Vision 2 displays entities in **dev/debug mode** only
- **No DEIMv2 safety alerts are emitted in Sprint 4A**
- MediaPipe Pose continues to handle `unsafe_lift`, `person_proximity`, `restricted_zone`
- DEIMv2 will drive `ppe_missing`, `forklift_proximity`, `blocked_exit` in Sprint 4B+

---

## Security

- **Never expose your RunPod API key in the browser frontend**
- The frontend calls a Supabase Edge Function which holds the key as a secret
- See `supabase/functions/deimv2-proxy/` in the Eagle Vision 2 repo for the proxy
