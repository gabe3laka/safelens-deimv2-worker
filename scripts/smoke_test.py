"""
scripts/smoke_test.py -- Local smoke test for the DEIMv2 RunPod worker.

Usage (from repo root, with handler + deps installed):

  # Option 1: Direct handler test (no RunPod SDK needed)
  python scripts/smoke_test.py --image path/to/test.jpg

  # Option 2: Against a live RunPod endpoint
  RUNPOD_ENDPOINT_ID=<id> RUNPOD_API_KEY=<key> \
      python scripts/smoke_test.py --endpoint --image path/to/test.jpg

The script exits non-zero if inference fails or returns no entities.
"""

import argparse
import base64
import json
import os
import sys
import time
from pathlib import Path

# ── Argument parsing ──────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="DEIMv2 worker smoke test")
parser.add_argument("--image", required=True, help="Path to a test image (JPEG/PNG)")
parser.add_argument("--conf", type=float, default=0.35, help="Confidence threshold")
parser.add_argument("--img-size", type=int, default=640, help="Inference image size")
parser.add_argument("--endpoint", action="store_true", help="Hit a live RunPod endpoint")
parser.add_argument("--timeout", type=int, default=120, help="RunPod polling timeout (s)")
args = parser.parse_args()

image_path = Path(args.image)
if not image_path.exists():
    print(f"ERROR: image not found: {image_path}", file=sys.stderr)
    sys.exit(1)

# Encode image
with open(image_path, "rb") as f:
    image_b64 = base64.b64encode(f.read()).decode()

request_payload = {
    "image_b64": image_b64,
    "conf": args.conf,
    "img_size": args.img_size,
    "classes": None,
}

# ── Local handler test ────────────────────────────────────────────────────────
if not args.endpoint:
    print("[smoke] Running local handler test...")
    # Add parent dir (repo root) to path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from handler import handler as _handler

    t0 = time.perf_counter()
    result = _handler({"input": request_payload})
    elapsed = (time.perf_counter() - t0) * 1000

    print(f"[smoke] handler returned in {elapsed:.1f} ms")
    print(json.dumps(result, indent=2))

    if "error" in result:
        print(f"[smoke] FAIL: handler returned error: {result['error']}", file=sys.stderr)
        sys.exit(1)

    entities = result.get("entities", [])
    print(f"[smoke] OK: {len(entities)} entities detected in {result.get('inference_ms')} ms")
    sys.exit(0)

# ── Live RunPod endpoint test ─────────────────────────────────────────────────
import requests  # type: ignore

endpoint_id = os.environ.get("RUNPOD_ENDPOINT_ID", "")
api_key = os.environ.get("RUNPOD_API_KEY", "")

if not endpoint_id or not api_key:
    print("ERROR: set RUNPOD_ENDPOINT_ID and RUNPOD_API_KEY", file=sys.stderr)
    sys.exit(1)

BASE = f"https://api.runpod.ai/v2/{endpoint_id}"
headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

print(f"[smoke] Submitting job to RunPod endpoint {endpoint_id}...")
resp = requests.post(f"{BASE}/run", json={"input": request_payload}, headers=headers, timeout=30)
resp.raise_for_status()
job = resp.json()
job_id = job["id"]
print(f"[smoke] Job id: {job_id}")

deadline = time.time() + args.timeout
while time.time() < deadline:
    time.sleep(3)
    status_resp = requests.get(f"{BASE}/status/{job_id}", headers=headers, timeout=10)
    status_resp.raise_for_status()
    data = status_resp.json()
    status = data.get("status")
    print(f"[smoke]   status: {status}")
    if status == "COMPLETED":
        output = data.get("output", {})
        print(json.dumps(output, indent=2))
        entities = output.get("entities", [])
        print(f"[smoke] OK: {len(entities)} entities, {output.get('inference_ms')} ms")
        sys.exit(0)
    if status in ("FAILED", "CANCELLED"):
        print(f"[smoke] FAIL: job {status}", file=sys.stderr)
        sys.exit(1)

print(f"[smoke] TIMEOUT after {args.timeout}s", file=sys.stderr)
sys.exit(1)
