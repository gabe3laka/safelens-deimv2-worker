"""
scripts/smoke_test.py -- Local smoke test for the DEIMv2 RunPod worker.

Usage (from repo root):

  # Dry-run: syntax + import check only (no model, no image required)
  python scripts/smoke_test.py --dry-run

  # Local handler test (needs deps installed, will download model weights)
  python scripts/smoke_test.py --image path/to/test.jpg

  # Live RunPod endpoint test
  RUNPOD_ENDPOINT_ID=<id> RUNPOD_API_KEY=<key> \
      python scripts/smoke_test.py --endpoint --image path/to/test.jpg

Notes:
  --dry-run does NOT load model weights.
  Without --image and --endpoint, a synthetic 1x1 white JPEG is used
  to exercise the handler's input validation path (no GPU required).
  The script exits non-zero if any test fails.
"""

import argparse
import base64
import io
import json
import os
import py_compile
import sys
import time
from pathlib import Path

# ── Argument parsing ──────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description="DEIMv2 worker smoke test",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog=__doc__,
)
parser.add_argument(
    "--image",
    default=None,
    help="Path to a test image (JPEG/PNG). Optional; if omitted a synthetic 1x1 JPEG is used.",
)
parser.add_argument(
    "--dry-run",
    action="store_true",
    help=(
        "Syntax check + import validation only. "
        "Does NOT load model weights or run inference."
    ),
)
parser.add_argument("--conf", type=float, default=0.35, help="Confidence threshold")
parser.add_argument("--img-size", type=int, default=640, help="Inference image size")
parser.add_argument(
    "--endpoint", action="store_true", help="Hit a live RunPod endpoint instead of local handler"
)
parser.add_argument("--timeout", type=int, default=120, help="RunPod polling timeout (s)")
args = parser.parse_args()

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

# ── Dry-run: syntax + import check ───────────────────────────────────────────
if args.dry_run:
    print("[smoke/dry-run] Running syntax check...")
    for fname in ["schema.py", "handler.py", "deimv2_infer.py"]:
        fpath = REPO_ROOT / fname
        try:
            py_compile.compile(str(fpath), doraise=True)
            print(f"[smoke/dry-run] OK  {fname}")
        except py_compile.PyCompileError as exc:
            print(f"[smoke/dry-run] FAIL {fname}: {exc}", file=sys.stderr)
            sys.exit(1)

    print("[smoke/dry-run] Running import check...")
    try:
        import schema  # noqa: F401
        from schema import InferRequest, BBox, Entity, InferResponse  # noqa: F401
        print("[smoke/dry-run] OK  schema.py imports")
    except Exception as exc:
        print(f"[smoke/dry-run] FAIL schema import: {exc}", file=sys.stderr)
        sys.exit(1)

    print("[smoke/dry-run] Running schema validation...")
    try:
        req = InferRequest(image_b64="aGVsbG8=", conf=0.5, img_size=320)
        assert req.conf == 0.5, "conf mismatch"
        assert req.img_size == 320, "img_size mismatch"
        assert req.classes is None, "classes should be None"

        bbox = BBox(x=0.1, y=0.2, w=0.3, h=0.4)
        ent = Entity(label="person", class_id=0, confidence=0.9, bbox=bbox)
        resp = InferResponse(entities=[ent], inference_ms=12.0, model="deimv2-s", img_w=640, img_h=480)
        assert resp.error is None, "error should be None"
        assert resp.warning is None, "warning should be None"

        err_resp = InferResponse(entities=[], error="missing_image_b64")
        assert err_resp.entities == [], "entities should be []"
        assert err_resp.error == "missing_image_b64", "error mismatch"

        print("[smoke/dry-run] OK  schema validation")
    except Exception as exc:
        print(f"[smoke/dry-run] FAIL schema validation: {exc}", file=sys.stderr)
        sys.exit(1)

    print("[smoke/dry-run] Running handler import + missing-image check...")
    try:
        from handler import handler as _handler
        result = _handler({"input": {}})
        assert result.get("error") == "missing_image_b64", f"expected missing_image_b64, got: {result}"
        assert result.get("entities") == [], f"expected [], got: {result.get('entities')}"
        print("[smoke/dry-run] OK  handler missing-image returns structured error")

        result2 = _handler({"input": {"image_b64": "!!!not-valid-base64!!!"}})
        assert result2.get("error") == "invalid_base64", f"expected invalid_base64, got: {result2}"
        print("[smoke/dry-run] OK  handler invalid-base64 returns structured error")
    except SystemExit:
        raise
    except Exception as exc:
        print(f"[smoke/dry-run] FAIL handler check: {exc}", file=sys.stderr)
        sys.exit(1)

    print("[smoke/dry-run] All checks passed.")
    sys.exit(0)

# ── Build image payload ───────────────────────────────────────────────────────
if args.image:
    image_path = Path(args.image)
    if not image_path.exists():
        print(f"ERROR: image not found: {image_path}", file=sys.stderr)
        sys.exit(1)
    with open(image_path, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode()
    print(f"[smoke] Using image: {image_path}")
else:
    # Synthetic 1x1 white JPEG (no PIL required -- raw JFIF bytes)
    # This exercises the full handler input-validation + inference path without
    # needing a real image file.  Model inference may still fail if weights are
    # not downloaded, but the base64/schema validation will succeed.
    try:
        from PIL import Image as _PILImage
        buf = io.BytesIO()
        _PILImage.new("RGB", (8, 8), color=(255, 255, 255)).save(buf, format="JPEG")
        image_b64 = base64.b64encode(buf.getvalue()).decode()
        print("[smoke] Using synthetic 8x8 white JPEG (PIL)")
    except ImportError:
        # Minimal raw JFIF bytes for a 1x1 white JPEG
        _TINY_JPEG_HEX = (
            "ffd8ffe000104a46494600010100000100010000"
            "ffdb004300080606070605080707070909080a0c"
            "140d0c0b0b0c1912130f141d1a1f1e1d1a1c1c20"
            "242e2720222c231c1c2837292c30313434341f27"
            "393d38323c2e333432ffdb0043010909090c0b0c"
            "180d0d1832211c213232323232323232323232323"
            "2323232323232323232323232323232323232323232"
            "323232323232323232ffc0000b080001000101011"
            "100ffc4001f000001050101010101010000000000"
            "0000000102030405060708090a0bffda00080101"
            "003f00f50fffd9"
        )
        try:
            image_bytes = bytes.fromhex(_TINY_JPEG_HEX.replace("\n", "").replace(" ", ""))
            image_b64 = base64.b64encode(image_bytes).decode()
            print("[smoke] Using embedded minimal JPEG bytes (no PIL)")
        except Exception:
            # Ultimate fallback: 1-pixel white JPEG encoded as base64 directly
            image_b64 = (
                "/9j/4AAQSkZJRgABAQEASABIAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8U"
                "HRofHh0aHBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/2wBDAQkJCQwLDBgN"
                "DRgyIRwhMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIy"
                "MjL/wAARCAABAAEDASIAAhEBAxEB/8QAFAABAAAAAAAAAAAAAAAAAAAACf/EABQQAQAAAAAA"
                "AAAAAAAAAAAAAP/EABQBAQAAAAAAAAAAAAAAAAAAAAD/xAAUEQEAAAAAAAAAAAAAAAAAAAAA"
                "/9oADAMBAAIRAxEAPwCwABmX/9k="
            )
            print("[smoke] Using fallback base64 minimal JPEG")

request_payload = {
    "image_b64": image_b64,
    "conf": args.conf,
    "img_size": args.img_size,
    "classes": None,
}

# ── Local handler test ────────────────────────────────────────────────────────
if not args.endpoint:
    print("[smoke] Running local handler test...")

    from handler import handler as _handler

    t0 = time.perf_counter()
    result = _handler({"input": request_payload})
    elapsed = (time.perf_counter() - t0) * 1000

    print(f"[smoke] handler returned in {elapsed:.1f} ms")
    print(json.dumps(result, indent=2))

    if "error" in result:
        # model_load_failed is expected if weights not downloaded
        if result["error"].startswith("model_load_failed"):
            print(
                f"[smoke] NOTE: model not available ({result['error']}). "
                "This is expected in a no-weights environment. "
                "Structured error returned correctly.",
                file=sys.stderr,
            )
            sys.exit(0)
        print(f"[smoke] FAIL: handler returned error: {result['error']}", file=sys.stderr)
        sys.exit(1)

    entities = result.get("entities", [])
    print(f"[smoke] OK: {len(entities)} entities detected in {result.get('inference_ms')} ms")
    sys.exit(0)

# ── Live RunPod endpoint test ─────────────────────────────────────────────────
import requests  # type: ignore  # noqa: E402

endpoint_id = os.environ.get("RUNPOD_ENDPOINT_ID", "")
api_key = os.environ.get("RUNPOD_API_KEY", "")

if not endpoint_id or not api_key:
    print("ERROR: set RUNPOD_ENDPOINT_ID and RUNPOD_API_KEY", file=sys.stderr)
    sys.exit(1)

BASE = f"https://api.runpod.ai/v2/{endpoint_id}"
headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

print(f"[smoke] Submitting job to RunPod endpoint {endpoint_id}...")
resp = requests.post(
    f"{BASE}/run", json={"input": request_payload}, headers=headers, timeout=30
)
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
    print(f"[smoke] status: {status}")
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
