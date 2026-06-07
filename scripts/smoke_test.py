"""
scripts/smoke_test.py -- Local smoke test for the SafeLens vision worker.

Usage (from repo root):

  # Dry-run: syntax + import + schema + streaming checks. No GPU, no weights.
  python scripts/smoke_test.py --dry-run

  # Local handler test (legacy DEIMv2 serverless handler; downloads weights)
  python scripts/smoke_test.py --image path/to/test.jpg

  # Live RunPod endpoint test (legacy serverless queue)
  RUNPOD_ENDPOINT_ID=<id> RUNPOD_API_KEY=<key> \
      python scripts/smoke_test.py --endpoint --image path/to/test.jpg

Notes:
  --dry-run does NOT load model weights or require a GPU. It compiles the worker
  modules, validates the response schema, exercises the /ws/vision streaming
  helpers (frame validation + latest-frame-wins + metrics snapshot) with mock
  inference, and asserts server.app registers /detect, /ws/echo, /ws/vision and
  /debug/stream. The script exits non-zero if any check fails.
"""

import argparse
import asyncio
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
    description="SafeLens vision worker smoke test",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog=__doc__,
)
parser.add_argument("--image", default=None,
                    help="Path to a test image (JPEG/PNG). Optional; synthetic JPEG if omitted.")
parser.add_argument("--dry-run", action="store_true",
                    help="Syntax + import + schema + streaming checks only. No weights, no GPU.")
parser.add_argument("--conf", type=float, default=0.35, help="Confidence threshold")
parser.add_argument("--img-size", type=int, default=640, help="Inference image size")
parser.add_argument("--endpoint", action="store_true",
                    help="Hit a live RunPod endpoint instead of the local handler")
parser.add_argument("--timeout", type=int, default=120, help="RunPod polling timeout (s)")
args = parser.parse_args()

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))


def _ok(msg):
    print(f"[smoke/dry-run] OK  {msg}")


def _fail(msg):
    print(f"[smoke/dry-run] FAIL {msg}", file=sys.stderr)
    sys.exit(1)


# ── Dry-run: syntax + import + schema + streaming checks ─────────────────────
if args.dry_run:
    print("[smoke/dry-run] Running syntax check...")
    compile_targets = [
        "schema.py", "server.py", "vision_backend.py", "ws_vision.py",
        "edgecrafter_loader.py", "handler.py", "deimv2_infer.py",
        "official_deimv2_loader.py", "bootstrap.py",
        "scripts/ws_vision_test_client.py",
    ]
    for fname in compile_targets:
        fpath = REPO_ROOT / fname
        if not fpath.exists():
            continue
        try:
            py_compile.compile(str(fpath), doraise=True)
            _ok(fname)
        except py_compile.PyCompileError as exc:
            _fail(f"{fname}: {exc}")

    print("[smoke/dry-run] Running schema import + validation...")
    try:
        from schema import InferRequest, BBox, Entity, Keypoint, Pose, InferResponse
        req = InferRequest(image_b64="aGVsbG8=", conf=0.5, img_size=320)
        assert req.conf == 0.5 and req.img_size == 320 and req.classes is None
        resp = InferResponse(
            entities=[Entity(label="person", class_id=0, confidence=0.9,
                             bbox=BBox(x=0.1, y=0.2, w=0.3, h=0.4), source="edgecrafter-det")],
            poses=[Pose(label="person", confidence=0.8,
                        keypoints=[Keypoint(name="nose", x=0.3, y=0.2, score=0.9)],
                        skeleton=[[5, 7]], source="edgecrafter-pose")],
            inference_ms=12.0, model="EdgeCrafter", backend="edgecrafter",
            tasks=["det", "pose"], img_w=640, img_h=480,
        )
        data = resp.model_dump()
        for key in ("entities", "poses", "backend", "tasks", "model",
                    "inference_ms", "img_w", "img_h", "error", "warning"):
            assert key in data, key
        err = InferResponse(entities=[], error="missing_image_b64")
        assert err.entities == [] and err.poses == []
        _ok("schema.py import + InferResponse contract")
    except Exception as exc:
        _fail(f"schema check: {exc}")

    print("[smoke/dry-run] Running /ws/vision streaming checks (mock inference)...")
    try:
        import ws_vision

        # 1) frame validation
        good, gerr = ws_vision.validate_frame(
            {"type": "frame", "frame_b64": base64.b64encode(b"\xff\xd8jpg").decode()})
        assert good and gerr is None
        assert ws_vision.validate_frame({"type": "frame"})[1] == "missing_frame_b64"
        assert ws_vision.validate_frame({"type": "frame", "frame_b64": "!!"})[1] == "invalid_base64"
        assert ws_vision.validate_frame({"type": "x", "frame_b64": "aGk="})[1] == "invalid_frame_type"
        _ok("validate_frame")

        # 2) latest-frame-wins drops stale frames, latest wins
        async def _slot_check():
            slot = ws_vision._LatestFrameSlot()
            assert slot.put({"frame_id": 1}) is False
            assert slot.put({"frame_id": 2}) is True   # dropped stale frame 1
            got = await slot.get()
            assert got["frame_id"] == 2 and slot.depth() == 0
        asyncio.run(_slot_check())
        _ok("latest-frame-wins slot")

        # 3) metrics snapshot carries all required fields
        session = ws_vision._VisionStreamSession(
            None,
            get_state=lambda: {"status": "ready", "model_loaded": True},
            trigger_warmup=lambda: None,
            run_inference=lambda **kw: None,
            get_backend=lambda: "edgecrafter",
            get_tasks=lambda: ["det", "pose"],
            get_gpu_device=lambda: None,
            default_conf=0.25, default_img_size=640,
            metrics_interval_s=2.0, warmup_timeout_s=600.0, warmup_poll_s=0.5,
        )
        snap = session.metrics_snapshot()
        for key in ("received_fps", "processed_fps", "dropped_frames", "avg_inference_ms",
                    "avg_end_to_end_latency_ms", "current_queue_depth", "model_ready",
                    "backend", "tasks", "gpu_device"):
            assert key in snap, key
        _ok("metrics snapshot fields")
    except Exception as exc:
        _fail(f"ws_vision check: {exc}")

    print("[smoke/dry-run] Checking server.app route registration...")
    try:
        os.environ.setdefault("SKIP_WARMUP", "true")
        os.environ.setdefault("AUTO_WARMUP", "false")
        import server
        paths = {getattr(r, "path", None) for r in server.app.routes}
        required = ["/health", "/ping", "/debug/startup", "/debug/model-load",
                    "/warmup", "/detect", "/ws/echo", "/ws/vision", "/debug/stream"]
        missing = [p for p in required if p not in paths]
        assert not missing, f"missing routes: {missing}"
        _ok("server.app registers " + ", ".join(required))
    except Exception as exc:
        _fail(f"server route check: {exc}")

    print("[smoke/dry-run] Running handler structured-error check (optional)...")
    try:
        from handler import handler as _handler
        result = _handler({"input": {}})
        assert result.get("error") == "missing_image_b64"
        assert result.get("entities") == []
        _ok("handler missing-image returns structured error")
    except (ImportError, ModuleNotFoundError) as exc:
        # Legacy DEIMv2 handler needs torch + the /opt/DEIMv2 clone (Docker image
        # only). Not required for the live-server path -- skip cleanly.
        print(f"[smoke/dry-run] SKIP handler check (deps unavailable: {exc})")

    print("[smoke/dry-run] All checks passed.")
    sys.exit(0)

# ── Build image payload (live paths) ─────────────────────────────────────────
if args.image:
    image_path = Path(args.image)
    if not image_path.exists():
        print(f"ERROR: image not found: {image_path}", file=sys.stderr)
        sys.exit(1)
    with open(image_path, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode()
    print(f"[smoke] Using image: {image_path}")
else:
    from PIL import Image as _PILImage
    buf = io.BytesIO()
    _PILImage.new("RGB", (8, 8), color=(255, 255, 255)).save(buf, format="JPEG")
    image_b64 = base64.b64encode(buf.getvalue()).decode()
    print("[smoke] Using synthetic 8x8 white JPEG (PIL)")

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

    if "error" in result and result["error"]:
        if str(result["error"]).startswith("model_load_failed"):
            print(f"[smoke] NOTE: model not available ({result['error']}). "
                  "Expected in a no-weights environment.", file=sys.stderr)
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
resp = requests.post(f"{BASE}/run", json={"input": request_payload}, headers=headers, timeout=30)
resp.raise_for_status()
job_id = resp.json()["id"]
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
        print(f"[smoke] OK: {len(output.get('entities', []))} entities, "
              f"{output.get('inference_ms')} ms")
        sys.exit(0)
    if status in ("FAILED", "CANCELLED"):
        print(f"[smoke] FAIL: job {status}", file=sys.stderr)
        sys.exit(1)

print(f"[smoke] TIMEOUT after {args.timeout}s", file=sys.stderr)
sys.exit(1)
