#!/usr/bin/env python3
"""
scripts/ws_vision_test_client.py -- SERVER-SIDE test client for /ws/vision.

Connects to a local or RunPod /ws/vision WebSocket, streams JPEG frames from a
folder of images (or, optionally, a webcam via OpenCV), and prints the live
streaming stats reported by the worker.

SECURITY -- read this:
  * This is a SERVER-SIDE / operator tool only. The RunPod API key (Bearer
    token) is read from an environment variable (default: RUNPOD_API_KEY) and
    sent as an Authorization header. NEVER hardcode it, never print it, and
    NEVER ship a RunPod API key in frontend / browser / mobile code -- the
    browser must talk to your own backend, which proxies to RunPod.

Usage
-----
  # Local worker, frames from a folder, 5 FPS:
  python scripts/ws_vision_test_client.py \
      --url ws://localhost:8000/ws/vision --images ./examples/frames --fps 5

  # RunPod load-balancing endpoint (server-side), auth from env:
  export RUNPOD_API_KEY=...                       # never commit this
  python scripts/ws_vision_test_client.py \
      --url wss://<endpoint-id>-8000.proxy.runpod.net/ws/vision \
      --images ./frames --fps 5 --auth-env RUNPOD_API_KEY

  # Webcam (needs opencv-python):
  python scripts/ws_vision_test_client.py --url ws://localhost:8000/ws/vision --webcam

Requires: pip install websockets pillow   (opencv-python only for --webcam)
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import io
import os
import sys
import time
from pathlib import Path


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Server-side /ws/vision streaming test client",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--url", default="ws://localhost:8000/ws/vision",
                   help="WebSocket URL (ws:// local, wss:// RunPod). Default: local.")
    src = p.add_mutually_exclusive_group()
    src.add_argument("--images", default=None,
                     help="Folder of JPEG/PNG images to stream (looped).")
    src.add_argument("--webcam", action="store_true",
                     help="Capture from a webcam instead (requires opencv-python).")
    p.add_argument("--webcam-index", type=int, default=0, help="Webcam device index.")
    p.add_argument("--fps", type=float, default=5.0,
                   help="Send rate in frames/sec (start low, e.g. 5). Default: 5.")
    p.add_argument("--size", type=int, default=640, choices=(512, 640),
                   help="Resize longest side to this before JPEG encoding (512 or 640).")
    p.add_argument("--jpeg-quality", type=int, default=80, help="JPEG quality (1-100).")
    p.add_argument("--camera-id", default="browser-test", help="camera_id tag to send.")
    p.add_argument("--frames", type=int, default=0,
                   help="Stop after sending N frames (0 = run until --duration / Ctrl-C).")
    p.add_argument("--duration", type=float, default=20.0,
                   help="Stop after this many seconds (0 = no limit). Default: 20.")
    p.add_argument("--auth-env", default="RUNPOD_API_KEY",
                   help="Env var holding the RunPod API key for the Authorization "
                        "header. SERVER-SIDE ONLY -- never expose this in a browser.")
    p.add_argument("--insecure-auth", default=None,
                   help="(Discouraged) pass a Bearer token directly. Prefer --auth-env.")
    return p.parse_args(argv)


# ── Frame sources ──────────────────────────────────────────────────────────────

def _iter_image_paths(folder: str):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    paths = sorted(p for p in Path(folder).iterdir()
                   if p.suffix.lower() in exts and p.is_file())
    if not paths:
        raise SystemExit(f"[client] no images found in {folder}")
    return paths


def _encode_pil(img, size: int, quality: int) -> str:
    """Resize (longest side -> size, aspect preserved) and JPEG-encode to base64."""
    from PIL import Image  # local import so --help works without Pillow
    img = img.convert("RGB")
    img.thumbnail((size, size), Image.BILINEAR)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode()


def _frames_from_folder(folder, size, quality, limit):
    from PIL import Image
    paths = _iter_image_paths(folder)
    sent = 0
    while True:
        for path in paths:
            if limit and sent >= limit:
                return
            try:
                with Image.open(path) as im:
                    yield _encode_pil(im, size, quality)
                sent += 1
            except Exception as exc:  # noqa: BLE001
                print(f"[client] skip {path.name}: {exc}", file=sys.stderr)


def _frames_from_webcam(index, size, quality, limit):
    try:
        import cv2  # noqa: F401
    except ImportError:
        raise SystemExit("[client] --webcam needs opencv-python (pip install opencv-python)")
    import cv2
    from PIL import Image
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise SystemExit(f"[client] cannot open webcam {index}")
    sent = 0
    try:
        while True:
            if limit and sent >= limit:
                return
            ok, frame = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            yield _encode_pil(Image.fromarray(rgb), size, quality)
            sent += 1
    finally:
        cap.release()


# ── Stats ───────────────────────────────────────────────────────────────────────

class _Stats:
    def __init__(self):
        self.sent = 0
        self.vision = 0
        self.errors = 0
        self.entities = 0
        self.poses = 0
        self.inf_ms = []
        self.lat_ms = []
        self.start = time.monotonic()

    def processed_fps(self):
        dt = max(time.monotonic() - self.start, 1e-6)
        return self.vision / dt

    def avg(self, xs):
        return sum(xs) / len(xs) if xs else 0.0


# ── Async client ──────────────────────────────────────────────────────────────

def _connect_kwargs(args):
    """Build websockets.connect kwargs, adding Authorization only if present."""
    token = args.insecure_auth or os.environ.get(args.auth_env, "")
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
        print("[client] Authorization header set from "
              f"{'--insecure-auth' if args.insecure_auth else args.auth_env}. "
              "SERVER-SIDE ONLY -- never expose this key in browser/frontend code.")
    else:
        print("[client] no auth token (set --auth-env / RUNPOD_API_KEY for RunPod).")
    return headers


async def _sender(ws, frames, args, stats: _Stats, stop: asyncio.Event):
    import json
    period = 1.0 / args.fps if args.fps > 0 else 0.0
    for frame_b64 in frames:
        if stop.is_set():
            break
        msg = {
            "type": "frame",
            "camera_id": args.camera_id,
            "frame_id": stats.sent,
            "sent_at": int(time.time() * 1000),
            "frame_b64": frame_b64,
        }
        try:
            await ws.send(json.dumps(msg))
        except Exception as exc:  # noqa: BLE001
            print(f"[client] send failed: {exc}", file=sys.stderr)
            break
        stats.sent += 1
        if period:
            await asyncio.sleep(period)
    # Allow in-flight results to drain before the receiver is cancelled.
    await asyncio.sleep(1.0)
    stop.set()


async def _receiver(ws, stats: _Stats, stop: asyncio.Event):
    import json
    while not stop.is_set():
        try:
            raw = await ws.recv()
        except Exception:  # noqa: BLE001 -- connection closed
            break
        try:
            msg = json.loads(raw)
        except (ValueError, TypeError):
            continue
        mtype = msg.get("type")
        if mtype == "connected":
            print("[client] <- connected")
        elif mtype == "warming":
            print("[client] <- warming (model cold; worker is warming up...)")
        elif mtype == "ready":
            print(f"[client] <- ready backend={msg.get('backend')} tasks={msg.get('tasks')}")
        elif mtype == "vision":
            stats.vision += 1
            stats.entities += len(msg.get("entities") or [])
            stats.poses += len(msg.get("poses") or [])
            if msg.get("inference_ms") is not None:
                stats.inf_ms.append(float(msg["inference_ms"]))
            if msg.get("end_to_end_latency_ms") is not None:
                stats.lat_ms.append(float(msg["end_to_end_latency_ms"]))
        elif mtype == "metrics":
            print("[client] <- metrics "
                  f"recv_fps={msg.get('received_fps')} proc_fps={msg.get('processed_fps')} "
                  f"dropped={msg.get('dropped_frames')} avg_inf_ms={msg.get('avg_inference_ms')} "
                  f"avg_e2e_ms={msg.get('avg_end_to_end_latency_ms')} "
                  f"queue={msg.get('current_queue_depth')} ready={msg.get('model_ready')} "
                  f"gpu={msg.get('gpu_device')}")
        elif mtype == "error":
            stats.errors += 1
            print(f"[client] <- error: {msg.get('error')}", file=sys.stderr)


async def _run(args):
    try:
        import websockets
    except ImportError:
        raise SystemExit("[client] needs the 'websockets' package (pip install websockets)")

    if args.webcam:
        frames = _frames_from_webcam(args.webcam_index, args.size, args.jpeg_quality, args.frames)
    elif args.images:
        frames = _frames_from_folder(args.images, args.size, args.jpeg_quality, args.frames)
    else:
        raise SystemExit("[client] provide --images <folder> or --webcam")

    headers = _connect_kwargs(args)
    stats = _Stats()
    stop = asyncio.Event()

    # websockets renamed extra_headers -> additional_headers in v14; support both.
    connect = websockets.connect
    try:
        ctx = connect(args.url, additional_headers=headers) if headers else connect(args.url)
    except TypeError:
        ctx = connect(args.url, extra_headers=headers) if headers else connect(args.url)

    print(f"[client] connecting to {args.url} (fps={args.fps}, size={args.size}) ...")
    async with ctx as ws:
        tasks = [
            asyncio.create_task(_receiver(ws, stats, stop)),
            asyncio.create_task(_sender(ws, frames, args, stats, stop)),
        ]
        if args.duration > 0:
            try:
                await asyncio.wait_for(stop.wait(), timeout=args.duration)
            except asyncio.TimeoutError:
                stop.set()
        else:
            await stop.wait()
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    dt = max(time.monotonic() - stats.start, 1e-6)
    print("\n[client] ===== summary =====")
    print(f"[client] sent frames      : {stats.sent}")
    print(f"[client] vision results   : {stats.vision}")
    print(f"[client] processed FPS     : {stats.processed_fps():.2f}")
    print(f"[client] dropped (sent-got): {max(stats.sent - stats.vision, 0)}")
    print(f"[client] avg inference_ms  : {stats.avg(stats.inf_ms):.1f}")
    print(f"[client] avg e2e latency_ms: {stats.avg(stats.lat_ms):.1f}")
    print(f"[client] total entities    : {stats.entities}")
    print(f"[client] total poses       : {stats.poses}")
    print(f"[client] errors            : {stats.errors}")
    print(f"[client] wall time s       : {dt:.1f}")
    return 0 if stats.errors == 0 else 1


def main(argv=None):
    args = _parse_args(argv)
    try:
        return asyncio.run(_run(args))
    except KeyboardInterrupt:
        print("\n[client] interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
