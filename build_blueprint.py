"""
build_blueprint.py -- Build Mode lightweight blueprint processing (CPU-only).

Turns a selected-crop image + the app's MediaPipe hand landmarks/gesture into a
lightweight, replayable blueprint JSON frame:

    crop image -> grayscale -> blur -> Canny edges -> contours ->
    largest useful contour -> approxPolyDP outline -> normalized 0..1 points ->
    anchors (center / bbox corners / contour / high-curvature) ->
    hand landmarks mapped to crop-local coords ->
    step markers from active pinch gestures.

HARD separation from EdgeCrafter / the HSE detect pipeline:
  * never imports or loads EdgeCrafter / vision_backend
  * never touches the GPU, never triggers model warmup
  * OpenCV / NumPy / Pillow are imported lazily and the heavy work runs in a
    worker thread (via asyncio.to_thread) so the event loop -- and /detect --
    is never blocked.

Storage is in-memory MVP only: lightweight JSON keyframes (NEVER the source
image, never video), with per-session frame caps and TTL cleanup.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import logging
import math
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

from build_schema import (
    BlueprintFrame,
    BuildError,
    MAX_FRAMES_PER_SESSION,
    MAX_IMAGE_B64_CHARS,
    MAX_SESSIONS,
    SESSION_TTL_SECONDS,
)

log = logging.getLogger("safelens-vision-worker.build")

# In-memory MVP session store. Lightweight JSON only -- no images, no video.
BUILD_SESSIONS: Dict[str, Dict[str, Any]] = {}


# -- Session lifecycle --------------------------------------------------------

def _cleanup_expired() -> None:
    now = time.time()
    for sid in [s for s, v in list(BUILD_SESSIONS.items())
                if now - v.get("created_at", now) > SESSION_TTL_SECONDS]:
        BUILD_SESSIONS.pop(sid, None)


def _evict_oldest() -> None:
    if BUILD_SESSIONS:
        oldest = min(BUILD_SESSIONS.items(), key=lambda kv: kv[1].get("created_at", 0))[0]
        BUILD_SESSIONS.pop(oldest, None)


def _maybe_region(region: Any) -> Optional[Dict[str, float]]:
    try:
        return _validate_region(region)
    except BuildError:
        return None


def start_session(payload: Dict[str, Any]) -> Dict[str, Any]:
    _cleanup_expired()
    if len(BUILD_SESSIONS) >= MAX_SESSIONS:
        _evict_oldest()
    sid = "build_" + uuid.uuid4().hex[:16]
    now = time.time()
    BUILD_SESSIONS[sid] = {
        "created_at": now,
        "updated_at": now,
        "camera_id": payload.get("camera_id") or payload.get("cameraId"),
        "selection": _maybe_region(payload.get("selectedRegion") or payload.get("selection")),
        "locked": False,
        "finished": False,
        "frames": [],
    }
    return {
        "ok": True,
        "session_id": sid,
        "created_at": now,
        "max_frames": MAX_FRAMES_PER_SESSION,
        "ttl_seconds": SESSION_TTL_SECONDS,
    }


def _require_session(payload: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    sid = payload.get("sessionId") or payload.get("session_id")
    if not sid or not isinstance(sid, str):
        raise BuildError("missing_session_id", 400)
    sess = BUILD_SESSIONS.get(sid)
    if sess is None:
        raise BuildError("unknown_session", 404)
    return sid, sess


def lock_session(payload: Dict[str, Any]) -> Dict[str, Any]:
    sid, sess = _require_session(payload)
    region = _validate_region(payload.get("selectedRegion") or payload.get("selection"))
    sess["selection"] = region
    sess["locked"] = True
    sess["updated_at"] = time.time()
    return {"ok": True, "session_id": sid, "locked": True, "selection": region}


def finish_session(payload: Dict[str, Any]) -> Dict[str, Any]:
    sid, sess = _require_session(payload)
    sess["finished"] = True
    sess["updated_at"] = time.time()
    return {
        "ok": True,
        "session_id": sid,
        "replay_id": sid,
        "frame_count": len(sess["frames"]),
        "replay_url": f"/build/session/{sid}/replay",
    }


def get_replay(session_id: str) -> Dict[str, Any]:
    _cleanup_expired()
    sess = BUILD_SESSIONS.get(session_id)
    if sess is None:
        raise BuildError("unknown_session", 404)
    return {
        "ok": True,
        "session_id": session_id,
        "frame_count": len(sess["frames"]),
        "created_at": sess["created_at"],
        "selection": sess.get("selection"),
        "finished": sess.get("finished", False),
        "frames": sess["frames"],
    }


# -- Validation ---------------------------------------------------------------

def _validate_region(region: Any) -> Dict[str, float]:
    if not isinstance(region, dict):
        raise BuildError("invalid_selected_region", 400)
    try:
        x, y = float(region["x"]), float(region["y"])
        w, h = float(region["w"]), float(region["h"])
    except (KeyError, TypeError, ValueError):
        raise BuildError("invalid_selected_region", 400)
    if w <= 0 or h <= 0:
        raise BuildError("invalid_selected_region", 400)
    return {"x": x, "y": y, "w": w, "h": h}


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


# -- Frame processing ---------------------------------------------------------

async def process_frame_async(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Validate + decode on the event loop, run the CPU pipeline in a thread."""
    t0 = time.perf_counter()
    _cleanup_expired()
    sid, sess = _require_session(payload)
    if len(sess["frames"]) >= MAX_FRAMES_PER_SESSION:
        raise BuildError("too_many_frames", 409)

    image_b64 = payload.get("image_b64")
    if not image_b64 or not isinstance(image_b64, str):
        raise BuildError("missing_image_b64", 400)
    if len(image_b64) > MAX_IMAGE_B64_CHARS:
        raise BuildError("payload_too_large", 413)
    try:
        image_bytes = base64.b64decode(image_b64, validate=True)
    except (binascii.Error, ValueError):
        raise BuildError("invalid_base64", 400)
    if not image_bytes:
        raise BuildError("decode_failure", 400)

    region = _validate_region(payload.get("selectedRegion") or sess.get("selection"))
    frame_id = str(payload.get("frameId") or payload.get("frame_id") or f"f-{len(sess['frames'])}")
    timestamp_ms = _safe_int(payload.get("timestampMs") or payload.get("timestamp_ms"))
    hand_landmarks = payload.get("handLandmarks") or payload.get("hand_landmarks") or []
    gesture = payload.get("gesture") or {}

    try:
        blueprint = await asyncio.to_thread(
            _build_blueprint_frame, image_bytes, region, hand_landmarks, gesture,
            sid, frame_id, timestamp_ms,
        )
    except BuildError:
        raise
    except Exception as exc:  # noqa: BLE001 -- one bad frame must not crash the worker
        log.warning("build: frame processing failed: %s", exc)
        raise BuildError("processing_failure", 500)

    # Normalize/validate the output shape (also sanitizes NumPy scalar types).
    bp_frame = BlueprintFrame(**blueprint).model_dump()
    sess["frames"].append(bp_frame)
    sess["updated_at"] = time.time()
    return {
        "ok": True,
        "session_id": sid,
        "frame_id": frame_id,
        "blueprint_frame": bp_frame,
        "processing_ms": round((time.perf_counter() - t0) * 1000.0, 2),
    }


def _build_blueprint_frame(image_bytes: bytes, region: Dict[str, float],
                           hand_landmarks: List[Any], gesture: Any,
                           session_id: str, frame_id: str,
                           timestamp_ms: int) -> Dict[str, Any]:
    """Pure CPU pipeline (runs in a worker thread). Returns a blueprint dict."""
    import cv2
    import numpy as np

    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        # Pillow fallback for formats OpenCV may not decode directly.
        import io
        from PIL import Image
        pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    h, w = img.shape[:2]
    if w <= 0 or h <= 0:
        raise ValueError("empty image")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best = None
    if contours:
        cand = max(contours, key=cv2.contourArea)
        if cv2.contourArea(cand) >= (w * h) * 0.001:
            best = cand

    if best is not None:
        peri = cv2.arcLength(best, True)
        approx = cv2.approxPolyDP(best, 0.01 * peri, True).reshape(-1, 2)
        outline = [{"x": _n(px, w), "y": _n(py, h)} for px, py in approx]
        full = best.reshape(-1, 2)
        stride = max(1, len(full) // 48)
        sparse = [{"x": _n(px, w), "y": _n(py, h)} for px, py in full[::stride]][:64]
        bx, by, bw, bh = cv2.boundingRect(best)
        moments = cv2.moments(best)
        if moments["m00"]:
            cx, cy = _n(moments["m10"] / moments["m00"], w), _n(moments["m01"] / moments["m00"], h)
        else:
            cx, cy = _n(bx + bw / 2.0, w), _n(by + bh / 2.0, h)
    else:
        # No useful contour: fall back to the full crop rectangle.
        outline = [{"x": 0.0, "y": 0.0}, {"x": 1.0, "y": 0.0},
                   {"x": 1.0, "y": 1.0}, {"x": 0.0, "y": 1.0}]
        sparse = []
        bx, by, bw, bh = 0, 0, w, h
        cx, cy = 0.5, 0.5

    anchors: List[Dict[str, Any]] = [
        {"id": "center", "x": cx, "y": cy, "label": "center", "confidence": 1.0}
    ]
    nbx, nby, nbw, nbh = _n(bx, w), _n(by, h), _n(bw, w), _n(bh, h)
    for cid, ax, ay in (("bbox-tl", nbx, nby), ("bbox-tr", nbx + nbw, nby),
                        ("bbox-br", nbx + nbw, nby + nbh), ("bbox-bl", nbx, nby + nbh)):
        anchors.append({"id": cid, "x": _clamp(ax), "y": _clamp(ay),
                        "label": "bbox", "confidence": 0.9})
    if outline:
        n = len(outline)
        k = min(8, n)
        for i in range(k):
            p = outline[(i * n) // k]
            anchors.append({"id": f"contour-{i}", "x": p["x"], "y": p["y"],
                            "label": "contour", "confidence": 0.7})
        for j, p in enumerate(_high_curvature(outline)):
            anchors.append({"id": f"corner-{j}", "x": p["x"], "y": p["y"],
                            "label": "corner", "confidence": 0.8})

    # App hand landmarks (card coords) -> selected-crop-local coords. The app
    # already holds the card-space points; the worker's value-add is the
    # crop-local mapping that lines up with the blueprint outline above.
    hand_out: List[Dict[str, float]] = []
    index_local: Optional[Tuple[float, float]] = None
    for lm in hand_landmarks:
        if not isinstance(lm, dict):
            continue
        try:
            lx = (float(lm["x"]) - region["x"]) / region["w"]
            ly = (float(lm["y"]) - region["y"]) / region["h"]
        except (KeyError, TypeError, ValueError):
            continue
        hand_out.append({"x": lx, "y": ly})
        role = str(lm.get("role") or lm.get("id") or "").lower()
        if "index" in role and index_local is None:
            index_local = (lx, ly)

    step_markers: List[Dict[str, Any]] = []
    if isinstance(gesture, dict) and gesture.get("active"):
        if index_local is None and hand_out:
            index_local = (hand_out[0]["x"], hand_out[0]["y"])
        if index_local is not None:
            step_markers.append({
                "id": f"step-{frame_id}",
                "label": "Pinch / grab",
                "x": index_local[0],
                "y": index_local[1],
                "timestampMs": timestamp_ms,
            })

    return {
        "sessionId": session_id,
        "frameId": frame_id,
        "timestampMs": timestamp_ms,
        "outline": outline,
        "anchors": anchors,
        "sparsePoints": sparse,
        "handLandmarks": hand_out,
        "stepMarkers": step_markers,
        "gesture": {
            "type": gesture.get("type") if isinstance(gesture, dict) else None,
            "active": bool(gesture.get("active")) if isinstance(gesture, dict) else False,
            "strength": gesture.get("strength") if isinstance(gesture, dict) else None,
        },
    }


def _n(value: Any, size: Any) -> float:
    """Normalize a pixel coordinate to a clamped 0..1 fraction of `size`."""
    return float(max(0.0, min(1.0, float(value) / float(size)))) if size else 0.0


def _clamp(value: Any) -> float:
    return float(max(0.0, min(1.0, float(value))))


def _high_curvature(outline: List[Dict[str, float]], max_pts: int = 6,
                    angle_threshold: float = 110.0) -> List[Dict[str, float]]:
    """Pick sharp-angle vertices from the normalized outline (cheap, optional)."""
    n = len(outline)
    if n < 3:
        return []
    out: List[Dict[str, float]] = []
    for i in range(n):
        a, b, c = outline[(i - 1) % n], outline[i], outline[(i + 1) % n]
        v1x, v1y = a["x"] - b["x"], a["y"] - b["y"]
        v2x, v2y = c["x"] - b["x"], c["y"] - b["y"]
        n1, n2 = math.hypot(v1x, v1y), math.hypot(v2x, v2y)
        if n1 < 1e-6 or n2 < 1e-6:
            continue
        cosang = max(-1.0, min(1.0, (v1x * v2x + v1y * v2y) / (n1 * n2)))
        if math.degrees(math.acos(cosang)) < angle_threshold:
            out.append(b)
        if len(out) >= max_pts:
            break
    return out
