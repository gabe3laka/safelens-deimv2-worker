"""
build_blueprint.py -- Build Mode lightweight blueprint processing (CPU-only).

Turns a selected-crop image + the app's MediaPipe hand landmarks/gesture into a
lightweight, replayable BlueprintFrame v2:

    crop image
      -> optional SAM2-style segmentation (build_segmentation) / fallback Canny
      -> mask contour + outline + anchors + sparse points
      -> hand landmarks mapped to crop-local coords + pinch step markers
      -> workflowMode-aware AI notes / instructions / plan steps
      -> JSON replay frame (v2)

workflowMode:
    "build" -> user is doing the work; the worker DOCUMENTS activity (notes).
    "plan"  -> user wants guidance; the worker SUGGESTS steps / next actions.
    missing -> defaults to "build" (backward compatible).

HARD separation from EdgeCrafter / the HSE detect pipeline:
  * never imports or loads EdgeCrafter / vision_backend
  * never touches the GPU for the fallback path, never triggers model warmup
  * heavy CV runs in a worker thread (asyncio.to_thread) so /detect is never
    blocked. SAM2 is optional, lazy, and only used when explicitly enabled.

Storage is in-memory MVP only: lightweight JSON keyframes + a per-session mask
contour (NEVER the source image, never video), with frame caps + TTL cleanup.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import logging
import math
import os
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


# -- Config -------------------------------------------------------------------

def _seg_config() -> Dict[str, Any]:
    return {
        "backend": os.getenv("BUILD_SEGMENTATION_BACKEND", "fallback").strip().lower(),
        "mask_output": os.getenv("BUILD_MASK_OUTPUT", "contour").strip().lower(),
        "every_n": _safe_int(os.getenv("BUILD_SEGMENT_EVERY_N", "3"), 3),
        "on_extract": os.getenv("BUILD_SEGMENT_ON_EXTRACT", "true").strip().lower()
        in ("1", "true", "yes", "on"),
    }


def _norm_mode(value: Any) -> str:
    """Normalize workflowMode to 'plan' or 'build' (default build)."""
    return "plan" if str(value or "").strip().lower() == "plan" else "build"


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
    mode = _norm_mode(payload.get("workflowMode") or payload.get("workflow_mode") or "build")
    BUILD_SESSIONS[sid] = {
        "created_at": now,
        "updated_at": now,
        "camera_id": payload.get("camera_id") or payload.get("cameraId"),
        "selection": _maybe_region(payload.get("selectedRegion") or payload.get("selection")),
        "workflow_mode": mode,
        "locked": False,
        "finished": False,
        "frames": [],
        # v2 session-scoped state (lightweight JSON only -- never an image)
        "source_asset": None,
        "plan_steps": None,
        "plan_index": 0,
        "pinch_prev": False,
        "last_index": None,
    }
    return {
        "ok": True,
        "session_id": sid,
        "created_at": now,
        "workflow_mode": mode,
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
    if payload.get("workflowMode") or payload.get("workflow_mode"):
        sess["workflow_mode"] = _norm_mode(payload.get("workflowMode") or payload.get("workflow_mode"))
    sess["updated_at"] = time.time()
    return {
        "ok": True,
        "session_id": sid,
        "locked": True,
        "selection": region,
        "workflow_mode": sess.get("workflow_mode", "build"),
    }


def finish_session(payload: Dict[str, Any]) -> Dict[str, Any]:
    sid, sess = _require_session(payload)
    sess["finished"] = True
    sess["updated_at"] = time.time()
    return {
        "ok": True,
        "session_id": sid,
        "replay_id": sid,
        "frame_count": len(sess["frames"]),
        "workflow_mode": sess.get("workflow_mode", "build"),
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
        "workflow_mode": sess.get("workflow_mode", "build"),
        "selection": sess.get("selection"),
        "finished": sess.get("finished", False),
        "frames": sess["frames"],  # already BlueprintFrame v2 dicts -- JSON only
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
    """Validate + decode on the loop; run segmentation/CV in a worker thread."""
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
    frame_index = len(sess["frames"])
    frame_id = str(payload.get("frameId") or payload.get("frame_id") or f"f-{frame_index}")
    timestamp_ms = _safe_int(payload.get("timestampMs") or payload.get("timestamp_ms"))
    hand_landmarks = payload.get("handLandmarks") or payload.get("hand_landmarks") or []
    gesture = payload.get("gesture") or {}
    workflow_mode = _norm_mode(
        payload.get("workflowMode") or payload.get("workflow_mode") or sess.get("workflow_mode") or "build"
    )
    cfg = _seg_config()

    # Hand landmarks -> crop-local coords (cheap; on the event loop).
    hand_out, index_local = _convert_hand(hand_landmarks, region)
    gesture_active = bool(gesture.get("active")) if isinstance(gesture, dict) else False
    if gesture_active and index_local is None and hand_out:
        index_local = (hand_out[0]["x"], hand_out[0]["y"])

    # Event-based segmentation: only recompute the mask on meaningful events.
    prior = sess.get("source_asset")
    is_extraction = _is_extraction(payload, frame_index)
    hand_moved = _hand_moved(sess, hand_landmarks)
    should_segment = (
        prior is None
        or (is_extraction and cfg["on_extract"])
        or (cfg["every_n"] > 0 and frame_index % cfg["every_n"] == 0)
        or (gesture_active and hand_moved)
    )

    if should_segment:
        prompt_point = [
            _clamp(index_local[0]) if index_local else 0.5,
            _clamp(index_local[1]) if index_local else 0.5,
        ]
        try:
            geom = await asyncio.to_thread(_segment_geometry, image_bytes, region, cfg, prompt_point)
        except BuildError:
            raise
        except Exception as exc:  # noqa: BLE001 -- a bad frame must not crash the worker
            log.warning("build: frame processing failed: %s", exc)
            raise BuildError("processing_failure", 500)
        geom["id"] = payload.get("sourceAssetId") or (prior or {}).get("id") or ("asset_" + uuid.uuid4().hex[:12])
        geom["updatedAtFrame"] = frame_id
        sess["source_asset"] = geom
    else:
        geom = prior  # reuse the previous mask/geometry (JSON, no image work)

    # Track the index fingertip (card coords) for movement-based segmentation.
    sess["last_index"] = _index_card(hand_landmarks)

    step_markers = _step_markers(index_local, gesture, frame_id, timestamp_ms)
    ai = make_ai_fields(workflow_mode, geom, hand_out, index_local, gesture,
                        float(geom.get("confidence", 0.0)), frame_index, sess)

    mask_output = cfg["mask_output"]
    blueprint = {
        "version": 2,
        "workflowMode": workflow_mode,
        "sessionId": sid,
        "frameId": frame_id,
        "timestampMs": timestamp_ms,
        "sourceAssetId": payload.get("sourceAssetId") or geom.get("id"),
        "sourceMaskB64": geom.get("sourceMaskB64") if mask_output == "mask_thumbnail" else None,
        "maskSource": geom.get("maskSource", "none"),
        "maskContour": geom.get("maskContour", []) if mask_output == "contour" else [],
        "outline": geom.get("outline", []),
        "anchors": geom.get("anchors", []),
        "sparsePoints": geom.get("sparsePoints", []),
        "handLandmarks": hand_out,
        "stepMarkers": step_markers,
        "gesture": {
            "type": gesture.get("type") if isinstance(gesture, dict) else None,
            "active": gesture_active,
            "strength": gesture.get("strength") if isinstance(gesture, dict) else None,
        },
        "instruction": ai["instruction"],
        "aiNotes": ai["aiNotes"],
        "nextAction": ai["nextAction"],
        "safetyWarning": ai["safetyWarning"],
        "qualityCheck": ai["qualityCheck"],
        "activityLabel": ai["activityLabel"],
        "detectedIntent": ai["detectedIntent"],
        "importance": ai["importance"],
        "planSteps": ai["planSteps"],
        "currentPlanStepIndex": ai["currentPlanStepIndex"],
    }

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


# -- Geometry / segmentation (runs in a worker thread) ------------------------

def _segment_geometry(image_bytes: bytes, region: Dict[str, float], cfg: Dict[str, Any],
                      prompt_point: List[float]) -> Dict[str, Any]:
    """Decode + segment the crop. SAM2 if enabled+available, else Canny contour."""
    import cv2
    import numpy as np

    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        import io
        from PIL import Image
        pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    h, w = img.shape[:2]
    if w <= 0 or h <= 0:
        raise ValueError("empty image")

    # Cheap Canny contour -- always computed so `outline` is always present.
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best = None
    if contours:
        cand = max(contours, key=cv2.contourArea)
        if cv2.contourArea(cand) >= (w * h) * 0.001:
            best = cand

    canny_contour: List[Dict[str, float]] = []
    full_pts = None
    if best is not None:
        peri = cv2.arcLength(best, True)
        approx = cv2.approxPolyDP(best, 0.01 * peri, True).reshape(-1, 2)
        canny_contour = [{"x": _n(px, w), "y": _n(py, h)} for px, py in approx]
        full_pts = best.reshape(-1, 2)

    # Decide the mask (Step 4 fallback first; Step 5 optional SAM2).
    mask_contour: List[Dict[str, float]] = []
    mask_source = "none"
    confidence = 0.0
    chosen = None  # contour used for `outline`

    backend = cfg["backend"]
    if backend == "sam2":
        seg = {"ok": False}
        try:
            import build_segmentation
            seg = build_segmentation.segment_crop(img, prompt={"point": prompt_point})
        except Exception as exc:  # noqa: BLE001 -- SAM2 must never break the frame
            log.warning("build: sam2 call failed, using fallback: %s", exc)
        if seg.get("ok") and seg.get("mask_contour"):
            mask_contour = seg["mask_contour"]
            mask_source = "sam2"
            confidence = float(seg.get("confidence", 0.8))
            chosen = mask_contour
        elif canny_contour:
            mask_contour = canny_contour
            mask_source = "fallback-contour"
            confidence = 0.6
            chosen = canny_contour
        else:
            mask_source = "none"
            confidence = 0.3
    elif backend == "fallback":
        if canny_contour:
            mask_contour = canny_contour
            mask_source = "fallback-contour"
            confidence = 0.7
            chosen = canny_contour
        else:
            mask_source = "none"
            confidence = 0.3
    else:  # "none" -- no mask, but keep `outline` for backward compatibility
        mask_source = "none"
        confidence = 0.6 if canny_contour else 0.3
        chosen = canny_contour or None

    if chosen:
        outline = chosen
    elif canny_contour:
        outline = canny_contour
    else:
        outline = [{"x": 0.0, "y": 0.0}, {"x": 1.0, "y": 0.0},
                   {"x": 1.0, "y": 1.0}, {"x": 0.0, "y": 1.0}]

    xs = [p["x"] for p in outline]
    ys = [p["y"] for p in outline]
    minx, maxx, miny, maxy = min(xs), max(xs), min(ys), max(ys)
    center = {"x": (minx + maxx) / 2.0, "y": (miny + maxy) / 2.0}
    if best is not None:
        m = cv2.moments(best)
        if m["m00"]:
            center = {"x": _n(m["m10"] / m["m00"], w), "y": _n(m["m01"] / m["m00"], h)}

    sparse: List[Dict[str, float]] = []
    if full_pts is not None:
        stride = max(1, len(full_pts) // 48)
        sparse = [{"x": _n(px, w), "y": _n(py, h)} for px, py in full_pts[::stride]][:64]

    anchors = _build_anchors(outline, minx, miny, maxx, maxy, center)

    mask_output = cfg["mask_output"]
    source_mask_b64 = None
    if mask_output == "mask_thumbnail" and mask_contour:
        source_mask_b64 = _mask_thumbnail(mask_contour)

    return {
        "outline": outline,
        "anchors": anchors,
        "sparsePoints": sparse,
        "maskContour": mask_contour,
        "sourceMaskB64": source_mask_b64,
        "maskSource": mask_source,
        "confidence": confidence,
        "center": center,
    }


def _build_anchors(outline, minx, miny, maxx, maxy, center) -> List[Dict[str, Any]]:
    anchors = [{"id": "center", "x": _clamp(center["x"]), "y": _clamp(center["y"]),
                "label": "center", "confidence": 1.0}]
    for cid, ax, ay in (("bbox-tl", minx, miny), ("bbox-tr", maxx, miny),
                        ("bbox-br", maxx, maxy), ("bbox-bl", minx, maxy)):
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
    return anchors


def _mask_thumbnail(contour_norm: List[Dict[str, float]], size: int = 96) -> Optional[str]:
    """Small filled-mask PNG (base64) from a normalized contour. Best-effort."""
    try:
        import cv2
        import numpy as np
        pts = np.array([[int(p["x"] * size), int(p["y"] * size)] for p in contour_norm], dtype=np.int32)
        mask = np.zeros((size, size), dtype=np.uint8)
        if len(pts) >= 3:
            cv2.fillPoly(mask, [pts], 255)
        ok, buf = cv2.imencode(".png", mask)
        return base64.b64encode(buf.tobytes()).decode() if ok else None
    except Exception:  # noqa: BLE001
        return None


# -- Hand landmarks / gestures (event loop) -----------------------------------

def _convert_hand(hand_landmarks, region) -> Tuple[List[Dict[str, float]], Optional[Tuple[float, float]]]:
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
    return hand_out, index_local


def _index_card(hand_landmarks) -> Optional[Tuple[float, float]]:
    for lm in hand_landmarks:
        if isinstance(lm, dict) and "index" in str(lm.get("role") or lm.get("id") or "").lower():
            try:
                return (float(lm["x"]), float(lm["y"]))
            except (KeyError, TypeError, ValueError):
                return None
    for lm in hand_landmarks:
        if isinstance(lm, dict):
            try:
                return (float(lm["x"]), float(lm["y"]))
            except (KeyError, TypeError, ValueError):
                continue
    return None


def _hand_moved(sess, hand_landmarks, threshold: float = 0.05) -> bool:
    cur = _index_card(hand_landmarks)
    prev = sess.get("last_index")
    if cur is None:
        return False
    if prev is None:
        return True
    return math.hypot(cur[0] - prev[0], cur[1] - prev[1]) > threshold


def _is_extraction(payload, frame_index) -> bool:
    if frame_index == 0:
        return True
    return bool(payload.get("extract") or payload.get("isExtraction")
                or payload.get("isExtractionFrame") or payload.get("extraction"))


def _step_markers(index_local, gesture, frame_id, timestamp_ms) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if isinstance(gesture, dict) and gesture.get("active") and index_local is not None:
        out.append({"id": f"step-{frame_id}", "label": "Pinch / grab",
                    "x": index_local[0], "y": index_local[1], "timestampMs": timestamp_ms})
    return out


# -- Build/Plan rule-based notes + plan steps (event loop) --------------------

def make_ai_fields(workflow_mode, geom, hand_out, index_local, gesture,
                   confidence, frame_index, session) -> Dict[str, Any]:
    """Rule-based, cautious notes/instructions. No ML, no exact procedures."""
    center = (geom or {}).get("center") or {"x": 0.5, "y": 0.5}
    cx, cy = center["x"], center["y"]
    hx, hy = index_local if index_local else (cx, cy)
    active = bool(gesture.get("active")) if isinstance(gesture, dict) else False
    low_conf = confidence < 0.5
    notes: List[Dict[str, Any]] = []

    def note(nid, ntype, text, x, y, conf=None):
        notes.append({"id": f"note-{frame_index}-{nid}", "type": ntype, "text": text,
                      "x": _clamp(x), "y": _clamp(y), "timestampMs": 0, "confidence": conf})

    if workflow_mode == "plan":
        note("intent", "next-step", "The user may be trying to inspect this item.", cx, cy)
        note("next", "next-step", "Possible next step: check the highlighted area.", hx, hy)
        note("safety", "safety", "Before continuing, verify the item is safe to handle.", cx, cy)
        note("quality", "quality", "Check that the area is clear before proceeding.", cx, cy)
        if low_conf:
            note("confirm", "intent",
                 "What are you trying to do with this item? Inspect, repair, clean, install, remove, or other?",
                 cx, cy, confidence)
        plan_steps, current_index = _update_plan_steps(session, center, active)
        fields = {
            "instruction": "Guidance: inspect the highlighted area and confirm the task you want to perform.",
            "nextAction": "Possible next step: check the highlighted area.",
            "safetyWarning": "Verify the item is safe to handle before continuing.",
            "qualityCheck": "Ensure the highlighted area is visible and unobstructed.",
            "activityLabel": "Planning area",
            "detectedIntent": None,
            "importance": "medium",
            "planSteps": plan_steps,
            "currentPlanStepIndex": current_index,
        }
    else:  # build
        note("obs", "observation", "The user appears to be working near this area.", hx, hy)
        if active:
            note("inspect", "observation", "Possible inspection point.", hx, hy)
        note("quality", "quality", "Check this area before finishing.", cx, cy)
        note("safety", "safety", "Safety reminder: verify the area is safe before continuing.", cx, cy)
        if low_conf:
            note("confirm", "intent",
                 "What are you trying to do with this item? Inspect, repair, clean, install, remove, or other?",
                 cx, cy, confidence)
        fields = {
            "instruction": "The user appears to be focusing on this selected object.",
            "nextAction": "Pin the blueprint, then record or follow the next step.",
            "safetyWarning": None,
            "qualityCheck": "Keep the selected object clearly visible.",
            "activityLabel": "Selected work area",
            "detectedIntent": None,
            "importance": "medium",
            "planSteps": [],
            "currentPlanStepIndex": None,
        }
    fields["aiNotes"] = notes
    return fields


def _update_plan_steps(session, center, active) -> Tuple[List[Dict[str, Any]], int]:
    """Create starter plan steps once, then advance the index on each pinch."""
    steps = session.get("plan_steps")
    if not steps:
        steps = [
            {"id": "step-1", "title": "Inspect selected area",
             "instruction": "Check the highlighted area and confirm what task you want to perform.",
             "status": "active", "x": center["x"], "y": center["y"],
             "safetyNote": "Verify the item is safe to handle before continuing.",
             "qualityCheck": "Ensure the area is visible and unobstructed."},
            {"id": "step-2", "title": "Perform work",
             "instruction": "Follow the next confirmed action while keeping the item in view.",
             "status": "pending"},
            {"id": "step-3", "title": "Final check",
             "instruction": "Verify the result and inspect the highlighted points.",
             "status": "pending"},
        ]
        session["plan_steps"] = steps
        session["plan_index"] = 0

    idx = session.get("plan_index", 0)
    if active and not session.get("pinch_prev", False):  # rising edge of a pinch
        idx = min(idx + 1, len(steps) - 1)
    session["pinch_prev"] = active
    session["plan_index"] = idx

    out = []
    for i, s in enumerate(steps):
        s2 = dict(s)
        s2["status"] = "completed" if i < idx else ("active" if i == idx else "pending")
        out.append(s2)
    return out, idx


# -- Normalization helpers ----------------------------------------------------

def _n(value: Any, size: Any) -> float:
    return float(max(0.0, min(1.0, float(value) / float(size)))) if size else 0.0


def _clamp(value: Any) -> float:
    return float(max(0.0, min(1.0, float(value))))


def _high_curvature(outline: List[Dict[str, float]], max_pts: int = 6,
                    angle_threshold: float = 110.0) -> List[Dict[str, float]]:
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
