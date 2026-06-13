"""
yolo26_loader.py -- YOLO26 (Ultralytics) backend adapter for the SafeLens worker.

Default vision backend, used in TASK-BASED MODES (never one heavy model doing
everything on every frame):

    live  (/detect HSE loop)        -> YOLO26_LIVE_TASKS  (default: det)
    build (/build/session/frame)    -> YOLO26_BUILD_TASKS (default: det,seg)
    plan  (/build/session/frame)    -> YOLO26_PLAN_TASKS  (default: det,seg)
    pose                            -> opt-in only (YOLO26_POSE_ENABLED=true or
                                       an explicit 'pose' in a task list)

Loading strategy: warmup loads ONLY the live-task models (det by default).
seg/pose models are lazy-loaded the first time a mode needs them and cached in
memory. A seg/pose load failure drops that task gracefully; a det load failure
raises so vision_backend can auto-fall back to EdgeCrafter.

The app-facing /detect contract is unchanged: entities (normalized 0..1 bbox
x/y/w/h), poses (COCO-17), backend/tasks/model/inference_ms/img_w/img_h.
Optional additive fields only (source, segments/maskContour).

Model ids are env-configurable (YOLO26_DET_MODEL_ID / _SEG_ / _POSE_; legacy
YOLO26_MODEL_ID still honored) so compatible Ultralytics ids can be used while
keeping the backend name "yolo26". Weights resolve from:
  1. YOLO26_CACHE_DIR/<model_id>     (RunPod volume cache)
  2. /app/models/yolo26/<model_id>   (baked into the image, optional)
  3. bare model id                   (Ultralytics auto-download into the cache)
"""

from __future__ import annotations

import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)

# -- Environment --------------------------------------------------------------

def _env(name, default=""):
    return os.environ.get(name, default)

def _env_float(name, default):
    """Parse a float env var, falling back safely on missing/invalid values."""
    try:
        return float(_env(name, str(default)))
    except (TypeError, ValueError):
        return float(default)

def _env_int(name, default):
    """Parse an int env var (via float, so '300.0' works), safe on invalid."""
    try:
        return int(float(_env(name, str(default))))
    except (TypeError, ValueError):
        return int(default)

def _cache_dir():
    return _env("YOLO26_CACHE_DIR", "/runpod-volume/models/yolo26")

BAKED_DIR = "/app/models/yolo26"

def _model_id(task):
    if task == "seg":
        return _env("YOLO26_SEG_MODEL_ID", "yolo26n-seg.pt")
    if task == "pose":
        return _env("YOLO26_POSE_MODEL_ID", "yolo26n-pose.pt")
    # det: new name first, legacy YOLO26_MODEL_ID still honored.
    return _env("YOLO26_DET_MODEL_ID", "") or _env("YOLO26_MODEL_ID", "yolo26n.pt")


def pose_enabled():
    return _env("YOLO26_POSE_ENABLED", "false").strip().lower() in ("1", "true", "yes", "on")


def _parse_task_list(raw, default):
    out = []
    for tok in (raw or default).split(","):
        t = tok.strip().lower()
        if t in ("det", "seg", "pose") and t not in out:
            out.append(t)
    return out or ["det"]


def mode_tasks(mode="live"):
    """Task list for a mode. pose only with YOLO26_POSE_ENABLED or explicit."""
    if mode == "build":
        tasks = _parse_task_list(_env("YOLO26_BUILD_TASKS"), "det,seg")
    elif mode == "plan":
        tasks = _parse_task_list(_env("YOLO26_PLAN_TASKS"), "det,seg")
    else:  # live -- legacy YOLO26_TASKS honored as an alias
        tasks = _parse_task_list(_env("YOLO26_LIVE_TASKS") or _env("YOLO26_TASKS"), "det")
        if pose_enabled() and "pose" not in tasks:
            tasks.append("pose")
    if "pose" in tasks and not pose_enabled():
        # explicit 'pose' in the env list counts as opting in for that mode
        explicit = "pose" in (_env({"build": "YOLO26_BUILD_TASKS",
                                    "plan": "YOLO26_PLAN_TASKS"}.get(mode, "YOLO26_LIVE_TASKS"), "")
                              or _env("YOLO26_TASKS", "")).lower()
        if not explicit:
            tasks = [t for t in tasks if t != "pose"]
    return tasks


def parse_tasks(value=None):
    """Back-compat helper: live-mode tasks (or parse an explicit value)."""
    if value is not None:
        return _parse_task_list(value, "det")
    return mode_tasks("live")


def resolve_device(pref=None):
    pref = (pref or _env("YOLO26_DEVICE", "cuda")).lower()
    if pref == "cuda":
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
        except Exception:  # noqa: BLE001
            pass
        return "cpu"
    return pref or "cpu"


# -- COCO labels / keypoints (kept local so the adapter stays standalone) ------

COCO_KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]
_COCO_SKELETON_1BASED = [
    (16, 14), (14, 12), (17, 15), (15, 13), (12, 13),
    (6, 12), (7, 13), (6, 7), (6, 8), (7, 9),
    (8, 10), (9, 11), (2, 3), (1, 2), (1, 3),
    (2, 4), (3, 5), (4, 6), (5, 7),
]
COCO_SKELETON = [[a - 1, b - 1] for a, b in _COCO_SKELETON_1BASED]


# -- Pure normalization helpers (unit-testable without ultralytics) ------------

def _clamp01(v):
    return float(max(0.0, min(1.0, v)))


def normalize_bbox_xyxy(x1, y1, x2, y2, img_w, img_h):
    """xyxy pixels (original image coords) -> normalized {x, y, w, h} in 0..1."""
    nx = _clamp01(x1 / img_w)
    ny = _clamp01(y1 / img_h)
    nw = max(0.0, min(1.0 - nx, (x2 - x1) / img_w))
    nh = max(0.0, min(1.0 - ny, (y2 - y1) / img_h))
    return {"x": nx, "y": ny, "w": nw, "h": nh}


def normalize_detections(boxes_xyxy, class_ids, scores, names, img_w, img_h,
                         class_filter=None, source="yolo26"):
    """Plain arrays -> SafeLens entity dicts (normalized bbox, label, class_id)."""
    out = []
    for (x1, y1, x2, y2), cid, sc in zip(boxes_xyxy, class_ids, scores):
        cid = int(cid)
        if class_filter is not None and cid not in class_filter:
            continue
        label = names.get(cid, f"class_{cid}") if isinstance(names, dict) else f"class_{cid}"
        out.append({
            "label": str(label),
            "class_id": cid,
            "confidence": float(sc),
            "bbox": normalize_bbox_xyxy(float(x1), float(y1), float(x2), float(y2), img_w, img_h),
            "source": source,
        })
    return out


def normalize_poses(kpts_xy, kpts_conf, person_scores, img_w, img_h,
                    source="yolo26-pose"):
    """Per-person keypoint pixel arrays -> SafeLens pose dicts (COCO-17)."""
    out = []
    for i, person in enumerate(kpts_xy):
        kp_list = []
        for j, (px, py) in enumerate(person):
            if j >= len(COCO_KEYPOINT_NAMES):
                break
            score = 0.0
            if kpts_conf is not None and i < len(kpts_conf) and j < len(kpts_conf[i]):
                score = float(kpts_conf[i][j])
            kp_list.append({
                "name": COCO_KEYPOINT_NAMES[j],
                "x": _clamp01(float(px) / img_w),
                "y": _clamp01(float(py) / img_h),
                "score": score,
            })
        conf = float(person_scores[i]) if person_scores is not None and i < len(person_scores) else 0.0
        out.append({
            "label": "person",
            "confidence": conf,
            "keypoints": kp_list,
            "skeleton": COCO_SKELETON,
            "source": source,
        })
    return out


def normalize_segments(polygons_xy, class_ids, scores, names, img_w, img_h,
                       source="yolo26-seg", max_points=120):
    """Per-instance mask polygons (pixel coords) -> optional segment dicts."""
    out = []
    for i, poly in enumerate(polygons_xy):
        if poly is None or len(poly) < 3:
            continue
        stride = max(1, len(poly) // max_points)
        contour = [{"x": _clamp01(float(px) / img_w), "y": _clamp01(float(py) / img_h)}
                   for px, py in poly[::stride]]
        cid = int(class_ids[i]) if class_ids is not None and i < len(class_ids) else -1
        label = names.get(cid, f"class_{cid}") if isinstance(names, dict) else f"class_{cid}"
        out.append({
            "label": str(label),
            "class_id": cid,
            "confidence": float(scores[i]) if scores is not None and i < len(scores) else 0.0,
            "maskContour": contour,
            "source": source,
        })
    return out


# -- Weight resolution ----------------------------------------------------------

def resolve_weights(model_id):
    """Cache dir -> baked dir -> bare id (Ultralytics auto-download)."""
    for root in (_cache_dir(), BAKED_DIR):
        p = Path(root) / model_id
        try:
            if p.exists() and p.stat().st_size > 0:
                return str(p)
        except Exception:  # noqa: BLE001
            continue
    return model_id  # bare name -- Ultralytics downloads it


# -- Lazy singleton state --------------------------------------------------------

class _YoloState:
    def __init__(self):
        self.device = None
        self.models = {}           # task -> YOLO model (lazy-filled)
        self.failed = {}           # task -> error string (lazy-load failures)
        self.warnings = []
        self.loaded = False        # live (det) warmup completed

_STATE = _YoloState()
_LOAD_LOCK = threading.Lock()


def _load_one(task):
    """Load one task model. Downloads bare ids into the cache dir."""
    from ultralytics import YOLO
    model_id = _model_id(task)
    weights = resolve_weights(model_id)
    if weights == model_id:  # bare id -> download into the cache dir
        cache = _cache_dir()
        Path(cache).mkdir(parents=True, exist_ok=True)
        old_cwd = os.getcwd()
        try:
            os.chdir(cache)
            model = YOLO(model_id)
        finally:
            os.chdir(old_cwd)
    else:
        model = YOLO(weights)
    log.info("[yolo26] %s model ready: %s", task, model_id)
    return model


def ensure_task(task):
    """Lazily load (and cache) one task's model. Returns it, or None on failure.

    Used so seg/pose only load the first time Build/Plan (or pose opt-in)
    actually needs them. Failures are recorded once and not retried per frame.
    """
    if task in _STATE.models:
        return _STATE.models[task]
    if task in _STATE.failed:
        return None
    with _LOAD_LOCK:
        if task in _STATE.models:
            return _STATE.models[task]
        if task in _STATE.failed:
            return None
        try:
            if _STATE.device is None:
                _STATE.device = resolve_device()
            _STATE.models[task] = _load_one(task)
            return _STATE.models[task]
        except Exception as exc:  # noqa: BLE001
            msg = f"{type(exc).__name__}: {exc}"
            _STATE.failed[task] = msg
            _STATE.warnings.append(f"{task} model load failed: {msg}")
            log.warning("[yolo26] lazy %s load failed: %s", task, msg)
            return None


def load(tasks=None, device=None):
    """Warmup: load ONLY the live-task models (det by default). Idempotent.

    'det' is required and re-raises on failure (so the dispatcher can fall back
    to EdgeCrafter). Other live tasks (e.g. opt-in pose) degrade with a warning.
    seg/pose for Build/Plan stay lazy until first use.
    """
    if _STATE.loaded:
        return model_summary()

    live = tasks or mode_tasks("live")
    _STATE.device = device or resolve_device()
    log.info("[yolo26] warmup: live tasks=%s device=%s (seg/pose lazy)", live, _STATE.device)

    for task in live:
        if task == "det":
            _STATE.models["det"] = _load_one("det")  # required -- raises on failure
        elif ensure_task(task) is None:
            log.warning("[yolo26] continuing without optional live task %s", task)

    if "det" not in _STATE.models:
        raise RuntimeError("yolo26: det model not loaded")
    _STATE.loaded = True
    return model_summary()


def model_summary():
    return {
        "backend": "yolo26",
        "tasks_loaded": sorted(_STATE.models.keys()),
        "model_classes": {t: type(m).__name__ for t, m in _STATE.models.items()},
        "model_ids": {t: _model_id(t) for t in ("det", "seg", "pose")},
        "device": _STATE.device,
        "warnings": list(_STATE.warnings),
    }


def status():
    """Mode/task + per-model load status block for /debug/state (no secrets)."""
    return {
        "det_loaded": "det" in _STATE.models,
        "seg_loaded": "seg" in _STATE.models,
        "pose_loaded": "pose" in _STATE.models,
        "load_failures": dict(_STATE.failed),
        "live_tasks": mode_tasks("live"),
        "build_tasks": mode_tasks("build"),
        "plan_tasks": mode_tasks("plan"),
        "pose_enabled": pose_enabled(),
        "det_model_id": _model_id("det"),
        "seg_model_id": _model_id("seg"),
        "pose_model_id": _model_id("pose"),
        "device": _STATE.device,
    }


def is_ready():
    return _STATE.loaded


# -- Inference --------------------------------------------------------------------

def _predict(task, pil_img, conf, img_size, iou=None, max_det=None):
    """Run one task model on a PIL image (RGB-safe) and return the Result.

    iou / max_det fall back to YOLO26_IOU / YOLO26_MAX_DETECTIONS when not
    provided, so the Ultralytics call always gets concrete NMS parameters.
    """
    model = _STATE.models[task]
    if iou is None:
        iou = _env_float("YOLO26_IOU", 0.45)
    if max_det is None:
        max_det = _env_int("YOLO26_MAX_DETECTIONS", 300)
    results = model(pil_img, conf=conf, imgsz=img_size, iou=iou, max_det=max_det,
                    device=_STATE.device, verbose=False)
    return results[0]


def infer(pil_img, conf, class_filter=None, tasks=None,
          img_size=None, iou=None, max_det=None):
    """Run the given tasks (default: LIVE tasks) on a frame.

    /detect speed protection: by default only det runs (and pose only when
    opted in); seg never runs here unless explicitly listed in the live tasks.
    img_size / iou / max_det come from the resolved /detect config; each falls
    back to its YOLO26_* env var when None.
    """
    img_w, img_h = pil_img.size
    if img_size is None:
        img_size = _env_int("YOLO26_IMG_SIZE", 640)
    run_tasks = list(tasks) if tasks else mode_tasks("live")
    t0 = time.perf_counter()
    entities: List[Dict[str, Any]] = []
    poses: List[Dict[str, Any]] = []
    segments: List[Dict[str, Any]] = []
    ran: List[str] = []

    if "det" in run_tasks and "det" in _STATE.models:
        res = _predict("det", pil_img, conf, img_size, iou=iou, max_det=max_det)
        ran.append("det")
        if res.boxes is not None and len(res.boxes):
            entities = normalize_detections(
                res.boxes.xyxy.cpu().numpy(),
                res.boxes.cls.cpu().numpy(),
                res.boxes.conf.cpu().numpy(),
                getattr(res, "names", {}) or {},
                img_w, img_h, class_filter, source="yolo26",
            )

    if "pose" in run_tasks and ensure_task("pose") is not None:
        res = _predict("pose", pil_img, conf, img_size, iou=iou, max_det=max_det)
        ran.append("pose")
        kpts = getattr(res, "keypoints", None)
        if kpts is not None and kpts.xy is not None and len(kpts.xy):
            person_scores = None
            if res.boxes is not None and len(res.boxes):
                person_scores = res.boxes.conf.cpu().numpy()
            kconf = kpts.conf.cpu().numpy() if kpts.conf is not None else None
            poses = normalize_poses(
                kpts.xy.cpu().numpy(), kconf, person_scores,
                img_w, img_h, source="yolo26-pose",
            )

    if "seg" in run_tasks and ensure_task("seg") is not None:
        res = _predict("seg", pil_img, conf, img_size, iou=iou, max_det=max_det)
        ran.append("seg")
        masks = getattr(res, "masks", None)
        if masks is not None and masks.xy:
            cls = res.boxes.cls.cpu().numpy() if res.boxes is not None else None
            scs = res.boxes.conf.cpu().numpy() if res.boxes is not None else None
            segments = normalize_segments(
                masks.xy, cls, scs, getattr(res, "names", {}) or {},
                img_w, img_h, source="yolo26-seg",
            )

    ms = (time.perf_counter() - t0) * 1000.0
    return {
        "entities": entities,
        "poses": poses,
        "segments": segments,
        "inference_ms": round(ms, 2),
        "tasks": ran,
        "model": "YOLO26",
    }


# -- Selected-crop analysis for Build/Plan Mode -----------------------------------

def crop_analysis(pil_img, conf, mode="build"):
    """Run the mode's tasks on a SELECTED CROP only (never the live frame).

    Returns {ok, mask_contour, mask_source, confidence, parts}:
      mask_contour -- normalized contour of the largest seg instance (or [])
      parts        -- detected objects in the crop (normalized bboxes) for
                      Plan Mode visual grounding.
    Never raises: any failure returns ok=False so Build Mode's fallback
    contour pipeline takes over.
    """
    out = {"ok": False, "mask_contour": [], "mask_source": "none",
           "confidence": 0.0, "parts": []}
    try:
        if not _STATE.loaded:
            return out  # only piggyback on an already-warmed yolo26 backend
        img_w, img_h = pil_img.size
        img_size = _env_int("YOLO26_IMG_SIZE", 640)
        tasks = mode_tasks("build" if mode == "build" else "plan")

        if "det" in tasks and "det" in _STATE.models:
            res = _predict("det", pil_img, conf, img_size)
            if res.boxes is not None and len(res.boxes):
                out["parts"] = normalize_detections(
                    res.boxes.xyxy.cpu().numpy(),
                    res.boxes.cls.cpu().numpy(),
                    res.boxes.conf.cpu().numpy(),
                    getattr(res, "names", {}) or {},
                    img_w, img_h, None, source="yolo26",
                )

        if "seg" in tasks and ensure_task("seg") is not None:
            res = _predict("seg", pil_img, conf, img_size)
            masks = getattr(res, "masks", None)
            if masks is not None and masks.xy:
                cls = res.boxes.cls.cpu().numpy() if res.boxes is not None else None
                scs = res.boxes.conf.cpu().numpy() if res.boxes is not None else None
                segs = normalize_segments(masks.xy, cls, scs,
                                          getattr(res, "names", {}) or {},
                                          img_w, img_h, source="yolo26-seg")
                if segs:
                    best = max(segs, key=lambda s: s["confidence"])
                    out["mask_contour"] = best["maskContour"]
                    out["mask_source"] = "yolo26-seg"
                    out["confidence"] = best["confidence"]

        out["ok"] = bool(out["mask_contour"] or out["parts"])
        return out
    except Exception as exc:  # noqa: BLE001 -- crop analysis must never break a frame
        log.warning("[yolo26] crop_analysis failed: %s", exc)
        return out
