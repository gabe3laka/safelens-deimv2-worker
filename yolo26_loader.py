"""
yolo26_loader.py -- YOLO26 (Ultralytics) backend adapter for the SafeLens worker.

Default vision backend as of the yolo26 migration:

    VISION_BACKEND=yolo26 (default) -> this adapter
    VISION_BACKEND=edgecrafter      -> EdgeCrafter fallback (unchanged)
    VISION_BACKEND=deimv2           -> legacy debug fallback (unchanged)

The adapter keeps the exact app-facing output contract used by /detect:
entities (normalized 0..1 bbox x/y/w/h), poses (COCO-17 keypoints, normalized),
plus an OPTIONAL `segments` list ({maskContour, source: "yolo26-seg"}) when the
seg task is enabled -- additive only, the app may ignore it.

Model ids are env-configurable so the same adapter keeps working whether the
installed Ultralytics ships true YOLO26 weights or only an earlier YOLO family
(set YOLO26_MODEL_ID etc. to any available .pt). Weights are resolved from:
  1. YOLO26_CACHE_DIR/<model_id>     (RunPod volume cache)
  2. /app/models/yolo26/<model_id>   (baked into the image, optional)
  3. bare model id                   (Ultralytics auto-download into the cache)

Failure policy: the 'det' model is required -- a det load failure raises so
vision_backend can auto-fall back to EdgeCrafter. 'seg'/'pose' failures only
drop that task (recorded in the load summary) and never sink the backend.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)

# -- Environment --------------------------------------------------------------

def _env(name, default=""):
    return os.environ.get(name, default)

def _cache_dir():
    return _env("YOLO26_CACHE_DIR", "/runpod-volume/models/yolo26")

BAKED_DIR = "/app/models/yolo26"

def _model_id(task):
    if task == "seg":
        return _env("YOLO26_SEG_MODEL_ID", "yolo26n-seg.pt")
    if task == "pose":
        return _env("YOLO26_POSE_MODEL_ID", "yolo26n-pose.pt")
    return _env("YOLO26_MODEL_ID", "yolo26n.pt")


def parse_tasks(value=None):
    """Parse YOLO26_TASKS into an ordered, de-duplicated task list.

    Supports det, seg, pose. Defaults to "det,pose" -- boxes and poses are the
    required first outputs; seg is opt-in (YOLO26_TASKS=det,seg,pose).
    """
    raw = value if value is not None else _env("YOLO26_TASKS", "det,pose")
    out = []
    for tok in raw.split(","):
        t = tok.strip().lower()
        if t in ("det", "seg", "pose") and t not in out:
            out.append(t)
    return out or ["det"]


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
        self.tasks = []            # requested tasks
        self.loaded_tasks = []     # tasks whose model actually loaded
        self.models = {}           # task -> YOLO model
        self.warnings = []
        self.loaded = False

_STATE = _YoloState()


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


def load(tasks=None, device=None):
    """Load YOLO models for the enabled tasks. Idempotent.

    'det' is required and re-raises on failure (so the dispatcher can fall back
    to EdgeCrafter). 'seg'/'pose' failures drop the task with a warning.
    """
    if _STATE.loaded:
        return model_summary()

    _STATE.tasks = tasks or parse_tasks()
    _STATE.device = device or resolve_device()
    log.info("[yolo26] loading tasks=%s device=%s", _STATE.tasks, _STATE.device)

    for task in _STATE.tasks:
        try:
            _STATE.models[task] = _load_one(task)
            _STATE.loaded_tasks.append(task)
        except Exception as exc:  # noqa: BLE001
            msg = f"{task} model load failed: {type(exc).__name__}: {exc}"
            if task == "det":
                log.error("[yolo26] %s", msg)
                raise
            _STATE.warnings.append(msg)
            log.warning("[yolo26] %s -- continuing without %s", msg, task)

    if not _STATE.loaded_tasks:
        raise RuntimeError("yolo26: no models loaded")
    _STATE.loaded = True
    return model_summary()


def model_summary():
    return {
        "backend": "yolo26",
        "tasks_loaded": list(_STATE.loaded_tasks),
        "model_classes": {t: type(m).__name__ for t, m in _STATE.models.items()},
        "model_ids": {t: _model_id(t) for t in _STATE.tasks},
        "device": _STATE.device,
        "warnings": list(_STATE.warnings),
    }


def is_ready():
    return _STATE.loaded


# -- Inference --------------------------------------------------------------------

def _predict(task, pil_img, conf, img_size):
    """Run one task model on a PIL image (RGB-safe) and return the Result."""
    model = _STATE.models[task]
    results = model(pil_img, conf=conf, imgsz=img_size, device=_STATE.device,
                    verbose=False)
    return results[0]


def infer(pil_img, conf, class_filter=None):
    """Run all enabled tasks; return normalized entities/poses/segments."""
    img_w, img_h = pil_img.size
    img_size = int(float(_env("YOLO26_IMG_SIZE", "640")))
    t0 = time.perf_counter()
    entities: List[Dict[str, Any]] = []
    poses: List[Dict[str, Any]] = []
    segments: List[Dict[str, Any]] = []

    if "det" in _STATE.models:
        res = _predict("det", pil_img, conf, img_size)
        if res.boxes is not None and len(res.boxes):
            entities = normalize_detections(
                res.boxes.xyxy.cpu().numpy(),
                res.boxes.cls.cpu().numpy(),
                res.boxes.conf.cpu().numpy(),
                getattr(res, "names", {}) or {},
                img_w, img_h, class_filter, source="yolo26",
            )

    if "pose" in _STATE.models:
        res = _predict("pose", pil_img, conf, img_size)
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

    if "seg" in _STATE.models:
        res = _predict("seg", pil_img, conf, img_size)
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
        "tasks": list(_STATE.loaded_tasks),
        "model": "YOLO26",
    }
