"""
edgecrafter_loader.py -- Official EdgeCrafter model loader for the SafeLens worker.

EdgeCrafter (https://github.com/Intellindust-AI-Lab/EdgeCrafter) is the
successor to DEIMv2. It ships task-specialized compact ViTs:

  ECDet-S  -> object / person detection (boxes)         [ecdetseg subtree]
  ECPose-S -> human pose estimation (COCO-17 keypoints) [ecpose subtree]
  ECSeg-S  -> instance segmentation (masks)             [optional, later]

Both detection and pose are built with the upstream YAMLConfig engine and
loaded from a config (.yml) + checkpoint (.pth). The two subtrees ship their
own engine package, so each must be imported with the correct subtree on
sys.path. This module mirrors the official inference scripts:

  ecdetseg/tools/inference/torch_inf.py  -> (labels, boxes, scores)
  ecpose/tools/inference/torch_inf.py    -> (scores, labels, keypoints)

Preprocessing for both is the official transform:
  Resize(eval_spatial_size) + ToTensor() + ImageNet Normalize.

The postprocessor is fed the *original* image size, so boxes/keypoints come
back in original-image pixel coordinates -- we normalize to 0..1 against the
true image width/height (NOT the model input size).

Weights are NOT baked into the image. They are downloaded at runtime into
EDGECRAFTER_CACHE_DIR (a RunPod volume) if missing.
"""

from __future__ import annotations

import logging
import os
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image

log = logging.getLogger(__name__)

# -- Environment --------------------------------------------------------------

def _env(name, default=""):
    return os.environ.get(name, default)

REPO_DIR = _env("EDGECRAFTER_REPO_DIR", "/opt/EdgeCrafter")
CACHE_DIR = _env("EDGECRAFTER_CACHE_DIR", "/runpod-volume/models/edgecrafter")

DET_CONFIG = _env("EDGECRAFTER_DET_CONFIG", REPO_DIR + "/ecdetseg/configs/ecdet/ecdet_s.yml")
DET_CKPT_URL = _env(
    "EDGECRAFTER_DET_CHECKPOINT_URL",
    "https://github.com/capsule2077/edgecrafter/releases/download/edgecrafterv1/ecdet_s.pth",
)
DET_CKPT_PATH = _env("EDGECRAFTER_DET_CHECKPOINT_PATH", CACHE_DIR + "/ecdet_s.pth")

POSE_CONFIG = _env("EDGECRAFTER_POSE_CONFIG", REPO_DIR + "/ecpose/configs/ecpose/ecpose_s_coco.yml")
POSE_CKPT_URL = _env(
    "EDGECRAFTER_POSE_CHECKPOINT_URL",
    "https://github.com/capsule2077/edgecrafter/releases/download/edgecrafterv1/ecpose_s.pth",
)
POSE_CKPT_PATH = _env("EDGECRAFTER_POSE_CHECKPOINT_PATH", CACHE_DIR + "/ecpose_s.pth")


def parse_tasks(value=None):
    """Parse EDGECRAFTER_TASKS into an ordered, de-duplicated task list.

    Accepts e.g. "det,pose" / "det, pose" / "DET". Only 'det' and 'pose' are
    supported in this PR ('seg' is reserved for a later ECSeg-S layer).
    """
    raw = value if value is not None else _env("EDGECRAFTER_TASKS", "det,pose")
    out = []
    for tok in raw.split(","):
        t = tok.strip().lower()
        if t in ("det", "pose") and t not in out:
            out.append(t)
    return out or ["det"]


def resolve_device(pref=None):
    pref = (pref or _env("EDGECRAFTER_DEVICE", "cuda")).lower()
    if pref == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# -- COCO label / keypoint maps ----------------------------------------------

# 0-indexed COCO-80 detection labels (EdgeCrafter postprocessor emits 0..79).
COCO_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]

def coco_label(class_id_0_indexed):
    cid = int(class_id_0_indexed)
    if 0 <= cid < len(COCO_NAMES):
        return COCO_NAMES[cid]
    return "class_" + str(cid)

# COCO-17 keypoint names, in the canonical order produced by the model.
COCO_KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]

# COCO skeleton edges as 0-based index pairs (matches upstream torch_inf.py).
_COCO_SKELETON_1BASED = [
    (16, 14), (14, 12), (17, 15), (15, 13), (12, 13),
    (6, 12), (7, 13), (6, 7), (6, 8), (7, 9),
    (8, 10), (9, 11), (2, 3), (1, 2), (1, 3),
    (2, 4), (3, 5), (4, 6), (5, 7),
]
COCO_SKELETON = [[a - 1, b - 1] for a, b in _COCO_SKELETON_1BASED]


# -- Checkpoint download ------------------------------------------------------

def ensure_checkpoint(url, path):
    """Download url to path if the file does not already exist.

    Returns the local path. Skips the download when the file is already
    present and non-empty. Creates parent directories as needed.
    """
    p = Path(path)
    if p.exists() and p.stat().st_size > 0:
        log.info("[edgecrafter] checkpoint present, skipping download: %s (%d bytes)",
                 path, p.stat().st_size)
        return str(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".part")
    log.info("[edgecrafter] downloading checkpoint %s -> %s", url, path)
    urllib.request.urlretrieve(url, tmp)
    tmp.replace(p)
    log.info("[edgecrafter] downloaded %s (%d bytes)", path, p.stat().st_size)
    return str(p)


# -- sys.path wiring ----------------------------------------------------------

def _add_subtree_to_path(subtree):
    """Prepend an EdgeCrafter subtree (ecdetseg|ecpose) to sys.path."""
    root = os.path.join(REPO_DIR, subtree)
    if root not in sys.path:
        sys.path.insert(0, root)


def _purge_engine_modules():
    """Drop cached engine.* modules so the other subtree can be imported.

    ecdetseg and ecpose both define a top-level engine package; Python caches
    the first one imported. Purging lets us load the second cleanly.
    """
    for name in list(sys.modules):
        if name == "engine" or name.startswith("engine."):
            del sys.modules[name]


def _build_transforms(size):
    """Official EdgeCrafter preprocessing: Resize + ToTensor + ImageNet norm."""
    return T.Compose([
        T.Resize(size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


# -- Detection model ----------------------------------------------------------

def build_det_model(config_path, ckpt_path, device):
    """Build ECDet-S exactly as ecdetseg/tools/inference/torch_inf.py does."""
    _purge_engine_modules()
    _add_subtree_to_path("ecdetseg")
    from engine.core import YAMLConfig  # type: ignore

    cfg = YAMLConfig(config_path, resume=ckpt_path)
    if "ViTAdapter" in cfg.yaml_cfg:
        cfg.yaml_cfg["ViTAdapter"]["skip_load_backbone"] = True

    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    state = checkpoint["ema"]["module"] if "ema" in checkpoint else checkpoint["model"]
    cfg.model.load_state_dict(state)

    class DetModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()

        def forward(self, images, orig_target_sizes):
            return self.postprocessor(self.model(images), orig_target_sizes)

    model = DetModel().to(device)
    model.eval()
    size = cfg.yaml_cfg["eval_spatial_size"]  # [h, w]
    return model, tuple(size)


# -- Pose model ---------------------------------------------------------------

def build_pose_model(config_path, ckpt_path, device):
    """Build ECPose-S exactly as ecpose/tools/inference/torch_inf.py does."""
    _purge_engine_modules()
    _add_subtree_to_path("ecpose")
    from engine.core import YAMLConfig  # type: ignore

    cfg = YAMLConfig(config_path, resume=ckpt_path)
    if "ViTAdapter" in cfg.yaml_cfg:
        cfg.yaml_cfg["ViTAdapter"]["skip_load_backbone"] = True

    # Pose checkpoints require weights_only=False (upstream torch_inf.py).
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = checkpoint["ema"]["module"] if "ema" in checkpoint else checkpoint["model"]
    cfg.model.load_state_dict(state)

    class PoseModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()

        def forward(self, images, orig_target_sizes):
            return self.postprocessor(self.model(images), orig_target_sizes)

    model = PoseModel().to(device)
    model.eval()
    size = cfg.yaml_cfg["eval_spatial_size"]  # [h, w]
    return model, tuple(size)


# -- Lazy singleton state -----------------------------------------------------

class _ECState:
    def __init__(self):
        self.device = None
        self.tasks = []
        self.det_model = None
        self.det_size = None
        self.pose_model = None
        self.pose_size = None
        self.loaded = False

_STATE = _ECState()


def load(tasks=None, device=None):
    """Lazily load EdgeCrafter models for the enabled tasks.

    Idempotent: a second call is a no-op once loaded. Returns a structured
    summary suitable for /debug/model-load.
    """
    if _STATE.loaded:
        return model_summary()

    _STATE.tasks = tasks or parse_tasks()
    _STATE.device = device or resolve_device()
    log.info("[edgecrafter] loading tasks=%s device=%s", _STATE.tasks, _STATE.device)

    if "det" in _STATE.tasks:
        ensure_checkpoint(DET_CKPT_URL, DET_CKPT_PATH)
        _STATE.det_model, _STATE.det_size = build_det_model(DET_CONFIG, DET_CKPT_PATH, _STATE.device)
        log.info("[edgecrafter] ECDet-S ready, input size=%s", _STATE.det_size)

    if "pose" in _STATE.tasks:
        ensure_checkpoint(POSE_CKPT_URL, POSE_CKPT_PATH)
        _STATE.pose_model, _STATE.pose_size = build_pose_model(POSE_CONFIG, POSE_CKPT_PATH, _STATE.device)
        log.info("[edgecrafter] ECPose-S ready, input size=%s", _STATE.pose_size)

    _STATE.loaded = True
    return model_summary()


def model_summary():
    tasks_loaded = []
    classes = {}
    if _STATE.det_model is not None:
        tasks_loaded.append("det")
        classes["det"] = type(_STATE.det_model).__name__
    if _STATE.pose_model is not None:
        tasks_loaded.append("pose")
        classes["pose"] = type(_STATE.pose_model).__name__
    return {
        "backend": "edgecrafter",
        "tasks_loaded": tasks_loaded,
        "model_classes": classes,
        "checkpoint_paths": {"det": DET_CKPT_PATH, "pose": POSE_CKPT_PATH},
        "device": str(_STATE.device) if _STATE.device else None,
    }


def is_ready():
    return _STATE.loaded


# -- Normalization helpers ----------------------------------------------------

def normalize_bbox_xyxy(x1, y1, x2, y2, img_w, img_h):
    """Convert an xyxy pixel box (original image coords) to normalized x,y,w,h."""
    nx = max(0.0, min(1.0, x1 / img_w))
    ny = max(0.0, min(1.0, y1 / img_h))
    nw = max(0.0, min(1.0 - nx, (x2 - x1) / img_w))
    nh = max(0.0, min(1.0 - ny, (y2 - y1) / img_h))
    return {"x": nx, "y": ny, "w": nw, "h": nh}


def _decode_keypoints(arr):
    """Decode model keypoints into (xy[K,2], score[K]). Supports [K,2]/[K,3]/flat."""
    import numpy as np
    kpts = np.asarray(arr)
    if kpts.ndim == 2:
        xy = kpts[:, :2].astype(float)
        sc = kpts[:, 2].astype(float) if kpts.shape[1] >= 3 else np.ones((kpts.shape[0],))
        return xy, sc
    if kpts.ndim == 1:
        if kpts.size % 3 == 0:
            k = kpts.reshape(-1, 3)
            return k[:, :2].astype(float), k[:, 2].astype(float)
        if kpts.size % 2 == 0:
            k = kpts.reshape(-1, 2)
            return k.astype(float), np.ones((k.shape[0],))
    return np.zeros((0, 2)), np.zeros((0,))


# -- Inference ----------------------------------------------------------------

@torch.no_grad()
def run_detection(pil_img, conf, class_filter=None):
    """Run ECDet-S and return normalized SafeLens entity dicts."""
    img_w, img_h = pil_img.size
    tf = _build_transforms(_STATE.det_size)
    tensor = tf(pil_img).unsqueeze(0).to(_STATE.device)
    orig_sizes = torch.tensor([[img_w, img_h]], device=_STATE.device)
    labels, boxes, scores = _STATE.det_model(tensor, orig_sizes)

    keep = scores[0] > conf
    lbls = labels[0][keep]
    bxs = boxes[0][keep]
    scs = scores[0][keep]

    out = []
    for i in range(len(lbls)):
        cid = int(lbls[i].item())
        if class_filter is not None and cid not in class_filter:
            continue
        x1, y1, x2, y2 = [float(v) for v in bxs[i].tolist()]
        out.append({
            "label": coco_label(cid),
            "class_id": cid,
            "confidence": float(scs[i].item()),
            "bbox": normalize_bbox_xyxy(x1, y1, x2, y2, img_w, img_h),
            "source": "edgecrafter-det",
        })
    return out


@torch.no_grad()
def run_pose(pil_img, conf):
    """Run ECPose-S and return normalized SafeLens pose dicts (COCO-17)."""
    img_w, img_h = pil_img.size
    tf = _build_transforms(_STATE.pose_size)
    tensor = tf(pil_img).unsqueeze(0).to(_STATE.device)
    orig_sizes = torch.tensor([[img_w, img_h]], device=_STATE.device, dtype=torch.int64)
    # Pose output order is (scores, labels, keypoints) -- scores first.
    scores, labels, keypoints = _STATE.pose_model(tensor, orig_sizes)

    keep = scores[0] > conf
    scs = scores[0][keep]
    kps = keypoints[0][keep]

    out = []
    for i in range(len(scs)):
        xy, kp_scores = _decode_keypoints(kps[i].detach().cpu().numpy())
        kp_list = []
        for j in range(min(len(xy), len(COCO_KEYPOINT_NAMES))):
            kp_list.append({
                "name": COCO_KEYPOINT_NAMES[j],
                "x": max(0.0, min(1.0, float(xy[j][0]) / img_w)),
                "y": max(0.0, min(1.0, float(xy[j][1]) / img_h)),
                "score": float(kp_scores[j]) if j < len(kp_scores) else 0.0,
            })
        out.append({
            "label": "person",
            "confidence": float(scs[i].item()),
            "keypoints": kp_list,
            "skeleton": COCO_SKELETON,
            "source": "edgecrafter-pose",
        })
    return out


def infer(pil_img, conf, class_filter=None):
    """Run all enabled tasks and return {entities, poses, inference_ms}."""
    t0 = time.perf_counter()
    entities = []
    poses = []
    if "det" in _STATE.tasks and _STATE.det_model is not None:
        entities = run_detection(pil_img, conf, class_filter)
    if "pose" in _STATE.tasks and _STATE.pose_model is not None:
        poses = run_pose(pil_img, conf)
    ms = (time.perf_counter() - t0) * 1000.0
    return {"entities": entities, "poses": poses, "inference_ms": round(ms, 2)}
