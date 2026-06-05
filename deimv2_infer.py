"""
deimv2_infer.py -- DEIMv2 inference wrapper for SafeLens RunPod worker.

Handles lazy model loading, image decode, forward pass, normalised output.

Environment variables:
  DEIMV2_MODEL_ID   HuggingFace model id  Default: Intellindust-AI-Lab/DEIMv2-S
  DEIMV2_DEVICE     cuda | cpu            Default: cuda (auto-falls-back to cpu)
  DEIMV2_CONF       Confidence threshold  Default: 0.35
  DEIMV2_IMG_SIZE   Shorter-side resize   Default: 640
"""

from __future__ import annotations
import base64, io, logging, os, threading, time
from typing import List, Optional

import numpy as np
import torch
from PIL import Image
from schema import BBox, Entity, InferResponse

log = logging.getLogger(__name__)

# COCO class names — used as fallback when model config lacks id2label.
COCO_NAMES = [
    "person","bicycle","car","motorcycle","airplane","bus","train","truck",
    "boat","traffic light","fire hydrant","stop sign","parking meter","bench",
    "bird","cat","dog","horse","sheep","cow","elephant","bear","zebra",
    "giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee",
    "skis","snowboard","sports ball","kite","baseball bat","baseball glove",
    "skateboard","surfboard","tennis racket","bottle","wine glass","cup",
    "fork","knife","spoon","bowl","banana","apple","sandwich","orange",
    "broccoli","carrot","hot dog","pizza","donut","cake","chair","couch",
    "potted plant","bed","dining table","toilet","tv","laptop","mouse",
    "remote","keyboard","cell phone","microwave","oven","toaster","sink",
    "refrigerator","book","clock","vase","scissors","teddy bear",
    "hair drier","toothbrush",
]

_lock = threading.Lock()
_model = None
_processor = None
_device = None


def _resolve_device():
    pref = os.environ.get("DEIMV2_DEVICE", "cuda").lower()
    if pref == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _load_model():
    global _model, _processor, _device
    model_id = os.environ.get("DEIMV2_MODEL_ID", "Intellindust-AI-Lab/DEIMv2-S")
    _device = _resolve_device()
    log.info("[deimv2] loading %s on %s", model_id, _device)
    from transformers import AutoImageProcessor, AutoModelForObjectDetection
    _processor = AutoImageProcessor.from_pretrained(model_id)
    _model = AutoModelForObjectDetection.from_pretrained(model_id)
    _model.to(_device)
    _model.eval()
    log.info("[deimv2] ready")


def get_model():
    global _model, _processor, _device
    if _model is None:
        with _lock:
            if _model is None:
                _load_model()
    return _model, _processor, _device


def _get_label(model, cid: int) -> str:
    """Prefer model config id2label if available; fall back to COCO_NAMES."""
    id2label = getattr(getattr(model, "config", None), "id2label", None)
    if isinstance(id2label, dict):
        # keys may be int or str depending on how the config was loaded
        label = id2label.get(cid, id2label.get(str(cid)))
        if label:
            return str(label)
    if 0 <= cid < len(COCO_NAMES):
        return COCO_NAMES[cid]
    return f"class_{cid}"


def decode_image(image_b64: str) -> Image.Image:
    raw = base64.b64decode(image_b64)
    return Image.open(io.BytesIO(raw)).convert("RGB")


def run_inference(
    image_b64: str,
    conf_threshold: float = 0.35,
    img_size: int = 640,
    class_filter: Optional[List[int]] = None,
) -> InferResponse:
    model, processor, device = get_model()
    t0 = time.perf_counter()
    pil_img = decode_image(image_b64)
    orig_w, orig_h = pil_img.size
    scale = img_size / min(orig_w, orig_h)
    new_w, new_h = int(round(orig_w * scale)), int(round(orig_h * scale))
    pil_r = pil_img.resize((new_w, new_h), Image.BILINEAR)
    inputs = processor(images=pil_r, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    target_sizes = torch.tensor([[new_h, new_w]], device=device)
    results = processor.post_process_object_detection(
        outputs, threshold=conf_threshold, target_sizes=target_sizes,
    )[0]
    scores = results["scores"].cpu().numpy()
    labels = results["labels"].cpu().numpy()
    boxes = results["boxes"].cpu().numpy()
    ms = (time.perf_counter() - t0) * 1000.0
    entities = []
    for score, lid, box in zip(scores, labels, boxes):
        cid = int(lid)
        if class_filter is not None and cid not in class_filter:
            continue
        x1, y1, x2, y2 = box
        nx = max(0.0, min(1.0, float(x1) / new_w))
        ny = max(0.0, min(1.0, float(y1) / new_h))
        nw = max(0.0, min(1.0 - nx, float(x2 - x1) / new_w))
        nh = max(0.0, min(1.0 - ny, float(y2 - y1) / new_h))
        name = _get_label(model, cid)
        entities.append(Entity(
            label=name, class_id=cid, confidence=float(score),
            bbox=BBox(x=nx, y=ny, w=nw, h=nh),
        ))
    mid = os.environ.get("DEIMV2_MODEL_ID", "deimv2-s").split("/")[-1].lower()
    return InferResponse(
        entities=entities, inference_ms=round(ms, 2),
        model=mid, img_w=orig_w, img_h=orig_h,
    )
