"""
deimv2_infer.py -- DEIMv2 inference wrapper for SafeLens RunPod worker.

Loads the OFFICIAL DEIMv2 model (PyTorchModelHubMixin custom class from the
upstream /opt/DEIMv2 engine package) -- NOT transformers Auto classes. DEIMv2
checkpoints on the Hub contain only config.json + model.safetensors and have
no preprocessor_config.json, so AutoImageProcessor.from_pretrained does not
apply. See official_deimv2_loader.py for the loader/inference details.

Environment variables:
  DEIMV2_MODEL_ID   HuggingFace model id  Default: Intellindust/DEIMv2_DINOv3_S_COCO
  DEIMV2_DEVICE     cuda | cpu            Default: cuda (auto-falls-back to cpu)
  DEIMV2_CONF       Confidence threshold  Default: 0.35
  DEIMV2_IMG_SIZE   Square resize size    Default: 640
  DEIMV2_BACKEND    official-deimv2-hf | transformers-fallback
                    Default: official-deimv2-hf. The fallback backend loads a
                    standard transformers detector (facebook/detr-resnet-50)
                    purely to validate the Eagle Vision 2 teal-box pipeline.
                    It is clearly labelled as a fallback, NOT DEIMv2.
  HF_TOKEN          Optional HF token (private/gated mirrors only). Never logged.
"""

from __future__ import annotations
import base64, io, logging, os, threading, time, traceback
from typing import List, Optional

import torch
from PIL import Image
from schema import BBox, Entity, InferResponse

log = logging.getLogger(__name__)

DEFAULT_MODEL_ID = "Intellindust/DEIMv2_DINOv3_S_COCO"

# COCO class names -- fallback labelling for the transformers fallback backend.
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
_processor = None      # transformers fallback only; None for official DEIMv2
_device = None
_backend = None        # "official-deimv2-hf" | "transformers-fallback"


def _backend_name() -> str:
    return os.environ.get("DEIMV2_BACKEND", "official-deimv2-hf").strip().lower()


def _resolve_device():
    pref = os.environ.get("DEIMV2_DEVICE", "cuda").lower()
    if pref == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _load_official():
    """Load the official DEIMv2 model via the PyTorchModelHubMixin loader."""
    global _model, _processor, _device, _backend
    from official_deimv2_loader import load_official_deimv2
    model_id = os.environ.get("DEIMV2_MODEL_ID", DEFAULT_MODEL_ID)
    _device = _resolve_device()
    log.info("[deimv2] loading OFFICIAL %s on %s", model_id, _device)
    model, device, _cls = load_official_deimv2(model_id=model_id, device=_device)
    _model = model
    _processor = None
    _device = device
    _backend = "official-deimv2-hf"
    log.info("[deimv2] official ready (%s)", _cls)


def _load_fallback():
    """Load a standard transformers detector (facebook/detr-resnet-50).

    FALLBACK ONLY -- used to validate the Eagle Vision 2 teal-box pipeline when
    the official DEIMv2 loader needs more work. Clearly labelled, not DEIMv2.
    """
    global _model, _processor, _device, _backend
    from transformers import AutoImageProcessor, AutoModelForObjectDetection
    fb_id = os.environ.get("DEIMV2_FALLBACK_MODEL_ID", "facebook/detr-resnet-50")
    _device = _resolve_device()
    log.info("[deimv2] loading FALLBACK %s on %s", fb_id, _device)
    _processor = AutoImageProcessor.from_pretrained(fb_id)
    _model = AutoModelForObjectDetection.from_pretrained(fb_id)
    _model.to(_device)
    _model.eval()
    _backend = "transformers-fallback"
    log.info("[deimv2] fallback ready")


def _load_model():
    """Load the model for the configured backend. Raises on failure so the
    caller (warmup / /debug/model-load) records the full traceback."""
    backend = _backend_name()
    try:
        if backend == "transformers-fallback":
            _load_fallback()
        else:
            _load_official()
    except Exception as exc:
        tb = traceback.format_exc()
        log.error("[deimv2] model load FAILED (%s): %s: %s\n%s",
                  backend, type(exc).__name__, exc, tb)
        raise


def get_model():
    global _model
    if _model is None:
        with _lock:
            if _model is None:
                _load_model()
    return _model, _processor, _device


def decode_image(image_b64: str) -> Image.Image:
    raw = base64.b64decode(image_b64)
    return Image.open(io.BytesIO(raw)).convert("RGB")


def _model_name() -> str:
    mid = os.environ.get("DEIMV2_MODEL_ID", DEFAULT_MODEL_ID).split("/")[-1]
    if _backend == "transformers-fallback":
        fb = os.environ.get('DEIMV2_FALLBACK_MODEL_ID', 'detr-resnet-50').split('/')[-1]
        return "FALLBACK:" + fb
    return mid


def _run_official(pil_img, conf_threshold, img_size, class_filter):
    from official_deimv2_loader import run_official_inference
    dets = run_official_inference(
        _model, _device, pil_img,
        conf_threshold=conf_threshold, img_size=img_size, class_filter=class_filter,
    )
    entities = []
    for d in dets:
        b = d["bbox"]
        entities.append(Entity(
            label=d["label"], class_id=d["class_id"], confidence=d["confidence"],
            bbox=BBox(x=b["x"], y=b["y"], w=b["w"], h=b["h"]),
        ))
    return entities


def _run_fallback(pil_img, conf_threshold, img_size, class_filter):
    """transformers DETR inference path (fallback backend only)."""
    orig_w, orig_h = pil_img.size
    inputs = _processor(images=pil_img, return_tensors="pt")
    inputs = {k: v.to(_device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = _model(**inputs)
    target_sizes = torch.tensor([[orig_h, orig_w]], device=_device)
    results = _processor.post_process_object_detection(
        outputs, threshold=conf_threshold, target_sizes=target_sizes,
    )[0]
    scores = results["scores"].cpu().numpy()
    labels = results["labels"].cpu().numpy()
    boxes = results["boxes"].cpu().numpy()
    entities = []
    for score, lid, box in zip(scores, labels, boxes):
        cid = int(lid)
        if class_filter is not None and cid not in class_filter:
            continue
        x1, y1, x2, y2 = box
        nx = max(0.0, min(1.0, float(x1) / orig_w))
        ny = max(0.0, min(1.0, float(y1) / orig_h))
        nw = max(0.0, min(1.0 - nx, float(x2 - x1) / orig_w))
        nh = max(0.0, min(1.0 - ny, float(y2 - y1) / orig_h))
        id2label = getattr(getattr(_model, "config", None), "id2label", None)
        name = None
        if isinstance(id2label, dict):
            name = id2label.get(cid, id2label.get(str(cid)))
        if not name:
            name = COCO_NAMES[cid] if 0 <= cid < len(COCO_NAMES) else "class_" + str(cid)
        entities.append(Entity(
            label=str(name), class_id=cid, confidence=float(score),
            bbox=BBox(x=nx, y=ny, w=nw, h=nh),
        ))
    return entities


def run_inference(
    image_b64: str,
    conf_threshold: float = 0.35,
    img_size: int = 640,
    class_filter: Optional[List[int]] = None,
) -> InferResponse:
    get_model()  # ensure loaded; sets _backend
    t0 = time.perf_counter()
    pil_img = decode_image(image_b64)
    orig_w, orig_h = pil_img.size
    if _backend == "transformers-fallback":
        entities = _run_fallback(pil_img, conf_threshold, img_size, class_filter)
    else:
        entities = _run_official(pil_img, conf_threshold, img_size, class_filter)
    ms = (time.perf_counter() - t0) * 1000.0
    return InferResponse(
        entities=entities, inference_ms=round(ms, 2),
        model=_model_name(), img_w=orig_w, img_h=orig_h,
    )
