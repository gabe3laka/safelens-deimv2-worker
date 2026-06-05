"""
official_deimv2_loader.py -- Official DEIMv2 model loader for SafeLens worker.

DEIMv2 is NOT a standard Transformers AutoImageProcessor / AutoModelForObject-
Detection model. The official checkpoints on the Hugging Face Hub
(e.g. Intellindust/DEIMv2_DINOv3_S_COCO) are pushed via
huggingface_hub.PyTorchModelHubMixin and contain only config.json +
model.safetensors -- there is NO preprocessor_config.json and NO custom HF
modeling code on the Hub. The architecture lives in the upstream GitHub repo
(https://github.com/Intellindust-AI-Lab/DEIMv2), cloned into /opt/DEIMv2 and
placed on PYTHONPATH by the Dockerfile.

This module reproduces the official loading + inference pattern documented in
the upstream hf_models.ipynb notebook:

    from engine.backbone import HGNetv2, DINOv3STAs
    from engine.deim import HybridEncoder, LiteEncoder
    from engine.deim import DFINETransformer, DEIMTransformer
    from engine.deim.postprocessor import PostProcessor

    class DEIMv2(nn.Module, PyTorchModelHubMixin):
        ...
        def forward(self, x, orig_target_sizes):
            ...

    model = DEIMv2.from_pretrained("Intellindust/DEIMv2_DINOv3_S_COCO")

Preprocessing (per the notebook) is a plain Resize(IMAGE_SIZE) + ToTensor()
with NO normalization, and the forward call takes orig_target_sizes.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

log = logging.getLogger(__name__)

# Official DEIMv2-S HF checkpoint (DINOv3 ViT-Tiny backbone, COCO).
DEFAULT_MODEL_ID = "Intellindust/DEIMv2_DINOv3_S_COCO"

# COCO 80-class id->name map. The DEIMv2 PostProcessor emits 0-indexed class
# ids; the official notebook maps them with (id + 1) into this 1-indexed table.
COCO_LABEL_MAP = {
    1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane',
    6: 'bus', 7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light',
    11: 'fire hydrant', 12: 'stop sign', 13: 'parking meter', 14: 'bench',
    15: 'bird', 16: 'cat', 17: 'dog', 18: 'horse', 19: 'sheep',
    20: 'cow', 21: 'elephant', 22: 'bear', 23: 'zebra', 24: 'giraffe',
    25: 'backpack', 26: 'umbrella', 27: 'handbag', 28: 'tie', 29: 'suitcase',
    30: 'frisbee', 31: 'skis', 32: 'snowboard', 33: 'sports ball', 34: 'kite',
    35: 'baseball bat', 36: 'baseball glove', 37: 'skateboard', 38: 'surfboard',
    39: 'tennis racket', 40: 'bottle', 41: 'wine glass', 42: 'cup', 43: 'fork',
    44: 'knife', 45: 'spoon', 46: 'bowl', 47: 'banana', 48: 'apple',
    49: 'sandwich', 50: 'orange', 51: 'broccoli', 52: 'carrot', 53: 'hot dog',
    54: 'pizza', 55: 'donut', 56: 'cake', 57: 'chair', 58: 'couch',
    59: 'potted plant', 60: 'bed', 61: 'dining table', 62: 'toilet', 63: 'tv',
    64: 'laptop', 65: 'mouse', 66: 'remote', 67: 'keyboard', 68: 'cell phone',
    69: 'microwave', 70: 'oven', 71: 'toaster', 72: 'sink', 73: 'refrigerator',
    74: 'book', 75: 'clock', 76: 'vase', 77: 'scissors', 78: 'teddy bear',
    79: 'hair drier', 80: 'toothbrush',
}


def coco_label(class_id_0_indexed: int) -> str:
    """Map a 0-indexed DEIMv2 class id to a COCO name (notebook uses id + 1)."""
    name = COCO_LABEL_MAP.get(int(class_id_0_indexed) + 1)
    return name if name else f"class_{int(class_id_0_indexed)}"


def build_deimv2_class():
    """Import the upstream engine modules and build the official DEIMv2 class.

    Imports are done lazily so that merely importing this module (e.g. during
    diagnostics or unit tests) does not require the heavy /opt/DEIMv2 engine
    package to be present.
    """
    from huggingface_hub import PyTorchModelHubMixin
    from engine.backbone import HGNetv2, DINOv3STAs
    from engine.deim import HybridEncoder, LiteEncoder
    from engine.deim import DFINETransformer, DEIMTransformer
    from engine.deim.postprocessor import PostProcessor

    class DEIMv2(nn.Module, PyTorchModelHubMixin):
        """Mirrors the official DEIMv2 wrapper from hf_models.ipynb."""

        def __init__(self, config):
            super().__init__()
            if 'DINOv3STAs' in config:
                self.backbone = DINOv3STAs(**config["DINOv3STAs"])
            else:
                self.backbone = HGNetv2(**config["HGNetv2"])
            if 'LiteEncoder' in config:
                self.encoder = LiteEncoder(**config["LiteEncoder"])
            else:
                self.encoder = HybridEncoder(**config["HybridEncoder"])
            if 'DEIMTransformer' in config:
                self.decoder = DEIMTransformer(**config["DEIMTransformer"])
            else:
                self.decoder = DFINETransformer(**config["DFINETransformer"])
            self.postprocessor = PostProcessor(**config["PostProcessor"])

        def forward(self, x, orig_target_sizes):
            x = self.backbone(x)
            x = self.encoder(x)
            x = self.decoder(x)
            x = self.postprocessor(x, orig_target_sizes)
            return x

    return DEIMv2


def resolve_device(pref: Optional[str] = None) -> torch.device:
    pref = (pref or os.environ.get("DEIMV2_DEVICE", "cuda")).lower()
    if pref == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _hf_token() -> Optional[str]:
    """Optional HF token. DEIMv2 public repos do not require it; supported only
    for private/gated mirrors. Never logged, never returned in diagnostics."""
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")


def load_official_deimv2(
    model_id: Optional[str] = None,
    device: Optional[torch.device] = None,
) -> Tuple[nn.Module, torch.device, str]:
    """Load the official DEIMv2 model via PyTorchModelHubMixin.from_pretrained.

    Returns (model, device, model_class_name). Raises on failure so the caller
    can capture the full structured traceback.
    """
    model_id = model_id or os.environ.get("DEIMV2_MODEL_ID", DEFAULT_MODEL_ID)
    device = device or resolve_device()
    log.info("[deimv2] official loader: %s on %s", model_id, device)

    DEIMv2 = build_deimv2_class()

    token = _hf_token()
    from_pretrained_kwargs: Dict[str, Any] = {}
    if token:
        # PyTorchModelHubMixin forwards **model_kwargs/hub kwargs to hf_hub
        # download; "token" is the supported keyword. Do not log it.
        from_pretrained_kwargs["token"] = token

    try:
        model = DEIMv2.from_pretrained(model_id, **from_pretrained_kwargs)
    except TypeError:
        # Older mixin signatures may not accept token kwarg; retry without it.
        model = DEIMv2.from_pretrained(model_id)

    model.to(device)
    model.eval()
    log.info("[deimv2] official loader ready: %s", type(model).__name__)
    return model, device, type(model).__name__


def preprocess(pil_img: "Image.Image", img_size: int = 640) -> torch.Tensor:
    """Official preprocessing: Resize((S, S)) + ToTensor(), no normalization."""
    tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])
    return tf(pil_img).unsqueeze(0)


def run_official_inference(
    model: nn.Module,
    device: torch.device,
    pil_img: "Image.Image",
    conf_threshold: float = 0.35,
    img_size: int = 640,
    class_filter: Optional[List[int]] = None,
) -> List[Dict[str, Any]]:
    """Run DEIMv2 forward + postprocess, returning normalized detections.

    Each detection: {label, class_id (0-indexed), confidence, bbox{x,y,w,h}}
    with bbox normalized to 0..1 (per the official notebook: coords / IMAGE_SIZE).
    """
    input_tensor = preprocess(pil_img, img_size).to(device)
    orig_target_sizes = torch.tensor([[img_size, img_size]], device=device)

    with torch.no_grad():
        outputs = model(input_tensor, orig_target_sizes=orig_target_sizes)

    out0 = outputs[0]
    labels = out0["labels"].detach().cpu()
    boxes = out0["boxes"].detach().cpu()
    scores = out0["scores"].detach().cpu()

    detections: List[Dict[str, Any]] = []
    for label, box, score in zip(labels, boxes, scores):
        s = float(score.item())
        if s < conf_threshold:
            continue
        cid = int(label.item())  # 0-indexed model class id
        if class_filter is not None and cid not in class_filter:
            continue
        x1, y1, x2, y2 = [float(v) for v in box.tolist()]
        nx = max(0.0, min(1.0, x1 / img_size))
        ny = max(0.0, min(1.0, y1 / img_size))
        nw = max(0.0, min(1.0 - nx, (x2 - x1) / img_size))
        nh = max(0.0, min(1.0 - ny, (y2 - y1) / img_size))
        detections.append({
            "label": coco_label(cid),
            "class_id": cid,
            "confidence": s,
            "bbox": {"x": nx, "y": ny, "w": nw, "h": nh},
        })
    return detections
