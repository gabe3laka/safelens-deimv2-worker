"""
risk/grounding_dino_scanner.py -- REAL open-vocabulary GroundingDINO adapter.

GroundingDINO is NOT the reasoning brain and NOT a safety authority. It is an
open-vocabulary grounding/scanner: given an image + a text prompt of classes it
returns candidate boxes/phrases, used for unknown/rare-object discovery and
dataset-candidate creation -- never to approve an HSE risk.

Disabled by default (OPEN_VOCAB_SCANNER_ENABLED=false). Real but lazy: torch/
transformers import only on first use; weights resolve at runtime (HF cache or
GROUNDING_DINO_WEIGHTS_PATH), never baked at Docker build. Every result is
candidate_only + requires_human_review (enforced by reason_schema).
"""

from __future__ import annotations

import base64
import io
import logging
import os
import threading
import time
from typing import Any, Dict, List, Optional

from .reason_schema import OpenVocabCandidate, OpenVocabResult

log = logging.getLogger("safelens-vision-worker.openvocab")

_DEFAULT_PROMPT = (
    "open hole . manhole . spill . broken glass . fire extinguisher . gas cylinder . "
    "cable . forklift . ladder . scaffold . person . hard hat . safety vest . harness . "
    "lanyard . hook . barricade . warning sign .")

_STATE: Dict[str, Any] = {}  # lazy model/processor cache
_LOCK = threading.RLock()


def enabled() -> bool:
    return os.getenv("OPEN_VOCAB_SCANNER_ENABLED", "false").strip().lower() in (
        "1", "true", "yes", "on")


def _model_id() -> str:
    # HF transformers GroundingDINO (e.g. IDEA-Research/grounding-dino-tiny).
    return os.getenv("GROUNDING_DINO_MODEL_ID", "IDEA-Research/grounding-dino-tiny")


def default_prompt() -> str:
    return os.getenv("OPEN_VOCAB_SCAN_PROMPT", _DEFAULT_PROMPT)


def _box_threshold() -> float:
    try:
        return float(os.getenv("GROUNDING_DINO_BOX_THRESHOLD", "0.35"))
    except (TypeError, ValueError):
        return 0.35


def _text_threshold() -> float:
    try:
        return float(os.getenv("GROUNDING_DINO_TEXT_THRESHOLD", "0.25"))
    except (TypeError, ValueError):
        return 0.25


def available() -> bool:
    """Best-effort: deps importable (model loads lazily on first scan)."""
    if not enabled():
        return False
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
        return True
    except Exception:  # noqa: BLE001
        return False


def _ensure_loaded():
    with _LOCK:
        if _STATE.get("model") is not None:
            return _STATE["model"], _STATE["processor"]
        import torch  # noqa: F401
        from transformers import (AutoModelForZeroShotObjectDetection,
                                   AutoProcessor)
        cache_dir = os.getenv("GROUNDING_DINO_CACHE_DIR",
                              "/runpod-volume/models/groundingdino")
        device = os.getenv("OPEN_VOCAB_DEVICE", os.getenv("REASONER_DEVICE", "cuda"))
        mid = _model_id()
        processor = AutoProcessor.from_pretrained(mid, cache_dir=cache_dir)
        model = AutoModelForZeroShotObjectDetection.from_pretrained(mid, cache_dir=cache_dir)
        try:
            model = model.to(device)
        except Exception:  # noqa: BLE001
            device = "cpu"
        model.eval()
        _STATE.update(model=model, processor=processor, device=device)
        return model, processor


def scan(frame_b64: Optional[str], *, prompt: Optional[str] = None,
         session_id: Optional[str] = None, frame_id: Optional[str] = None) -> Dict[str, Any]:
    """Run an open-vocab scan; ALWAYS returns a candidate-only result dict.

    Never raises into the caller. Disabled / missing-deps / missing-image all
    degrade to a clear status with no candidates.
    """
    t0 = time.perf_counter()
    text = prompt or default_prompt()
    res = OpenVocabResult(prompt=text, source_model="GroundingDINO",
                          session_id=session_id, frame_id=frame_id)
    if not enabled():
        res.status = "disabled"
        return res.enforce_candidate_contract().model_dump()
    if not frame_b64:
        res.status = "error"
        res.error = "missing_frame_b64"
        return res.enforce_candidate_contract().model_dump()
    try:
        from PIL import Image
        img = Image.open(io.BytesIO(base64.b64decode(frame_b64))).convert("RGB")
    except Exception as exc:  # noqa: BLE001
        res.status = "error"
        res.error = f"decode: {exc}"
        return res.enforce_candidate_contract().model_dump()
    try:
        import torch
        model, processor = _ensure_loaded()
        device = _STATE.get("device", "cpu")
        inputs = processor(images=img, text=text, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        w, h = img.size
        results = processor.post_process_grounded_object_detection(
            outputs, inputs.get("input_ids"),
            box_threshold=_box_threshold(), text_threshold=_text_threshold(),
            target_sizes=[(h, w)])
        r0 = results[0] if results else {"boxes": [], "scores": [], "labels": []}
        cands: List[OpenVocabCandidate] = []
        boxes = r0.get("boxes", [])
        scores = r0.get("scores", [])
        labels = r0.get("labels", r0.get("text_labels", []))
        for i in range(len(boxes)):
            x0, y0, x1, y1 = [float(v) for v in (boxes[i].tolist() if hasattr(boxes[i], "tolist") else boxes[i])]
            cands.append(OpenVocabCandidate(
                label=str(labels[i]) if i < len(labels) else "object",
                phrase=str(labels[i]) if i < len(labels) else None,
                confidence=float(scores[i]) if i < len(scores) else 0.0,
                bbox={"x": max(0.0, x0 / w), "y": max(0.0, y0 / h),
                      "w": max(0.0, (x1 - x0) / w), "h": max(0.0, (y1 - y0) / h)}))
        res.candidates = cands
        res.status = "ok"
    except Exception as exc:  # noqa: BLE001
        res.status = "unavailable"
        res.error = f"{type(exc).__name__}: {exc}"
    res.latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
    return res.enforce_candidate_contract().model_dump()


def reset() -> None:
    with _LOCK:
        _STATE.clear()
