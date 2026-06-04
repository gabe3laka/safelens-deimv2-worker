"""
handler.py -- RunPod serverless handler for the DEIMv2 SafeLens worker.

Request shape (matches schema.InferRequest):
  { "input": { "image_b64": "<base64>", "conf": 0.35, "img_size": 640, "classes": null } }

Response shape (matches schema.InferResponse):
  { "entities": [...], "inference_ms": 45.2, "model": "deimv2-s", "img_w": 640, "img_h": 480 }
"""

import logging
import os
import runpod
from deimv2_infer import run_inference
from schema import InferRequest

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger(__name__)


def handler(event: dict) -> dict:
    try:
        raw = event.get("input", {})
        req = InferRequest(**raw)
        conf = float(os.environ.get("DEIMV2_CONF", req.conf))
        img_size = int(os.environ.get("DEIMV2_IMG_SIZE", req.img_size))
        resp = run_inference(
            image_b64=req.image_b64,
            conf_threshold=conf,
            img_size=img_size,
            class_filter=req.classes,
        )
        return resp.model_dump()
    except Exception as exc:
        log.exception("[handler] error: %s", exc)
        return {"error": str(exc)}


def _warmup():
    from deimv2_infer import get_model
    log.info("[warmup] loading model...")
    get_model()
    log.info("[warmup] done")


if __name__ == "__main__":
    _warmup()
    runpod.serverless.start({"handler": handler})
