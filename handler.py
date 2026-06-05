"""
handler.py -- RunPod serverless handler for the DEIMv2 SafeLens worker.

Request shape (matches schema.InferRequest):
  { "input": { "image_b64": "<base64>", "conf": 0.35, "img_size": 640, "classes": null } }

Response shape (matches schema.InferResponse):
  { "entities": [...], "inference_ms": 45.2, "model": "deimv2-s", "img_w": 640, "img_h": 480 }

Structured error shapes (always returned, never crash the handler):
  { "error": "missing_image_b64", "entities": [] }
  { "error": "invalid_base64", "entities": [] }
  { "error": "model_load_failed: <msg>", "entities": [] }
  { "error": "<inference error>", "entities": [] }
"""

import binascii
import logging
import os
import runpod

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
log = logging.getLogger(__name__)


def handler(event: dict) -> dict:
    inp = event.get("input", {})

    # 1. Validate image_b64 presence
    image_b64 = inp.get("image_b64")
    if not image_b64:
        log.warning("[handler] missing image_b64")
        return {"error": "missing_image_b64", "entities": []}

    # 2. Validate base64 encoding
    try:
        import base64 as _b64
        _b64.b64decode(image_b64, validate=True)
    except (binascii.Error, ValueError):
        log.warning("[handler] invalid base64")
        return {"error": "invalid_base64", "entities": []}

    # 3. Load model (lazy, cached)
    try:
        from deimv2_infer import get_model
        model, processor, device = get_model()
    except Exception as exc:
        log.exception("[handler] model load failed: %s", exc)
        return {"error": f"model_load_failed: {exc}", "entities": []}

    # 4. Run inference
    try:
        from deimv2_infer import run_inference
        conf = float(os.environ.get("DEIMV2_CONF", inp.get("conf", 0.35)))
        img_size = int(os.environ.get("DEIMV2_IMG_SIZE", inp.get("img_size", 640)))
        class_filter = inp.get("classes")
        resp = run_inference(
            image_b64=image_b64,
            conf_threshold=conf,
            img_size=img_size,
            class_filter=class_filter,
        )
        return resp.model_dump()
    except Exception as exc:
        log.exception("[handler] inference error: %s", exc)
        return {"error": str(exc), "entities": []}


def _warmup():
    from deimv2_infer import get_model
    log.info("[warmup] loading model...")
    get_model()
    log.info("[warmup] done")


if __name__ == "__main__":
    _warmup()
    runpod.serverless.start({"handler": handler})
