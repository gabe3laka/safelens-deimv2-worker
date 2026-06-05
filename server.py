"""
server.py -- FastAPI/uvicorn live-server for the SafeLens DEIMv2 worker.

Architecture: long-running HTTP server (RunPod load-balancing endpoint mode).
Pattern: adapted from Kingo333/fluxrt-serverless.

Routes
------
GET  /health          -- returns immediately, no model required
GET  /ping            -- alias for /health
GET  /debug/startup   -- environment + torch diagnostics
POST /detect          -- run DEIMv2 inference (503 if model not ready)
POST /warmup          -- trigger background model load
"""

import asyncio
import logging
import os
import shutil
import sys
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("safelens-deimv2-worker")

# -- Config -------------------------------------------------------------------

PORT = int(os.getenv("PORT", "8000"))
WORKER_VERSION = "0.2.0-live-server"
SKIP_WARMUP = os.getenv("SKIP_WARMUP", "false").strip().lower() in ("1", "true", "yes", "on")
AUTO_WARMUP = os.getenv("AUTO_WARMUP", "true").strip().lower() in ("1", "true", "yes", "on")
WARMUP_TIMEOUT_S = int(os.getenv("WARMUP_TIMEOUT_S", "600"))
STARTUP_LOG_PATH = Path(os.getenv("STARTUP_LOG", "/tmp/safelens_startup.log"))

# -- State --------------------------------------------------------------------

_STATE_LOCK = threading.RLock()
_STATE: Dict[str, Any] = {
    "status": "cold",
    "error": None,
    "model": None,
    "processor": None,
    "device": None,
    "warmup_started_at": None,
    "ready_at": None,
}
_BOOT_TS = time.time()

# -- Startup log helper -------------------------------------------------------

def _log_startup(msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}] {msg}"
    print(line, flush=True)
    try:
        STARTUP_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with STARTUP_LOG_PATH.open("a") as fh:
            fh.write(line + "\n")
    except Exception:
        pass


# -- Public state helper ------------------------------------------------------

def _public_state() -> Dict[str, Any]:
    with _STATE_LOCK:
        return {
            "status": _STATE["status"],
            "model_loaded": _STATE["model"] is not None,
            "error": _STATE["error"],
            "warmup_started_at": _STATE["warmup_started_at"],
            "ready_at": _STATE["ready_at"],
            "uptime_seconds": round(time.time() - _BOOT_TS, 2),
        }


# -- Model warmup (background thread) -----------------------------------------

def _warmup_blocking() -> None:
    with _STATE_LOCK:
        if _STATE["status"] in ("loading", "ready"):
            return
        _STATE["status"] = "loading"
        _STATE["error"] = None
        _STATE["warmup_started_at"] = time.time()

    _log_startup("warmup: starting DEIMv2 model load")
    try:
        from deimv2_infer import get_model
        model, processor, device = get_model()
        with _STATE_LOCK:
            _STATE["model"] = model
            _STATE["processor"] = processor
            _STATE["device"] = device
            _STATE["status"] = "ready"
            _STATE["ready_at"] = time.time()
        _log_startup(f"warmup: ready -- model on {device}")
    except Exception as exc:
        err_msg = f"{type(exc).__name__}: {exc}"
        _log_startup(f"warmup: FAILED -- {err_msg}")
        with _STATE_LOCK:
            _STATE["status"] = "error"
            _STATE["error"] = err_msg


def _start_warmup_background() -> None:
    with _STATE_LOCK:
        if _STATE["status"] in ("loading", "ready"):
            return
    t = threading.Thread(target=_warmup_blocking, name="deimv2-warmup", daemon=True)
    t.start()


# -- FastAPI lifespan ---------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    _log_startup(f"server starting -- port={PORT} version={WORKER_VERSION}")
    _log_startup(f"skip_warmup={SKIP_WARMUP} auto_warmup={AUTO_WARMUP}")
    if SKIP_WARMUP:
        _log_startup("SKIP_WARMUP=true: skipping model load (diagnostic mode)")
    elif AUTO_WARMUP:
        _log_startup("AUTO_WARMUP=true: triggering background model load")
        _start_warmup_background()
    else:
        _log_startup("AUTO_WARMUP=false: model load deferred until POST /warmup")
    yield
    _log_startup("server shutting down")


app = FastAPI(title="safelens-deimv2-worker", version=WORKER_VERSION, lifespan=lifespan)


# -- Health / ping (no model dependency) -------------------------------------

@app.get("/health")
async def health():
    """Returns immediately without loading DEIMv2."""
    return JSONResponse({
        "ok": True,
        "worker": "safelens-deimv2-worker",
        "mode": "live-server",
        "version": WORKER_VERSION,
        **_public_state(),
    })


@app.get("/ping")
async def ping():
    """Alias for /health."""
    return JSONResponse({
        "ok": True,
        "worker": "safelens-deimv2-worker",
        "mode": "live-server",
        "version": WORKER_VERSION,
        **_public_state(),
    })


# -- Debug / startup diagnostics ----------------------------------------------

@app.get("/debug/startup")
async def debug_startup(deep: bool = Query(False)):
    """Environment info, disk usage, and optional torch/CUDA diagnostics."""
    total, used, free = shutil.disk_usage("/")
    info: Dict[str, Any] = {
        "ok": True,
        "worker": "safelens-deimv2-worker",
        "version": WORKER_VERSION,
        "mode": "live-server",
        "port": PORT,
        "cwd": os.getcwd(),
        "python_version": sys.version,
        "skip_warmup": SKIP_WARMUP,
        "auto_warmup": AUTO_WARMUP,
        "startup_log": str(STARTUP_LOG_PATH),
        "env_names": sorted(os.environ.keys()),
        "disk": {
            "total_gb": round(total / 1e9, 1),
            "used_gb": round(used / 1e9, 1),
            "free_gb": round(free / 1e9, 1),
        },
        **_public_state(),
    }
    try:
        lines = STARTUP_LOG_PATH.read_text().splitlines()
        info["startup_log_tail"] = lines[-50:]
    except Exception:
        info["startup_log_tail"] = []

    if deep:
        try:
            import torch
            info["torch"] = {
                "version": torch.__version__,
                "cuda_version": torch.version.cuda,
                "cuda_available": bool(torch.cuda.is_available()),
                "device_count": torch.cuda.device_count(),
            }
            if torch.cuda.is_available():
                info["torch"]["gpu_name"] = torch.cuda.get_device_name(0)
                free_mem, _ = torch.cuda.mem_get_info(0)
                info["torch"]["gpu_mem_free_gb"] = round(free_mem / 1e9, 2)
        except Exception as exc:
            info["torch_error"] = f"{type(exc).__name__}: {exc}"
        try:
            import transformers
            info["transformers_version"] = transformers.__version__
        except Exception as exc:
            info["transformers_error"] = f"{type(exc).__name__}: {exc}"

    return JSONResponse(info)


# -- Warmup trigger -----------------------------------------------------------

@app.post("/warmup")
async def warmup(wait: bool = Query(False)):
    """Trigger background model load. Use ?wait=true to block until ready."""
    _start_warmup_background()
    if wait:
        for _ in range(WARMUP_TIMEOUT_S):
            state = _public_state()
            if state["status"] in ("ready", "error"):
                return JSONResponse({"ok": state["status"] == "ready", **state})
            await asyncio.sleep(1)
    return JSONResponse({"ok": True, "triggered": True, **_public_state()})


# -- Detect endpoint ----------------------------------------------------------

@app.post("/detect")
async def detect(payload: Dict[str, Any]):
    """
    Run DEIMv2 object detection.

    Request body:
        {"image_b64": "<base64>", "conf": 0.35, "img_size": 640, "classes": null}

    Returns 503 if the model is not ready yet.
    """
    state = _public_state()
    if state["status"] != "ready":
        return JSONResponse(
            {"error": "model_not_ready", "status": state["status"], "entities": []},
            status_code=503,
        )

    import binascii
    import base64 as _b64

    image_b64 = payload.get("image_b64")
    if not image_b64:
        return JSONResponse({"error": "missing_image_b64", "entities": []}, status_code=400)
    try:
        _b64.b64decode(image_b64, validate=True)
    except (binascii.Error, ValueError):
        return JSONResponse({"error": "invalid_base64", "entities": []}, status_code=400)

    try:
        from deimv2_infer import run_inference
        conf = float(os.environ.get("DEIMV2_CONF", payload.get("conf", 0.35)))
        img_size = int(os.environ.get("DEIMV2_IMG_SIZE", payload.get("img_size", 640)))
        class_filter = payload.get("classes")
        resp = run_inference(
            image_b64=image_b64,
            conf_threshold=conf,
            img_size=img_size,
            class_filter=class_filter,
        )
        return JSONResponse(resp.model_dump())
    except Exception as exc:
        log.exception("detect failed: %s", exc)
        return JSONResponse(
            {"error": f"inference_failed: {type(exc).__name__}: {exc}", "entities": []},
            status_code=500,
        )


# -- Entrypoint ---------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level=os.getenv("UVICORN_LOG_LEVEL", "info"))
