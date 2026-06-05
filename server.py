"""
server.py -- FastAPI/uvicorn live-server for the SafeLens DEIMv2 worker.

Architecture: long-running HTTP server (RunPod load-balancing endpoint mode).
Pattern: adapted from Kingo333/fluxrt-serverless.

Routes
------
GET  /health            -- returns immediately, no model required
GET  /ping              -- alias for /health
GET  /debug/startup     -- environment + torch diagnostics (?deep=true for imports)
POST /debug/model-load  -- attempt model load only, return structured result
POST /detect            -- run DEIMv2 inference (503 if model not ready)
POST /warmup            -- trigger background model load
"""

import asyncio
import logging
import os
import shutil
import sys
import threading
import time
import traceback
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
WORKER_VERSION = "0.3.0-live-server"
DEFAULT_MODEL_ID = "Intellindust/DEIMv2_DINOv3_S_COCO"
SKIP_WARMUP = os.getenv("SKIP_WARMUP", "false").strip().lower() in ("1", "true", "yes", "on")
AUTO_WARMUP = os.getenv("AUTO_WARMUP", "true").strip().lower() in ("1", "true", "yes", "on")
WARMUP_TIMEOUT_S = int(os.getenv("WARMUP_TIMEOUT_S", "600"))
STARTUP_LOG_PATH = Path(os.getenv("STARTUP_LOG", "/tmp/safelens_startup.log"))

# -- State --------------------------------------------------------------------

_STATE_LOCK = threading.RLock()
_STATE: Dict[str, Any] = {
    "status": "cold",
    "error": None,
    "error_traceback": None,
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
            "error_traceback": _STATE["error_traceback"],
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
        _STATE["error_traceback"] = None
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
        tb = traceback.format_exc()
        _log_startup(f"warmup: FAILED -- {err_msg}")
        _log_startup(tb)
        with _STATE_LOCK:
            _STATE["status"] = "error"
            _STATE["error"] = err_msg
            _STATE["error_traceback"] = tb

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

# -- Import diagnostics helper (no secrets) -----------------------------------

def _safe_version(module_name: str) -> Optional[str]:
    try:
        mod = __import__(module_name)
        return getattr(mod, "__version__", "unknown")
    except Exception as exc:
        return f"ERROR: {type(exc).__name__}: {exc}"

def _import_check(callable_test) -> Dict[str, Any]:
    """Run a single import test and return ok / structured error (no secrets)."""
    try:
        callable_test()
        return {"status": "ok"}
    except Exception as exc:
        return {
            "status": "error",
            "exception_type": type(exc).__name__,
            "exception_message": str(exc),
            "traceback": traceback.format_exc()[-3000:],
        }

def _collect_import_diagnostics() -> Dict[str, Any]:
    """Deep import diagnostics. Never exposes secret values, only env names."""
    versions: Dict[str, Any] = {}
    for name in ("transformers", "huggingface_hub", "torch", "torchvision", "PIL",
                 "timm", "safetensors", "tokenizers", "accelerate", "numpy"):
        versions[name] = _safe_version(name)

    cuda: Dict[str, Any] = {}
    try:
        import torch
        cuda["available"] = bool(torch.cuda.is_available())
        cuda["device_count"] = torch.cuda.device_count()
        if torch.cuda.is_available():
            cuda["gpu_name"] = torch.cuda.get_device_name(0)
    except Exception as exc:
        cuda["error"] = f"{type(exc).__name__}: {exc}"

    # Dependency imports DEIMv2 needs (timm/safetensors/torchvision are required).
    def _imp_timm():
        import timm  # noqa: F401
    def _imp_safetensors():
        import safetensors  # noqa: F401
    def _imp_torchvision():
        import torchvision  # noqa: F401
    def _imp_hub_mixin():
        from huggingface_hub import PyTorchModelHubMixin  # noqa: F401

    # Official DEIMv2 engine imports (from /opt/DEIMv2 on PYTHONPATH).
    def _imp_engine_backbone():
        from engine.backbone import HGNetv2, DINOv3STAs  # noqa: F401
    def _imp_engine_deim():
        from engine.deim import HybridEncoder, DEIMTransformer  # noqa: F401
    def _imp_engine_postproc():
        from engine.deim.postprocessor import PostProcessor  # noqa: F401
    def _imp_official_loader():
        import official_deimv2_loader  # noqa: F401

    # AutoImageProcessor is OPTIONAL now -- only used by the transformers-fallback
    # backend, NOT by the official DEIMv2 loader.
    def _imp_auto_image_processor():
        from transformers import AutoImageProcessor  # noqa: F401
    def _imp_auto_model_obj_det():
        from transformers import AutoModelForObjectDetection  # noqa: F401

    required = {
        "timm": _import_check(_imp_timm),
        "safetensors": _import_check(_imp_safetensors),
        "torchvision": _import_check(_imp_torchvision),
        "PyTorchModelHubMixin": _import_check(_imp_hub_mixin),
    }
    deimv2_official = {
        "engine.backbone": _import_check(_imp_engine_backbone),
        "engine.deim": _import_check(_imp_engine_deim),
        "engine.deim.postprocessor": _import_check(_imp_engine_postproc),
        "official_deimv2_loader": _import_check(_imp_official_loader),
    }
    optional = {
        "AutoImageProcessor": _import_check(_imp_auto_image_processor),
        "AutoModelForObjectDetection": _import_check(_imp_auto_model_obj_det),
    }

    def _summary(d):
        return {k: ("ok" if v.get("status") == "ok" else "error") for k, v in d.items()}

    imports_summary = {}
    imports_summary.update(_summary(required))
    imports_summary.update(_summary(deimv2_official))
    imports_summary.update(_summary(optional))

    return {
        "versions": versions,
        "cuda": cuda,
        "imports": imports_summary,
        "imports_detail": {
            "required": required,
            "deimv2_official": deimv2_official,
            "optional_transformers_auto": optional,
        },
        "notes": {
            "AutoImageProcessor": "optional -- only used by transformers-fallback backend",
            "deimv2_backend": os.environ.get("DEIMV2_BACKEND", "official-deimv2-hf"),
        },
    }

# -- Debug / startup diagnostics ----------------------------------------------

@app.get("/debug/startup")
async def debug_startup(deep: bool = Query(False)):
    """Environment info, disk usage, and optional deep import diagnostics."""
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
            info["diagnostics"] = _collect_import_diagnostics()
        except Exception as exc:
            info["diagnostics_error"] = f"{type(exc).__name__}: {exc}"

    return JSONResponse(info)

# -- Model-load diagnostic route ----------------------------------------------

@app.post("/debug/model-load")
async def debug_model_load():
    """
    Attempt ONLY the model-loading part (no inference) using the OFFICIAL DEIMv2
    loader (PyTorchModelHubMixin), or the transformers-fallback backend if
    DEIMV2_BACKEND=transformers-fallback. Never crashes the worker; on failure
    returns the full structured traceback.
    """
    model_id = os.environ.get("DEIMV2_MODEL_ID", DEFAULT_MODEL_ID)
    backend = os.environ.get("DEIMV2_BACKEND", "official-deimv2-hf").strip().lower()
    result: Dict[str, Any] = {
        "ok": False,
        "backend": "official-deimv2-hf" if backend != "transformers-fallback" else "transformers-fallback",
        "model_id": model_id,
        "device": None,
        "transformers_version": None,
    }
    try:
        import transformers
        result["transformers_version"] = transformers.__version__
    except Exception as exc:
        result["transformers_version"] = f"ERROR: {type(exc).__name__}: {exc}"

    try:
        import torch
        device = "cuda" if (os.environ.get("DEIMV2_DEVICE", "cuda").lower() == "cuda"
                            and torch.cuda.is_available()) else "cpu"
        result["device"] = device
    except Exception as exc:
        result["device"] = f"ERROR: {type(exc).__name__}: {exc}"

    try:
        if backend == "transformers-fallback":
            from deimv2_infer import _load_fallback
            import deimv2_infer
            deimv2_infer._model = None
            _load_fallback()
            result["ok"] = True
            result["model_class"] = type(deimv2_infer._model).__name__
            result["backend"] = "transformers-fallback"
            deimv2_infer._model = None
        else:
            from official_deimv2_loader import load_official_deimv2
            model, dev, cls = load_official_deimv2(model_id=model_id)
            result["ok"] = True
            result["model_class"] = cls
            result["backend"] = "official-deimv2-hf"
            del model
    except Exception as exc:
        result["ok"] = False
        result["exception_type"] = type(exc).__name__
        result["exception_message"] = str(exc)
        result["traceback"] = traceback.format_exc()
        _log_startup(f"debug/model-load FAILED: {type(exc).__name__}: {exc}")

    return JSONResponse(result)

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
