"""
server.py -- FastAPI/uvicorn live-server for the SafeLens vision worker.

Architecture: long-running HTTP server (RunPod load-balancing endpoint mode).
Pattern: adapted from Kingo333/fluxrt-serverless.

Backends (selected via VISION_BACKEND):
    edgecrafter (default) -> EdgeCrafter ECDet-S boxes + optional ECPose-S poses
    deimv2      (fallback) -> legacy DEIMv2 boxes only

Routes
------
GET  /health         -- returns immediately, no model required
GET  /ping           -- alias for /health
GET  /debug/startup  -- environment + torch diagnostics (?deep=true for imports)
POST /debug/model-load -- attempt model load only, return structured result
POST /warmup         -- trigger background model load
POST /detect         -- run inference (503 if model not ready)
WS   /ws/echo        -- Phase 0 WebSocket connectivity probe (echoes JSON)
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

from fastapi import FastAPI, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("safelens-vision-worker")

# -- Config -------------------------------------------------------------------

PORT = int(os.getenv("PORT", "8000"))
WORKER_VERSION = "0.4.0-edgecrafter"
SKIP_WARMUP = os.getenv("SKIP_WARMUP", "false").strip().lower() in ("1", "true", "yes", "on")
AUTO_WARMUP = os.getenv("AUTO_WARMUP", "true").strip().lower() in ("1", "true", "yes", "on")
WARMUP_TIMEOUT_S = int(os.getenv("WARMUP_TIMEOUT_S", "600"))
STARTUP_LOG_PATH = Path(os.getenv("STARTUP_LOG", "/tmp/safelens_startup.log"))

def _active_backend():
    return os.getenv("VISION_BACKEND", "edgecrafter").strip().lower()

# -- State --------------------------------------------------------------------

_STATE_LOCK = threading.RLock()
_STATE = {
    "status": "cold",
    "error": None,
    "error_traceback": None,
    "summary": None,
    "ready_at": None,
    "warmup_started_at": None,
}
_BOOT_TS = time.time()

# In-memory tail of startup log lines, surfaced by GET /debug/state.
_STARTUP_LOG_BUFFER = []
_STARTUP_LOG_MAX = 200

def _log_startup(msg):
    line = "[" + time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()) + "] " + msg
    print(line, flush=True)
    _STARTUP_LOG_BUFFER.append(line)
    if len(_STARTUP_LOG_BUFFER) > _STARTUP_LOG_MAX:
        del _STARTUP_LOG_BUFFER[:-_STARTUP_LOG_MAX]
    try:
        STARTUP_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with STARTUP_LOG_PATH.open("a") as fh:
            fh.write(line + "\n")
    except Exception:
        pass

def _public_state():
    with _STATE_LOCK:
        return {
            "status": _STATE["status"],
            "model_loaded": _STATE["status"] == "ready",
            "backend": _active_backend(),
            "error": _STATE["error"],
            "error_traceback": _STATE["error_traceback"],
            "warmup_started_at": _STATE["warmup_started_at"],
            "ready_at": _STATE["ready_at"],
            "uptime_seconds": round(time.time() - _BOOT_TS, 2),
        }

def _ckpt_exists(env_name):
    """True if the checkpoint path held in env var `env_name` exists on disk."""
    path = os.getenv(env_name, "")
    try:
        return bool(path) and os.path.exists(path)
    except Exception:
        return False

def _torch_cuda_info():
    """(cuda_available, gpu_name) from torch if importable, else (False, None)."""
    try:
        import torch
        if torch.cuda.is_available():
            return True, torch.cuda.get_device_name(0)
    except Exception:
        pass
    return False, None

# -- Model warmup (background thread) -----------------------------------------

def _warmup_blocking():
    with _STATE_LOCK:
        if _STATE["status"] in ("loading", "ready"):
            return
        _STATE["status"] = "loading"
        _STATE["error"] = None
        _STATE["error_traceback"] = None
        _STATE["warmup_started_at"] = time.time()

    _log_startup("warmup: starting model load (backend=" + _active_backend() + ")")
    try:
        from vision_backend import load_models
        summary = load_models()
        with _STATE_LOCK:
            _STATE["summary"] = summary
            _STATE["status"] = "ready"
            _STATE["ready_at"] = time.time()
        _log_startup("warmup: ready -- " + str(summary))
    except Exception as exc:
        err_msg = type(exc).__name__ + ": " + str(exc)
        tb = traceback.format_exc()
        _log_startup("warmup: FAILED -- " + err_msg)
        _log_startup(tb)
        with _STATE_LOCK:
            _STATE["status"] = "error"
            _STATE["error"] = err_msg
            _STATE["error_traceback"] = tb

def _start_warmup_background():
    with _STATE_LOCK:
        if _STATE["status"] in ("loading", "ready"):
            return
    t = threading.Thread(target=_warmup_blocking, name="vision-warmup", daemon=True)
    t.start()

# -- FastAPI lifespan ---------------------------------------------------------

@asynccontextmanager
async def lifespan(app):
    _log_startup("server starting")
    _log_startup("port=" + str(PORT))
    _log_startup("version=" + WORKER_VERSION)
    _log_startup("backend=" + _active_backend())
    _log_startup("SKIP_WARMUP=" + str(SKIP_WARMUP))
    _log_startup("AUTO_WARMUP=" + str(AUTO_WARMUP))
    _log_startup("EDGECRAFTER_TASKS=" + os.getenv("EDGECRAFTER_TASKS", ""))
    _log_startup("EDGECRAFTER_REPO_DIR=" + os.getenv("EDGECRAFTER_REPO_DIR", ""))
    _log_startup("det_checkpoint_exists=" + str(_ckpt_exists("EDGECRAFTER_DET_CHECKPOINT_PATH")))
    _log_startup("pose_checkpoint_exists=" + str(_ckpt_exists("EDGECRAFTER_POSE_CHECKPOINT_PATH")))
    _cuda_available, _gpu_name = _torch_cuda_info()
    _log_startup("cuda_available=" + str(_cuda_available))
    if SKIP_WARMUP:
        _log_startup("SKIP_WARMUP=true: skipping model load (diagnostic mode)")
    elif AUTO_WARMUP:
        _log_startup("AUTO_WARMUP=true: triggering background model load")
        _start_warmup_background()
    else:
        _log_startup("AUTO_WARMUP=false: model load deferred until POST /warmup")
    yield
    _log_startup("server shutting down")

app = FastAPI(title="safelens-vision-worker", version=WORKER_VERSION, lifespan=lifespan)

# -- Health / ping (no model dependency) -------------------------------------

@app.get("/health")
async def health():
    return JSONResponse({
        "ok": True, "worker": "safelens-vision-worker",
        "mode": "live-server", "version": WORKER_VERSION, **_public_state(),
    })

@app.get("/ping")
async def ping():
    return JSONResponse({
        "ok": True, "worker": "safelens-vision-worker",
        "mode": "live-server", "version": WORKER_VERSION, **_public_state(),
    })

# -- Import diagnostics helper ------------------------------------------------

def _safe_version(module_name):
    try:
        mod = __import__(module_name)
        return getattr(mod, "__version__", "unknown")
    except Exception as exc:
        return "ERROR: " + type(exc).__name__ + ": " + str(exc)

def _import_check(callable_test):
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

def _checkpoint_info():
    """Existence + file size for the EdgeCrafter checkpoints (no secrets)."""
    out = {}
    for key, env in (("det", "EDGECRAFTER_DET_CHECKPOINT_PATH"),
                     ("pose", "EDGECRAFTER_POSE_CHECKPOINT_PATH")):
        path = os.getenv(env, "")
        info = {"path": path, "exists": False, "size_bytes": 0}
        try:
            p = Path(path)
            if path and p.exists():
                info["exists"] = True
                info["size_bytes"] = p.stat().st_size
        except Exception as exc:
            info["error"] = type(exc).__name__ + ": " + str(exc)
        out[key] = info
    return out

def _collect_edgecrafter_diagnostics():
    def _imp_torchvision():
        import torchvision  # noqa: F401
    def _imp_loader():
        import edgecrafter_loader  # noqa: F401
    def _imp_det_engine():
        import edgecrafter_loader as ec
        ec._purge_engine_modules()
        ec._add_subtree_to_path("ecdetseg")
        from engine.core import YAMLConfig  # noqa: F401
    def _imp_pose_engine():
        import edgecrafter_loader as ec
        ec._purge_engine_modules()
        ec._add_subtree_to_path("ecpose")
        from engine.core import YAMLConfig  # noqa: F401

    imports = {
        "torchvision": _import_check(_imp_torchvision),
        "edgecrafter_loader": _import_check(_imp_loader),
        "ecdetseg.engine.core": _import_check(_imp_det_engine),
        "ecpose.engine.core": _import_check(_imp_pose_engine),
    }

    tasks = []
    try:
        import edgecrafter_loader as ec
        tasks = ec.parse_tasks()
    except Exception:
        pass

    return {
        "vision_backend": _active_backend(),
        "edgecrafter_tasks": tasks,
        "edgecrafter_tasks_env": os.getenv("EDGECRAFTER_TASKS", ""),
        "imports": {k: v.get("status") for k, v in imports.items()},
        "imports_detail": imports,
        "checkpoints": _checkpoint_info(),
    }

def _collect_import_diagnostics():
    versions = {}
    for name in ("torch", "torchvision", "PIL", "numpy", "transformers",
                 "huggingface_hub", "yaml", "cv2"):
        versions[name] = _safe_version(name)

    cuda = {}
    try:
        import torch
        cuda["available"] = bool(torch.cuda.is_available())
        cuda["device_count"] = torch.cuda.device_count()
        if torch.cuda.is_available():
            cuda["gpu_name"] = torch.cuda.get_device_name(0)
    except Exception as exc:
        cuda["error"] = type(exc).__name__ + ": " + str(exc)

    return {
        "versions": versions,
        "cuda": cuda,
        "edgecrafter": _collect_edgecrafter_diagnostics(),
    }

# -- Debug / startup diagnostics ----------------------------------------------

@app.get("/debug/startup")
async def debug_startup(deep: bool = Query(False)):
    total, used, free = shutil.disk_usage("/")
    info = {
        "ok": True,
        "worker": "safelens-vision-worker",
        "version": WORKER_VERSION,
        "mode": "live-server",
        "port": PORT,
        "vision_backend": _active_backend(),
        "edgecrafter_tasks": os.getenv("EDGECRAFTER_TASKS", ""),
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
            info["torch_error"] = type(exc).__name__ + ": " + str(exc)
        try:
            info["diagnostics"] = _collect_import_diagnostics()
        except Exception as exc:
            info["diagnostics_error"] = type(exc).__name__ + ": " + str(exc)

    return JSONResponse(info)

# -- Model-load diagnostic route ----------------------------------------------

@app.post("/debug/model-load")
async def debug_model_load():
    """Attempt ONLY model loading (no inference). Never crashes the worker."""
    try:
        from vision_backend import model_load_summary
        result = model_load_summary()
        if result.get("ok"):
            with _STATE_LOCK:
                _STATE["summary"] = result
                if _STATE["status"] != "ready":
                    _STATE["status"] = "ready"
                    _STATE["ready_at"] = time.time()
        else:
            _log_startup("debug/model-load FAILED: " + str(result.get("exception_type")) + ": " + str(result.get("exception_message")))
        return JSONResponse(result)
    except Exception as exc:
        return JSONResponse({
            "ok": False,
            "backend": _active_backend(),
            "exception_type": type(exc).__name__,
            "exception_message": str(exc),
            "traceback": traceback.format_exc(),
        })

# -- State diagnostics route --------------------------------------------------

@app.get("/debug/state")
async def debug_state():
    """Non-sensitive snapshot of worker state, config, checkpoints and GPU.

    Surfaces only the EdgeCrafter config env vars listed below -- never
    secrets, API keys, or tokens (e.g. HF_TOKEN is deliberately excluded).
    """
    cuda_available, gpu_name = _torch_cuda_info()
    return JSONResponse({
        "ok": True,
        "worker_version": WORKER_VERSION,
        "backend": _active_backend(),
        "skip_warmup": SKIP_WARMUP,
        "auto_warmup": AUTO_WARMUP,
        "state": _public_state(),
        "env_subset": {
            "VISION_BACKEND": os.getenv("VISION_BACKEND", ""),
            "EDGECRAFTER_TASKS": os.getenv("EDGECRAFTER_TASKS", ""),
            "EDGECRAFTER_DEVICE": os.getenv("EDGECRAFTER_DEVICE", ""),
            "EDGECRAFTER_IMG_SIZE": os.getenv("EDGECRAFTER_IMG_SIZE", ""),
            "EDGECRAFTER_CONF": os.getenv("EDGECRAFTER_CONF", ""),
            "EDGECRAFTER_REPO_DIR": os.getenv("EDGECRAFTER_REPO_DIR", ""),
            "EDGECRAFTER_DET_CONFIG": os.getenv("EDGECRAFTER_DET_CONFIG", ""),
            "EDGECRAFTER_DET_CHECKPOINT_PATH": os.getenv("EDGECRAFTER_DET_CHECKPOINT_PATH", ""),
            "EDGECRAFTER_POSE_CONFIG": os.getenv("EDGECRAFTER_POSE_CONFIG", ""),
            "EDGECRAFTER_POSE_CHECKPOINT_PATH": os.getenv("EDGECRAFTER_POSE_CHECKPOINT_PATH", ""),
        },
        "checkpoint_exists": {
            "det": _ckpt_exists("EDGECRAFTER_DET_CHECKPOINT_PATH"),
            "pose": _ckpt_exists("EDGECRAFTER_POSE_CHECKPOINT_PATH"),
        },
        "cuda_available": cuda_available,
        "gpu_name": gpu_name,
        "startup_log_tail": _STARTUP_LOG_BUFFER[-50:],
    })

# -- Warmup trigger -----------------------------------------------------------

@app.post("/warmup")
async def warmup(wait: bool = Query(False)):
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
    """Run inference for the active backend.

    Returns entities (boxes) and poses (keypoints/skeleton). On any failure,
    returns a structured error with entities: [] and poses: [].
    """
    state = _public_state()
    if state["status"] != "ready":
        # Not ready yet: log the full state, trigger the existing background
        # warmup, then return a diagnostic model_not_ready body. error stays
        # "model_not_ready" so the app's existing handling is unchanged; the
        # underlying load error (if any) is in error_traceback / GET /debug/state.
        log.info("detect: model_not_ready -- state=%s", state)
        _start_warmup_background()
        return JSONResponse(
            {
                "error": "model_not_ready",
                "status": state["status"],
                "warmup_triggered": True,
                "backend": state["backend"],
                "model_loaded": state["model_loaded"],
                "error_traceback": state["error_traceback"],
                "warmup_started_at": state["warmup_started_at"],
                "ready_at": state["ready_at"],
                "entities": [],
                "poses": [],
            },
            status_code=503,
        )

    import binascii
    import base64 as _b64

    image_b64 = payload.get("image_b64")
    if not image_b64:
        return JSONResponse(
            {"error": "missing_image_b64", "entities": [], "poses": []},
            status_code=400,
        )
    try:
        _b64.b64decode(image_b64, validate=True)
    except (binascii.Error, ValueError):
        return JSONResponse(
            {"error": "invalid_base64", "entities": [], "poses": []},
            status_code=400,
        )

    try:
        from vision_backend import run_inference
        conf = float(os.getenv("EDGECRAFTER_CONF", payload.get("conf", 0.25)))
        img_size = int(os.getenv("EDGECRAFTER_IMG_SIZE", payload.get("img_size", 640)))
        class_filter = payload.get("classes")
        resp = run_inference(
            image_b64=image_b64, conf=conf,
            img_size=img_size, class_filter=class_filter,
        )
        return JSONResponse(resp.model_dump())
    except Exception as exc:
        log.exception("detect failed: %s", exc)
        return JSONResponse(
            {"error": "inference_failed: " + type(exc).__name__ + ": " + str(exc),
             "entities": [], "poses": [], "backend": _active_backend()},
            status_code=500,
        )

# -- Phase 0: WebSocket connectivity probe ------------------------------------

@app.websocket("/ws/echo")
async def ws_echo(websocket: WebSocket):
    """Minimal WebSocket echo route.

    Phase 0 connectivity probe used to verify that the RunPod load-balancing
    endpoint forwards and upgrades WebSocket connections end to end. It does
    NOT touch the model or any inference path.

    Behaviour:
      * accept the connection
      * send {"type": "connected"}
      * echo every received JSON message back verbatim
      * close cleanly on client disconnect (never crashes the server)
    """
    await websocket.accept()
    await websocket.send_json({"type": "connected"})
    try:
        while True:
            try:
                data = await websocket.receive_json()
            except (ValueError, TypeError):
                await websocket.send_json({"type": "error", "error": "invalid_json"})
                continue
            await websocket.send_json(data)
    except WebSocketDisconnect:
        log.info("ws/echo: client disconnected")
    except Exception as exc:
        log.warning("ws/echo: unexpected error: %s", exc)
        try:
            await websocket.close()
        except Exception:
            pass

# -- Phase 1: streaming vision WebSocket (/ws/vision) -------------------------
#
# Registered through ws_vision.register_ws_vision with injected dependencies so
# the streaming layer REUSES the existing warmed model + warmup + inference path
# (vision_backend.run_inference / _start_warmup_background). No model is loaded
# here and nothing is reloaded per frame. ws_vision is imported lazily and the
# whole block is guarded so a problem in the streaming module can never break
# server boot or any of the existing routes above.

def _ws_run_inference(image_b64, conf, img_size, class_filter):
    """Run one streamed frame through the same backend path as POST /detect."""
    from vision_backend import run_inference
    return run_inference(
        image_b64=image_b64, conf=conf,
        img_size=img_size, class_filter=class_filter,
    )

def _ws_active_tasks():
    """Active EdgeCrafter tasks, with a torch-free fallback to the env parse."""
    if _active_backend() == "deimv2":
        return ["det"]
    try:
        import edgecrafter_loader as ec
        return list(ec._STATE.tasks) if ec._STATE.tasks else ec.parse_tasks()
    except Exception:
        raw = os.getenv("EDGECRAFTER_TASKS", "det,pose")
        out = [t.strip().lower() for t in raw.split(",")
               if t.strip().lower() in ("det", "pose")]
        return out or ["det"]

def _ws_gpu_device():
    """GPU device name if CUDA is available, else None (never raises)."""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception:
        pass
    return None

try:
    import ws_vision
    ws_vision.register_ws_vision(
        app,
        get_state=_public_state,
        trigger_warmup=_start_warmup_background,
        run_inference=_ws_run_inference,
        get_backend=_active_backend,
        get_tasks=_ws_active_tasks,
        get_gpu_device=_ws_gpu_device,
    )
    _log_startup("ws/vision: streaming route + /debug/stream registered")
except Exception as exc:  # noqa: BLE001 -- streaming must never break server boot
    _log_startup("ws/vision: registration FAILED -- " + type(exc).__name__ + ": " + str(exc))
    log.warning("ws/vision registration failed: %s", exc)

# -- Entrypoint ---------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level=os.getenv("UVICORN_LOG_LEVEL", "info"))
