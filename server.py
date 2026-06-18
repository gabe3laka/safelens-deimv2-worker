"""
server.py -- FastAPI/uvicorn live-server for the SafeLens vision worker.

Architecture: long-running HTTP server (RunPod load-balancing endpoint mode).
Pattern: adapted from Kingo333/fluxrt-serverless.

Backends (selected via VISION_BACKEND):
    yolo26      (default)  -> YOLO26 boxes + poses (+ optional seg)
    edgecrafter (fallback) -> EdgeCrafter ECDet-S boxes + optional ECPose-S poses
    deimv2      (legacy)   -> DEIMv2 boxes only (debug)

Routes
------
GET  /health         -- returns immediately, no model required
GET  /ping           -- alias for /health
GET  /debug/startup  -- environment + torch diagnostics (?deep=true for imports)
GET  /debug/state    -- worker state/config/checkpoints/GPU snapshot (no secrets)
GET  /debug/stream   -- latest /ws/vision streaming-metrics snapshot
POST /debug/model-load -- attempt model load only, return structured result
POST /warmup         -- trigger background model load
POST /detect         -- run inference (503 if model not ready)
WS   /ws/echo        -- Phase 0 WebSocket connectivity probe (echoes JSON)
WS   /ws/vision      -- streaming inference (frames in, vision + metrics out)
POST /build/session/start  -- Build Mode: open a blueprint session
POST /build/session/lock   -- Build Mode: lock the selected region
POST /build/session/frame  -- Build Mode: process one selected-crop keyframe
POST /build/session/finish -- Build Mode: close session, return replay id
GET  /build/session/{id}/replay -- Build Mode: replay stored JSON keyframes
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

from fastapi import FastAPI, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, PlainTextResponse

import worker_guards as guards
import worker_runtime as runtime
import worker_security as security

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
    return os.getenv("VISION_BACKEND", "yolo26").strip().lower()

def _backend_status_safe():
    """vision_backend.backend_status(), guarded so debug routes never crash."""
    try:
        from vision_backend import backend_status
        return backend_status()
    except Exception as exc:  # noqa: BLE001
        return {"error": type(exc).__name__ + ": " + str(exc)}

def _plan_context_safe():
    """plan_context.config(), guarded so debug routes never crash."""
    try:
        import plan_context
        return plan_context.config()
    except Exception as exc:  # noqa: BLE001
        return {"error": type(exc).__name__ + ": " + str(exc)}

def _risk_config_safe():
    """risk.config(), guarded so debug routes never crash."""
    try:
        import risk
        return risk.config()
    except Exception as exc:  # noqa: BLE001
        return {"error": type(exc).__name__ + ": " + str(exc)}

def _reasoner_status_safe():
    """vlm_reasoner.status_snapshot(), guarded so debug routes never crash."""
    try:
        import risk.vlm_reasoner as vlm
        return vlm.status_snapshot()
    except Exception as exc:  # noqa: BLE001
        return {"error": type(exc).__name__ + ": " + str(exc)}

def _open_vocab_config_safe():
    """open_vocab_scanner.config(), guarded so debug routes never crash."""
    try:
        import risk.open_vocab_scanner as ovs
        return ovs.config()
    except Exception as exc:  # noqa: BLE001
        return {"error": type(exc).__name__ + ": " + str(exc)}

def _temporal_config_safe():
    """temporal_reasoning.config(), guarded so debug routes never crash."""
    try:
        import temporal_reasoning
        return temporal_reasoning.config()
    except Exception as exc:  # noqa: BLE001
        return {"error": type(exc).__name__ + ": " + str(exc)}

def _agentic_cpu_config_safe():
    """agentic_cpu.status_snapshot(), guarded so debug routes never crash."""
    try:
        import agentic_cpu
        return agentic_cpu.status_snapshot()
    except Exception as exc:  # noqa: BLE001
        return {"error": type(exc).__name__ + ": " + str(exc)}

def _gpu_vision_snapshot_safe():
    """gpu_vision.snapshot(), guarded so debug routes never crash."""
    try:
        import gpu_vision
        return gpu_vision.snapshot()
    except Exception as exc:  # noqa: BLE001
        return {"error": type(exc).__name__ + ": " + str(exc)}

def _runtime_block_safe():
    """build_sha + degradation + shutdown + guards/auth snapshot (no secrets)."""
    try:
        return {
            "build_sha": runtime.build_sha(),
            "uptime_s": runtime.uptime_s(),
            "degraded": runtime.degradation_mode() != "full",
            "degradation_mode": runtime.degradation_mode(),
            "degradation_ladder": list(runtime.LADDER),
            "accepting_frames": runtime.accepting(),
            "auth": security.config(),
            "input_guards": guards.config(),
            "active_sessions": _active_session_count(),
        }
    except Exception as exc:  # noqa: BLE001
        return {"error": type(exc).__name__ + ": " + str(exc)}

def _active_session_count():
    """Active risk-tracker sessions (torch-free; never raises)."""
    try:
        import risk.tracking as tracking
        return tracking.active_session_count()
    except Exception:  # noqa: BLE001
        return 0

def _ready_state():
    """Compute readiness: model loaded AND config valid AND matrix loaded.

    Returns (ready: bool, detail: dict). Used by GET /ready and /metrics.
    """
    detail = {"model_loaded": False, "matrix_valid": True, "config_valid": True,
              "accepting_frames": runtime.accepting(), "checks": {}}
    state = _public_state()
    detail["model_loaded"] = bool(state.get("model_loaded"))
    detail["status"] = state.get("status")
    # risk matrix must load + validate when the risk engine is enabled.
    try:
        import risk.risk_engine as re
        if re.enabled():
            from risk import risk_matrix
            risk_matrix.get_matrix()  # raises on malformed profile
            detail["checks"]["risk_matrix"] = "ok"
        else:
            detail["checks"]["risk_matrix"] = "disabled"
    except Exception as exc:  # noqa: BLE001
        detail["matrix_valid"] = False
        detail["checks"]["risk_matrix"] = "invalid: " + type(exc).__name__ + ": " + str(exc)
    ready = (detail["model_loaded"] and detail["matrix_valid"]
             and detail["config_valid"] and detail["accepting_frames"])
    return ready, detail

def _effective_config_safe():
    """Effective inference config for the actual serving backend (no secrets).

    Mirrors what /detect resolves (with an empty payload), so the actual
    backend/conf/img_size/iou/max_det can be verified from GET /debug/state.
    """
    try:
        import config_resolver
        try:
            from vision_backend import serving_backend
            backend = serving_backend()
        except Exception:  # noqa: BLE001
            backend = _active_backend()
        return config_resolver.resolve_effective_inference_config(backend, {})
    except Exception as exc:  # noqa: BLE001
        return {"error": type(exc).__name__ + ": " + str(exc)}

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

def _drain_s():
    try:
        return max(0.0, int(os.getenv("GRACEFUL_DRAIN_MS", "1500")) / 1000.0)
    except (TypeError, ValueError):
        return 1.5

@asynccontextmanager
async def lifespan(app):
    runtime.reset_shutdown()  # clear any stale flag (dev/test re-starts)
    _log_startup("server starting")
    _log_startup("build_sha=" + runtime.build_sha())
    _log_startup("worker_auth_enabled=" + str(security.auth_enabled()))
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
    # Graceful shutdown (B5): stop accepting new frames, drain in-flight work,
    # close WebSockets cleanly, then let uvicorn exit. Triggered by SIGTERM via
    # uvicorn's lifespan shutdown.
    runtime.begin_shutdown()
    _log_startup("server shutting down (graceful): no longer accepting frames")
    closed = 0
    try:
        import ws_vision
        closed = ws_vision.shutdown_all()
        _log_startup("ws/vision: closed " + str(closed) + " streaming session(s)")
    except Exception as exc:  # noqa: BLE001
        _log_startup("ws/vision shutdown error: " + type(exc).__name__ + ": " + str(exc))
    if closed > 0:
        await asyncio.sleep(min(2.0, _drain_s()))
    _log_startup("server shutdown complete")

app = FastAPI(title="safelens-vision-worker", version=WORKER_VERSION, lifespan=lifespan)


# -- Security + input-size middleware (B7) ------------------------------------
# Shared-secret auth on every route except /health,/ping (disabled in compat/
# test mode when WORKER_SHARED_SECRET is unset). A Content-Length cap rejects
# oversized bodies with a structured 4xx before they are read. Never logs the
# secret/auth header (worker_runtime.redact).
@app.middleware("http")
async def _hardening_middleware(request: Request, call_next):
    path = request.url.path
    if not security.is_public(path):
        if not security.check_http(request.headers):
            runtime.inc("auth_rejected_total", {"path": path})
            return JSONResponse({"error": "unauthorized",
                                 "detail": "missing or invalid worker secret"},
                                status_code=401)
        cl_err = guards.content_length_error(request.headers)
        if cl_err:
            runtime.inc("input_rejected_total", {"error": cl_err})
            return JSONResponse({"error": cl_err, "entities": [], "poses": []},
                                status_code=guards.status_for(cl_err))
    return await call_next(request)

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

# -- Readiness (distinct from liveness) ---------------------------------------

@app.get("/ready")
async def ready():
    """Readiness gate: 200 only when model loaded AND config/matrix valid AND
    accepting frames. /health and /ping stay liveness (always 200). The gateway
    should consult /ready before routing live traffic."""
    is_ready, detail = _ready_state()
    runtime.set_gauge("model_ready", float(detail["model_loaded"]))
    runtime.set_gauge("ready", float(is_ready))
    # Both layers surfaced; overall readiness is the GPU VISION worker only --
    # the CPU agent layer being disabled/not-ready must NOT fail vision readiness.
    agentic_enabled, agentic_ready = False, False
    try:
        import agentic_cpu
        agentic_enabled = agentic_cpu.enabled()
        agentic_ready = agentic_cpu.ready()
    except Exception:  # noqa: BLE001
        pass
    degraded = runtime.degradation_mode()
    body = {"ok": is_ready, "ready": is_ready, "build_sha": runtime.build_sha(),
            "degradation_mode": degraded,
            "degraded_mode": None if degraded == "full" else degraded,
            "gpu_vision_ready": is_ready,
            "agentic_cpu_ready": agentic_ready,
            "agentic_cpu_enabled": agentic_enabled, **detail}
    return JSONResponse(body, status_code=200 if is_ready else 503)

# -- Prometheus metrics -------------------------------------------------------

@app.get("/metrics")
async def metrics():
    """Prometheus text metrics. Separate from HSE risk alerts (system health)."""
    is_ready, detail = _ready_state()
    live = {
        "model_ready": float(detail["model_loaded"]),
        "ready": float(is_ready),
        "active_sessions": float(_active_session_count()),
        "degradation_rank": float(runtime.LADDER.index(runtime.degradation_mode())
                                  if runtime.degradation_mode() in runtime.LADDER else 0),
        "accepting_frames": float(runtime.accepting()),
    }
    try:
        from ws_vision import _GLOBAL
        live["ws_dropped_frames_total"] = float(_GLOBAL["totals"]["dropped_frames"])
        live["ws_processed_frames_total"] = float(_GLOBAL["totals"]["processed_frames"])
    except Exception:  # noqa: BLE001
        pass
    try:
        import risk.risk_engine as re
        last = re._LAST
        live["risk_active_tracks"] = float(last.get("active_tracks", 0))
        live["risk_last_count"] = float(last.get("risk_count", 0))
        live["risk_last_alerting"] = float(last.get("alerting_count", 0))
    except Exception:  # noqa: BLE001
        pass
    # GPU reasoner + CPU agent + temporal queue/job gauges (always present).
    try:
        import gpu_vision
        live["gpu_reasoner_jobs_inflight"] = float(gpu_vision.inflight())
        live["gpu_busy_ratio"] = float(gpu_vision.gpu_busy_ratio())
    except Exception:  # noqa: BLE001
        pass
    try:
        import agentic_cpu.jobs as _ajobs
        live["cpu_agent_jobs_inflight"] = float(_ajobs.queue_depth())
        live["cpu_agent_queue_depth"] = float(_ajobs.queue_depth())
    except Exception:  # noqa: BLE001
        pass
    try:
        import temporal_reasoning
        live["temporal_pending_reasoner_jobs"] = float(temporal_reasoning.pending_reasoner_jobs())
    except Exception:  # noqa: BLE001
        pass
    # Seed expected counters so they always render (value unchanged if no events).
    for _counter in ("gpu_reasoner_jobs_dropped_total", "temporal_triggers_total",
                     "cpu_agent_jobs_completed_total", "cpu_agent_jobs_failed_total",
                     "cpu_agent_approval_required_total"):
        runtime.inc(_counter, n=0.0)
    return PlainTextResponse(runtime.render_prometheus(live), media_type="text/plain; version=0.0.4")

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
        info["backend_status"] = _backend_status_safe()
        info["plan_context"] = _plan_context_safe()

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

    Surfaces only the config env vars listed below -- never
    secrets, API keys, or tokens (e.g. HF_TOKEN is deliberately excluded).
    """
    from config_resolver import get_effective_config_summary
    from vision_backend import get_last_detect_config
    
    cuda_available, gpu_name = _torch_cuda_info()
    
    return JSONResponse({
        "ok": True,
        "worker_version": WORKER_VERSION,
        "backend": _active_backend(),
        "backend_status": _backend_status_safe(),
        "effective_config": _effective_config_safe(),
        "plan_context": _plan_context_safe(),
        "risk_engine": _risk_config_safe(),
        "reasoner": _reasoner_status_safe(),
        "open_vocab_scanner": _open_vocab_config_safe(),
        "gpu_vision": {"enabled": True, "backend": _active_backend(),
                       **_gpu_vision_snapshot_safe()},
        "temporal_reasoning": _temporal_config_safe(),
        "agentic_cpu": _agentic_cpu_config_safe(),
        "runtime": _runtime_block_safe(),
        "skip_warmup": SKIP_WARMUP,
        "auto_warmup": AUTO_WARMUP,
        "state": _public_state(),
        "effective_config": get_effective_config_summary(),
        "last_detect_effective_config": get_last_detect_config(),
        "env_subset": {
            "VISION_BACKEND": os.getenv("VISION_BACKEND", ""),
            "FALLBACK_VISION_BACKEND": os.getenv("FALLBACK_VISION_BACKEND", ""),
            "AUTO_BACKEND_FALLBACK": os.getenv("AUTO_BACKEND_FALLBACK", ""),
            "YOLO26_MODEL_ID": os.getenv("YOLO26_MODEL_ID", ""),
            "YOLO26_DET_MODEL_ID": os.getenv("YOLO26_DET_MODEL_ID", ""),
            "YOLO26_SEG_MODEL_ID": os.getenv("YOLO26_SEG_MODEL_ID", ""),
            "YOLO26_POSE_MODEL_ID": os.getenv("YOLO26_POSE_MODEL_ID", ""),
            "YOLO26_TASKS": os.getenv("YOLO26_TASKS", ""),
            "YOLO26_LIVE_TASKS": os.getenv("YOLO26_LIVE_TASKS", ""),
            "YOLO26_BUILD_TASKS": os.getenv("YOLO26_BUILD_TASKS", ""),
            "YOLO26_PLAN_TASKS": os.getenv("YOLO26_PLAN_TASKS", ""),
            "YOLO26_POSE_ENABLED": os.getenv("YOLO26_POSE_ENABLED", ""),
            "YOLO26_SEG_EVERY_N": os.getenv("YOLO26_SEG_EVERY_N", ""),
            "YOLO26_DEVICE": os.getenv("YOLO26_DEVICE", ""),
            "YOLO26_IMG_SIZE": os.getenv("YOLO26_IMG_SIZE", ""),
            "YOLO26_CONF": os.getenv("YOLO26_CONF", ""),
            "YOLO26_IOU": os.getenv("YOLO26_IOU", ""),
            "YOLO26_MAX_DETECTIONS": os.getenv("YOLO26_MAX_DETECTIONS", ""),
            "YOLO26_CACHE_DIR": os.getenv("YOLO26_CACHE_DIR", ""),
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

# -- HSE intent detection + Live scene-risk helpers ---------------------------

def wants_hse_reasoning(payload: Dict[str, Any]) -> bool:
    """Return True when the payload signals live HSE scene-reasoning intent.

    Checked flags (any one suffices):
      * mode == "hse-monitoring"
      * scene_hint == "live_hse_monitoring"
      * "scene_reasoning" in tasks
      * reasoning_preferences.return_scene_risks == True
      * reasoning_preferences.return_reasoner_status == True
    """
    if payload.get("mode") == "hse-monitoring":
        return True
    if payload.get("scene_hint") == "live_hse_monitoring":
        return True
    tasks = payload.get("tasks") or []
    if "scene_reasoning" in tasks:
        return True
    prefs = payload.get("reasoning_preferences") or {}
    if prefs.get("return_scene_risks") is True:
        return True
    if prefs.get("return_reasoner_status") is True:
        return True
    return False


# Internal -> app-facing reasoner state vocabulary.
_RAW_TO_REASONER_STATE: Dict[str, str] = {
    "disabled":             "disabled",
    "not_triggered":        "rules_only",
    "throttled":            "queued",
    "triggered":            "running",
    "cached":               "ready",
    "cached_and_triggered": "running",
    "error":                "error",
    "timeout":              "timeout",
    "unavailable":          "unavailable",
    "ok":                   "ready",
}


def _normalize_reasoner_status(raw: Any, model_id: Optional[str] = None) -> Dict[str, Any]:
    """Convert a raw (string or dict) reasoner status to the standard app-facing dict.

    Standard states: ready | running | queued | unavailable | timeout |
                     disabled | rules_only | error
    """
    if isinstance(raw, dict):
        state = raw.get("state") or raw.get("status") or "unavailable"
        out = dict(raw)
        out["state"] = _RAW_TO_REASONER_STATE.get(state, state)
        return out
    state = _RAW_TO_REASONER_STATE.get(raw or "", raw or "unavailable")
    d: Dict[str, Any] = {"state": state}
    if model_id:
        d["model"] = model_id
    return d


def _is_linkable(risk: Dict[str, Any]) -> bool:
    """Return True when a risk entry has at least one link field the app can use."""
    return bool(
        risk.get("linked_entity_id")
        or risk.get("involved_detection_ids")
        or risk.get("involved_track_ids")
        or risk.get("bbox")
        or risk.get("approximate_region")
    )


def _build_scene_risks(
    det_risks: Optional[list],
    vlm_draft: Optional[Dict[str, Any]],
    tracks: Optional[list],
) -> list:
    """Build the clean app-facing scene_risks list.

    Rules:
    * Active deterministic risks that are linkable are copied to scene_risks.
    * VLM draft risks are enriched with track bbox data where possible.
    * Vague / unlinked risks are excluded (not active scene_risks).
    """
    scene_risks: list = []

    # 1. Active linkable deterministic risks
    for risk in (det_risks or []):
        if risk.get("risk_state", "active") == "active" and _is_linkable(risk):
            sr = dict(risk)
            sr.setdefault("produced_by", "risk_engine")
            sr.setdefault("reasoner_model", "risk_engine.v1")
            sr.setdefault("reasoner_status", "rules_only")
            sr.setdefault("risk_reason", risk.get("reason", ""))
            sr.setdefault("evidence", risk.get("visual_evidence") or
                          ([risk["reason"]] if risk.get("reason") else []))
            scene_risks.append(sr)

    # 2. VLM draft risks -- enrich with track bbox if not already linked
    if vlm_draft and vlm_draft.get("risks"):
        track_map = {t.get("track_id"): t
                     for t in (tracks or []) if t.get("track_id")}
        vlm_model = vlm_draft.get("reasoner_model", "")
        for r in vlm_draft["risks"]:
            try:
                risk_dict = r.model_dump() if hasattr(r, "model_dump") else dict(r)
            except Exception:
                continue
            # Try to fill bbox from the first matching track
            if not risk_dict.get("bbox") and risk_dict.get("involved_track_ids"):
                for tid in risk_dict["involved_track_ids"]:
                    if tid in track_map and track_map[tid].get("bbox"):
                        risk_dict = dict(risk_dict)
                        risk_dict["bbox"] = track_map[tid]["bbox"]
                        break
            if not _is_linkable(risk_dict):
                continue  # exclude vague/unlinked VLM risk
            risk_dict = dict(risk_dict)
            risk_dict.setdefault("produced_by", "vlm_reasoner")
            risk_dict.setdefault("reasoner_model", vlm_model)
            risk_dict.setdefault("reasoner_status", "ready")
            risk_dict.setdefault("risk_reason", risk_dict.get("reason", ""))
            risk_dict.setdefault("evidence",
                                 risk_dict.get("visual_evidence") or [])
            scene_risks.append(risk_dict)

    return scene_risks


def _add_warning(resp: Dict[str, Any], key: str) -> None:
    """Append *key* to resp['warnings'] if not already present."""
    existing = resp.get("warnings") or []
    if isinstance(existing, str):
        existing = [existing]
    if key not in existing:
        resp["warnings"] = existing + [key]


# -- Detect endpoint ----------------------------------------------------------

@app.post("/detect")
async def detect(payload: Dict[str, Any]):
    """Run inference for the active backend.

    Returns entities (boxes) and poses (keypoints/skeleton). On any failure,
    returns a structured error with entities: [] and poses: [].
    """
    _t0 = time.perf_counter()
    runtime.inc("detect_requests_total")
    if not runtime.accepting():
        # Graceful shutdown: stop accepting new frames, let in-flight drain.
        return JSONResponse(
            {"error": "shutting_down", "entities": [], "poses": []},
            status_code=503,
        )
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

    # Input protection (B7): presence, base64, byte size, and decoded
    # megapixels (decompression-bomb guard). Structured 4xx -- never a 500.
    # Part 1: accept frame_b64 (app alias) OR image_b64; use the same frame
    # for detection, risk engine, VLM, and temporal reasoning.
    frame_b64 = payload.get("frame_b64") or payload.get("image_b64")
    ok, err, _info = guards.validate_image_b64(frame_b64)
    if not ok:
        runtime.inc("input_rejected_total", {"error": err})
        return JSONResponse(
            {"error": err, "entities": [], "poses": []},
            status_code=guards.status_for(err),
        )

    try:
        from vision_backend import run_inference
        class_filter = payload.get("classes")
        resp = run_inference(
            image_b64=frame_b64,
            class_filter=class_filter,
            payload=payload,  # Pass payload so config_resolver can extract conf/img_size
        )
        resp_dict = resp.model_dump()
        session_id = payload.get("session_id") or payload.get("sessionId")
        frame_id = payload.get("frame_id") or payload.get("frameId")
        # Additive deterministic risk-aware layer. No-op unless RISK_ENGINE_ENABLED.
        # Degradation ladder: a risk failure returns the normal detection result
        # with a `warning`, never a 500 (matches the backend_fallback behaviour).
        try:
            import risk
            resp_dict = risk.attach_risk(resp_dict, session_id=session_id, frame_id=frame_id)
        except Exception as rexc:  # noqa: BLE001 -- risk must never break detection
            log.warning("detect: risk layer failed (returning detection): %s", rexc)
            if not resp_dict.get("warning"):
                resp_dict["warning"] = "risk_engine_error: " + type(rexc).__name__ + ": " + str(rexc)
        # Event-driven VLM reasoning (AI draft only): NON-BLOCKING + rate-limited.
        # Triggered off the deterministic risk level; /detect never waits for the
        # VLM -- it attaches the most recent cached draft (if any) as `scene_risks`
        # (each requires_human_review=true) plus a `reasoner_status`, and returns.
        # Parts 2, 3, 4, 5: HSE intent forces the VLM trigger; reasoner_status is
        # normalized to a stable dict; scene_risks are built from deterministic +
        # VLM risks and ONLY include linkable (non-vague) items.
        _hse = wants_hse_reasoning(payload)
        _vlm_unavailable = False
        try:
            import risk.vlm_reasoner as vlm
            if vlm.enabled() and (resp_dict.get("schema_version") or _hse):
                # For HSE intent, force trigger regardless of current risk level.
                effective_level = (
                    vlm.trigger_level() if _hse
                    else resp_dict.get("highest_risk_level", "GREEN")
                )
                draft, raw_status = vlm.maybe_trigger(
                    session_id, frame_b64=frame_b64,
                    highest_level=effective_level,
                    deterministic_risks=resp_dict.get("risks", []),
                    entities=resp_dict.get("entities", []),
                    scene_graph=resp_dict.get("scene_graph", {}),
                    tracks=resp_dict.get("tracks", []), frame_id=frame_id,
                )
                resp_dict["reasoner_status"] = _normalize_reasoner_status(
                    raw_status, vlm._model_id())
                resp_dict["scene_risks"] = _build_scene_risks(
                    resp_dict.get("risks", []), draft, resp_dict.get("tracks", []))
            elif _hse:
                # HSE mode requested but VLM is disabled -- degrade gracefully.
                _vlm_unavailable = True
                resp_dict["reasoner_status"] = _normalize_reasoner_status("unavailable")
                resp_dict["scene_risks"] = _build_scene_risks(
                    resp_dict.get("risks", []), None, resp_dict.get("tracks", []))
                _add_warning(resp_dict, "qwen_unavailable")
        except Exception as vexc:  # noqa: BLE001 -- VLM must never break detection
            log.warning("detect: vlm trigger failed: %s", vexc)
            if _hse:
                _vlm_unavailable = True
                resp_dict["reasoner_status"] = _normalize_reasoner_status("error")
                resp_dict["scene_risks"] = _build_scene_risks(
                    resp_dict.get("risks", []), None, resp_dict.get("tracks", []))
                _add_warning(resp_dict, "qwen_unavailable")
        # Event-triggered temporal perception (PR: single-worker GPU+CPU). ADDITIVE
        # + NON-BLOCKING: folds the frame into per-session memory, adds deterministic
        # object-near-edge risk, and (rarely, rate-limited) kicks an async VLM job
        # for scene_context + perception corrections. /detect NEVER waits for it;
        # it attaches the most recent cached blocks. No-op unless TEMPORAL_REASONING_ENABLED.
        try:
            import temporal_reasoning
            if temporal_reasoning.enabled():
                resp_dict = temporal_reasoning.attach_temporal(
                    resp_dict, session_id=session_id, frame_id=frame_id,
                    frame_b64=frame_b64, payload=payload)
        except Exception as texc:  # noqa: BLE001 -- temporal must never break detection
            log.warning("detect: temporal layer failed: %s", texc)
        # Degradation ladder (B4): surface degraded/degradation_mode + metrics.
        # VLM unavailability in HSE mode is also surfaced as degraded.
        rmeta = resp_dict.get("risk_engine") or {}
        mode = "no_risk" if isinstance(rmeta, dict) and rmeta.get("degraded") else "full"
        if _vlm_unavailable and _hse:
            mode = mode if mode == "no_risk" else "detect_only"
        resp_dict["degraded"] = mode != "full"
        resp_dict["degradation_mode"] = mode
        runtime.set_degradation(mode)
        _detect_ms = (time.perf_counter() - _t0) * 1000.0
        runtime.observe_latency("detect_latency_ms", _detect_ms)
        runtime.observe_latency("gpu_detect_latency_ms", _detect_ms)
        if resp_dict.get("schema_version"):
            runtime.inc("risk_level_total", {"level": resp_dict.get("highest_risk_level", "GREEN")})
        # reasoner_status is a string (legacy vlm trigger) or a dict (temporal
        # layer, when enabled). Use the state label either way (dict -> .state).
        _rs = resp_dict.get("reasoner_status")
        _rs_label = _rs.get("state") if isinstance(_rs, dict) else _rs
        if _rs_label:
            runtime.inc("reasoner_status_total", {"status": str(_rs_label)})
        runtime.log_event("detect", session_id=session_id, frame_id=frame_id,
                          backend=resp_dict.get("backend"),
                          entities=len(resp_dict.get("entities", []) or []),
                          highest_risk_level=resp_dict.get("highest_risk_level"),
                          degradation_mode=mode,
                          reasoner_status=_rs_label)
        return JSONResponse(resp_dict)
    except Exception as exc:
        runtime.inc("detect_errors_total")
        runtime.log_event("detect_failed", level=logging.ERROR,
                          error=type(exc).__name__ + ": " + str(exc))
        log.exception("detect failed: %s", exc)
        return JSONResponse(
            {"error": "inference_failed: " + type(exc).__name__ + ": " + str(exc),
             "entities": [], "poses": [], "backend": _active_backend(),
             "degraded": True, "degradation_mode": "detect_only"},
            status_code=500,
        )

# -- Reasoning endpoints (event-driven; AI draft only; never the safety authority) --
#
# POST /reason runs the REAL Qwen-VL (or DeepSeek-VL2 / mock) reasoner with a
# hard timeout and returns strict JSON. The deterministic engine remains the
# safety signal; VLM output is always produced_by="vlm_reasoner",
# requires_human_review=true, should_alert=false. The reasoner is event-driven
# (called after a deterministic candidate exists), NEVER per-frame, and degrades
# to a clear reasoner_status when disabled/unavailable/slow -- it can never stall
# /detect. POST /scan is the optional open-vocabulary GroundingDINO scanner
# (disabled by default; candidate-only output). Both are guarded so a reasoning
# problem can never break server boot or the live detection path.

@app.post("/reason")
async def reason(payload: Dict[str, Any]):
    """Event-driven VLM reasoning over a deterministic candidate. Strict JSON."""
    if not runtime.accepting():
        return JSONResponse({"schema_version": "reason.v1", "produced_by": "vlm_reasoner",
                             "reasoner_status": "shutting_down", "requires_human_review": True,
                             "should_alert": False, "risks": []}, status_code=503)
    # If a frame is supplied, it must pass the input guards (megapixel/bomb).
    frame = payload.get("frame_b64") or payload.get("image_b64")
    if frame:
        ok, err, _ = guards.validate_image_b64(frame)
        if not ok:
            runtime.inc("input_rejected_total", {"error": err})
            return JSONResponse({"schema_version": "reason.v1", "produced_by": "vlm_reasoner",
                                 "reasoner_status": "error", "error": err,
                                 "requires_human_review": True, "should_alert": False,
                                 "risks": []}, status_code=guards.status_for(err))
    try:
        import risk.vlm_reasoner as vlm
        result = await vlm.reason_async(payload)
        runtime.inc("reasoner_status_total", {"status": result.get("reasoner_status", "ok")})
        return JSONResponse(result)
    except Exception as exc:  # noqa: BLE001 -- never 500 the reasoning endpoint
        log.warning("reason failed: %s", exc)
        return JSONResponse({
            "schema_version": "reason.v1", "produced_by": "vlm_reasoner",
            "reasoner_status": "error", "requires_human_review": True,
            "should_alert": False, "risks": [], "uncertain_items": [],
            "error": type(exc).__name__ + ": " + str(exc),
        })

@app.post("/scan")
async def scan(payload: Dict[str, Any]):
    """Optional open-vocabulary (GroundingDINO) scan. Candidate-only; human review."""
    frame = payload.get("frame_b64") or payload.get("image_b64")
    ok, err, _ = guards.validate_image_b64(frame)
    if not ok:
        runtime.inc("input_rejected_total", {"error": err})
        return JSONResponse({"schema_version": "openvocab.v1", "produced_by": "open_vocab_scanner",
                             "source_model": "GroundingDINO", "status": "error", "error": err,
                             "candidate_only": True, "requires_human_review": True,
                             "candidates": []}, status_code=guards.status_for(err))
    try:
        import risk.open_vocab_scanner as ovs
        result = await asyncio.to_thread(
            ovs.scan,
            payload.get("frame_b64") or payload.get("image_b64"),
            prompt=payload.get("prompt"),
            session_id=payload.get("session_id") or payload.get("sessionId"),
            frame_id=payload.get("frame_id") or payload.get("frameId"),
            entities=payload.get("entities"),
            force=bool(payload.get("force", False)),
        )
        return JSONResponse(result)
    except Exception as exc:  # noqa: BLE001
        log.warning("scan failed: %s", exc)
        return JSONResponse({
            "schema_version": "openvocab.v1", "produced_by": "open_vocab_scanner",
            "source_model": "GroundingDINO", "status": "error", "candidate_only": True,
            "requires_human_review": True, "candidates": [],
            "error": type(exc).__name__ + ": " + str(exc),
        })

# -- Build Mode (lightweight blueprint processing; CPU-only, additive) --------
#
# Fully separate from EdgeCrafter / detect: Build Mode never loads a model,
# never triggers warmup, and never touches the GPU. The CPU image work
# (Pillow/OpenCV/NumPy) runs in a worker thread (asyncio.to_thread inside
# build_blueprint) so it cannot block /detect or /health. State is lightweight
# in-memory JSON keyframes only -- no images, no video -- with TTL + frame caps.
# build_blueprint is imported lazily and every handler is guarded so Build Mode
# can never break server boot or the existing routes above.

def _build_error_response(exc):
    from build_schema import BuildError
    if isinstance(exc, BuildError):
        return JSONResponse({"ok": False, "error": exc.code}, status_code=exc.status)
    log.exception("build: unexpected error: %s", exc)
    return JSONResponse({"ok": False, "error": "build_failed"}, status_code=500)

@app.post("/build/session/start")
async def build_session_start(payload: Dict[str, Any]):
    try:
        import build_blueprint
        return JSONResponse(build_blueprint.start_session(payload))
    except Exception as exc:  # noqa: BLE001
        return _build_error_response(exc)

@app.post("/build/session/lock")
async def build_session_lock(payload: Dict[str, Any]):
    try:
        import build_blueprint
        return JSONResponse(build_blueprint.lock_session(payload))
    except Exception as exc:  # noqa: BLE001
        return _build_error_response(exc)

@app.post("/build/session/frame")
async def build_session_frame(payload: Dict[str, Any]):
    try:
        import build_blueprint
        return JSONResponse(await build_blueprint.process_frame_async(payload))
    except Exception as exc:  # noqa: BLE001
        return _build_error_response(exc)

@app.post("/build/session/finish")
async def build_session_finish(payload: Dict[str, Any]):
    try:
        import build_blueprint
        return JSONResponse(build_blueprint.finish_session(payload))
    except Exception as exc:  # noqa: BLE001
        return _build_error_response(exc)

@app.get("/build/session/{session_id}/replay")
async def build_session_replay(session_id: str):
    try:
        import build_blueprint
        return JSONResponse(build_blueprint.get_replay(session_id))
    except Exception as exc:  # noqa: BLE001
        return _build_error_response(exc)

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
    # Shared-secret auth on connect (B7), except in compat/test mode.
    if not security.check_ws(websocket):
        try:
            await websocket.close(code=1008)  # policy violation
        except Exception:
            pass
        return
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
    """Active tasks for the SERVING backend, with torch-free env fallbacks."""
    try:
        from vision_backend import serving_backend
        backend = serving_backend()
    except Exception:
        backend = _active_backend()
    if backend == "deimv2":
        return ["det"]
    if backend in ("yolo26", "ultralytics"):
        try:
            import yolo26_loader
            return yolo26_loader.mode_tasks("live")
        except Exception:
            raw = os.getenv("YOLO26_LIVE_TASKS") or os.getenv("YOLO26_TASKS") or "det"
            out = [t.strip().lower() for t in raw.split(",")
                   if t.strip().lower() in ("det", "seg", "pose")]
            return out or ["det"]
    try:
        import edgecrafter_loader as ec
        return list(ec._STATE.tasks) if ec._STATE.tasks else ec.parse_tasks()
    except Exception:
        raw = os.getenv("EDGECRAFTER_TASKS", "det,pose")
        out = [t.strip().lower() for t in raw.split(",")
               if t.strip().lower() in ("det", "pose")]
        return out or ["det"]

def _ws_stream_config():
    """Active-backend conf/img_size for /ws/vision (A1.5; post-fallback aware).

    Resolves the SERVING backend's config via config_resolver so streaming uses
    the same generic YOLO_* / legacy YOLO26_* / EDGECRAFTER_* values as /detect,
    never a stale EdgeCrafter-derived default.
    """
    try:
        from vision_backend import serving_backend
        from config_resolver import resolve_effective_inference_config
        b = serving_backend()
        rb = "yolo26" if b in ("yolo26", "ultralytics") else b
        cfg = resolve_effective_inference_config(rb, {})
        return {"conf": cfg.conf, "img_size": cfg.img_size}
    except Exception:  # noqa: BLE001
        return {"conf": 0.25, "img_size": 640}

def _ws_gpu_device():
    """GPU device name if CUDA is available, else None (never raises)."""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception:
        pass
    return None

def _ws_attach_risk(message, session_id=None, frame_id=None):
    """Merge the deterministic risk block into a /ws/vision 'vision' message.

    No-op unless RISK_ENGINE_ENABLED; never raises (returns the message
    unchanged on any failure so streaming is never broken by the risk layer).
    The stream's session key is the camera_id, so per-camera tracker state
    stays isolated (B1).
    """
    try:
        import risk
        return risk.attach_risk(message, session_id=session_id, frame_id=frame_id)
    except Exception as exc:  # noqa: BLE001
        log.warning("ws/vision: risk layer failed: %s", exc)
        return message

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
        get_config=_ws_stream_config,
        risk_hook=_ws_attach_risk,
        authorize_ws=lambda ws: security.check_ws(ws),
        is_accepting=runtime.accepting,
    )
    _log_startup("ws/vision: streaming route + /debug/stream registered")
except Exception as exc:  # noqa: BLE001 -- streaming must never break server boot
    _log_startup("ws/vision: registration FAILED -- " + type(exc).__name__ + ": " + str(exc))
    log.warning("ws/vision registration failed: %s", exc)

# -- CPU agentic layer (/agent/*) ---------------------------------------------
#
# Mounted into THIS app + process (single RunPod worker), but on a SEPARATE
# bounded job queue with its own concurrency limit, so CPU agent work can never
# block /detect. CPU-only (no torch/cv2/ultralytics/transformers). Routes always
# exist; behaviour is gated by AGENTIC_CPU_ENABLED. Guarded so an agent-layer
# import problem can never break server boot or the vision routes above.
try:
    import agentic_cpu
    app.include_router(agentic_cpu.get_router(), prefix="/agent", tags=["agentic_cpu"])
    _log_startup("agentic_cpu: /agent/* routes registered (enabled="
                 + str(agentic_cpu.enabled()) + ", mode=" + agentic_cpu.config.mode() + ")")
except Exception as exc:  # noqa: BLE001 -- agent layer must never break server boot
    _log_startup("agentic_cpu: registration FAILED -- " + type(exc).__name__ + ": " + str(exc))
    log.warning("agentic_cpu registration failed: %s", exc)

# -- Entrypoint ---------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level=os.getenv("UVICORN_LOG_LEVEL", "info"))
