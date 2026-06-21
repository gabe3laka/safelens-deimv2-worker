# syntax=docker/dockerfile:1
# safelens-deimv2-worker/Dockerfile
# Builds the SafeLens vision live-server worker for RunPod load-balancing endpoints.
#
# Default backend:  YOLO26 (Ultralytics; boxes + poses, optional seg).
# Fallback backend: EdgeCrafter (ECDet-S boxes + optional ECPose-S poses),
#                   served automatically when YOLO26 fails to load and
#                   AUTO_BACKEND_FALLBACK=true.
# Legacy debug:     DEIMv2 (VISION_BACKEND=deimv2).
#
# Architecture: long-running FastAPI/uvicorn server (adapted from Kingo333/fluxrt-serverless).
# YOLO26 weights are best-effort pre-baked into /app/models/yolo26 and otherwise
# cached at runtime in YOLO26_CACHE_DIR (RunPod volume); EdgeCrafter checkpoints
# download at runtime into EDGECRAFTER_CACHE_DIR; DEIMv2 weights come from HF Hub.
#
# RunPod endpoint type: HTTP (load-balancing), not serverless queue.
# Health probe: GET /health or GET /ping (returns immediately, no model required).
#
# GPU compatibility: target Ampere/Ada GPUs (RTX 3090, L4, RTX A5000).
# AVOID Blackwell (PRO 6000 MIG, B200) -- this CUDA 12.4 / torch 2.6 image is not
# built for Blackwell (sm_100/sm_120). Pin RunPod worker GPU types accordingly.

# EdgeCrafter requires torch >= 2.6.0, so we use the 2.6.0 / CUDA 12.4 base.
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

WORKDIR /app

# Build provenance (B6): the CI build passes --build-arg BUILD_SHA=<git sha>;
# surfaced in /debug/state, /ready, /metrics, and every structured log line.
ARG BUILD_SHA=unknown
ENV BUILD_SHA=${BUILD_SHA}

# System dependencies (git for repo clones, libgl for opencv).
RUN apt-get update && apt-get install -y \
    git wget curl libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Python worker dependencies (fastapi + uvicorn for live-server mode).
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# ---- Clone EdgeCrafter (default backend) ------------------------------------
# The ECDet/ECPose architectures live in the upstream repo; the worker imports
# their engine.* packages (one per subtree) via edgecrafter_loader.py.
RUN git clone --depth=1 https://github.com/Intellindust-AI-Lab/EdgeCrafter.git /opt/EdgeCrafter

# Install EdgeCrafter's own requirements (numpy/pyyaml/opencv/etc.).
RUN if [ -f /opt/EdgeCrafter/requirements.txt ]; then \
        pip install --no-cache-dir -r /opt/EdgeCrafter/requirements.txt; \
    fi

# ---- Clone DEIMv2 (legacy fallback backend) ---------------------------------
ARG DEIMV2_REPO_URL=https://github.com/Intellindust-AI-Lab/DEIMv2.git
ARG DEIMV2_BRANCH=main
RUN git clone --depth 1 --branch ${DEIMV2_BRANCH} ${DEIMV2_REPO_URL} /opt/DEIMv2
RUN if [ -f /opt/DEIMv2/requirements.txt ]; then \
        pip install --no-cache-dir -r /opt/DEIMv2/requirements.txt; \
    fi

# Re-assert SafeLens pinned runtime deps AFTER upstream requirements, which can
# downgrade/overwrite shared packages. The final reinstall wins.
RUN pip install --no-cache-dir --upgrade \
    "huggingface-hub>=0.26.0" \
    "safetensors>=0.4.5" \
    "timm>=1.0.11" \
    "pyyaml>=6.0" \
    "opencv-python-headless>=4.10.0.84"

# ---- Pin huggingface-hub below 1.0 (MUST be the last hub install) -----------
# EdgeCrafter's engine.core.YAMLConfig -> calflops -> transformers requires
# huggingface-hub<1.0. The ">=0.26.0" --upgrade above resolves to 1.18.0, which
# transformers rejects at import time and breaks EdgeCrafter warmup. Force the
# compatible range, then verify at build time so a bad resolve fails the build.
RUN python -m pip install --no-cache-dir --force-reinstall "huggingface-hub>=0.26.0,<1.0"
RUN python - <<'PY'
import huggingface_hub
from packaging.version import Version
v = Version(huggingface_hub.__version__)
assert Version("0.26.0") <= v < Version("1.0"), f"bad huggingface-hub version: {v}"
print("huggingface-hub OK:", v)
PY

# ---- YOLO26 (Ultralytics) -- default vision backend ---------------------------
# Install pattern encodes the fb426cb / ddb143c lessons:
#  * record the working baseline FIRST, then assert it is byte-for-byte
#    unchanged after the install (never assert a hardcoded torch version -- the
#    working stack is torch 2.5.1+cu124 after the upstream requirement installs)
#  * NEVER uninstall/reinstall OpenCV (opencv-python and -headless share cv2/)
# The fb426cb build log proved a full `pip install ultralytics` leaves
# torch/cv2/numpy/huggingface-hub untouched (it only adds matplotlib/polars/
# thop/etc.), and YOLO needs those extras importable at runtime.
RUN python - <<'PY'
import json
import torch, torchvision, cv2, numpy, huggingface_hub
baseline = {
    "torch": torch.__version__,
    "torchvision": torchvision.__version__,
    "cv2": cv2.__version__,
    "numpy": numpy.__version__,
    "huggingface_hub": huggingface_hub.__version__,
}
with open("/opt/yolo_dep_baseline.json", "w") as fh:
    json.dump(baseline, fh)
print("dependency baseline recorded:", baseline)
PY
RUN pip install --no-cache-dir ultralytics
RUN python - <<'PY'
import json, sys
import torch, torchvision, cv2, numpy, huggingface_hub
baseline = json.load(open("/opt/yolo_dep_baseline.json"))
current = {
    "torch": torch.__version__,
    "torchvision": torchvision.__version__,
    "cv2": cv2.__version__,
    "numpy": numpy.__version__,
    "huggingface_hub": huggingface_hub.__version__,
}
assert current == baseline, f"ultralytics changed the stack: {baseline} -> {current}"
import ultralytics  # noqa: F401
# EdgeCrafter fallback must still import after the YOLO install.
sys.path.insert(0, "/opt/EdgeCrafter/ecdetseg")
from engine.core import YAMLConfig  # noqa: F401
print("stack preserved + ultralytics", ultralytics.__version__,
      "+ EdgeCrafter engine import OK")
PY

# Hardening (B10): NO model weights are baked into the image and NONE are
# downloaded at build time -- not YOLO, EdgeCrafter, DEIMv2, or GroundingDINO.
# Gemini is API-only (no weights). Weights resolve at RUNTIME only, from
# YOLO26_CACHE_DIR / EDGECRAFTER_CACHE_DIR / the HF cache on /runpod-volume, or
# an approved registry. We still create the cache dir so the runtime can write
# into it (auto-download / volume mount). Operators may pre-populate the volume
# for air-gapped/no-egress deployments (see docs/runbook.md).
RUN mkdir -p /app/models/yolo26

# ---- Optional SAM2 runtime support ------------------------------------------
# SAM2 code paths exist in build_segmentation.py, but dedicated SAM2 weights /
# packages are NOT part of this image's contract; Build Mode defaults to the
# fallback contour pipeline (BUILD_SEGMENTATION_BACKEND=fallback below).

# Copy worker code
COPY schema.py /app/schema.py
COPY edgecrafter_loader.py /app/edgecrafter_loader.py
COPY yolo26_loader.py /app/yolo26_loader.py
COPY vision_backend.py /app/vision_backend.py
COPY config_resolver.py /app/config_resolver.py
COPY ultralytics_loader.py /app/ultralytics_loader.py
COPY model_registry.example.json /app/model_registry.example.json
COPY deimv2_infer.py /app/deimv2_infer.py
COPY official_deimv2_loader.py /app/official_deimv2_loader.py
COPY server.py /app/server.py
COPY ws_vision.py /app/ws_vision.py
COPY bootstrap.py /app/bootstrap.py
COPY handler.py /app/handler.py
COPY worker_guards.py /app/worker_guards.py
COPY worker_runtime.py /app/worker_runtime.py
COPY worker_security.py /app/worker_security.py

# Build Mode (Build/Plan v2) modules -- required by server.py's /build routes.
COPY build_schema.py /app/build_schema.py
COPY build_blueprint.py /app/build_blueprint.py
COPY build_segmentation.py /app/build_segmentation.py
COPY plan_context.py /app/plan_context.py

# Risk-aware perception (deterministic engine + per-session tracking) + the
# CPU-only validation harness. No weights/datasets -- pure-Python rules + a JSON
# matrix profile. Additive and gated by RISK_ENGINE_ENABLED (default off).
COPY risk /app/risk
COPY validation /app/validation

# Single-worker GPU+CPU layers (additive, import-light, no weights):
#  * shared/             -- cross-layer pydantic schemas, prompts, wire contracts
#  * gpu_vision/         -- bounded GPU reasoner concurrency + GPU-pressure signal
#  * temporal_reasoning/ -- event-triggered temporal VLM perception (non-blocking)
#  * agentic_cpu/        -- CPU-only agentic orchestration mounted at /agent/*
# agentic_cpu imports NO torch/cv2/ultralytics/transformers (CI import guard).
COPY shared /app/shared
COPY gpu_vision /app/gpu_vision
COPY temporal_reasoning /app/temporal_reasoning
COPY agentic_cpu /app/agentic_cpu

# Worker code + upstream engine packages on PYTHONPATH. The EdgeCrafter ecdetseg
# and ecpose subtrees each ship their own engine package; edgecrafter_loader.py
# manages which one is active at import time, so we only add /app + /opt/DEIMv2
# here and let the loader insert the EdgeCrafter subtrees dynamically.
ENV PYTHONPATH="/app:/opt/DEIMv2:${PYTHONPATH}"

# ------- RunPod HTTP endpoint configuration ----------------------------------
ENV PORT="8000"
ENV UVICORN_LOG_LEVEL="info"
ENV SKIP_WARMUP="false"
ENV AUTO_WARMUP="true"
ENV WARMUP_TIMEOUT_S="600"
ENV STARTUP_LOG="/tmp/safelens_startup.log"

# ------- Vision backend selection --------------------------------------------
# yolo26 (default) | edgecrafter (fallback) | deimv2 (legacy debug). If the
# requested backend fails to LOAD and AUTO_BACKEND_FALLBACK=true, the worker
# automatically serves FALLBACK_VISION_BACKEND instead (visible in /debug/state
# and the /detect `warning` field).
ENV FALLBACK_VISION_BACKEND="edgecrafter"
ENV AUTO_BACKEND_FALLBACK="true"

# ------- YOLO26 configuration (task-based modes) -------------------------------
# Live /detect runs DETECTION ONLY for speed; segmentation runs only on
# selected Build/Plan crops (lazy-loaded); pose is opt-in (YOLO26_POSE_ENABLED
# or an explicit 'pose' in a task list).
ENV YOLO26_DET_MODEL_ID="yolo26n.pt"
ENV YOLO26_SEG_MODEL_ID="yolo26n-seg.pt"
ENV YOLO26_POSE_MODEL_ID="yolo26n-pose.pt"
ENV YOLO26_LIVE_TASKS="det"
ENV YOLO26_BUILD_TASKS="det,seg"
ENV YOLO26_PLAN_TASKS="det,seg"
ENV YOLO26_POSE_ENABLED="false"
ENV YOLO26_SEG_EVERY_N="3"
ENV YOLO26_DEVICE="cuda"
ENV YOLO26_IMG_SIZE="640"
ENV YOLO26_CONF="0.25"
ENV YOLO26_IOU="0.50"
ENV YOLO26_MAX_DETECTIONS="170"
ENV YOLO26_CACHE_DIR="/runpod-volume/models/yolo26"
# Ultralytics container hardening: writable config dir; never pip at runtime.
ENV YOLO_CONFIG_DIR="/tmp/Ultralytics"
ENV YOLO_AUTOINSTALL="false"

# ------- Generic detector config (A1) -----------------------------------------
# Generic YOLO_* names take PRECEDENCE over the legacy YOLO26_* above when set,
# so a stronger detector profile can be tested WITHOUT editing the image:
#   VISION_BACKEND=ultralytics
#   YOLO_DET_MODEL_ID=yolo11s.pt  YOLO_IMG_SIZE=960  YOLO_CONF=0.10
#   YOLO_IOU=0.60  YOLO_MAX_DETECTIONS=300
# Left UNSET here on purpose: yolo11s.pt is AGPL -- verify commercial rights
# before making it the production default (do NOT silently switch). Commercial
# direction: DEIM/DEIMv2 or RT-DETR (Apache-2.0). See model_registry.example.json.
# Weights are NEVER baked here; they resolve at runtime from cache/volume/registry.
ENV MODEL_REGISTRY_PATH="/app/model_registry.example.json"

# ------- (legacy comment) ------------------------------------------------------
# edgecrafter (default) | deimv2 (legacy fallback)
ENV VISION_BACKEND="yolo26"

# ------- EdgeCrafter configuration -------------------------------------------
ENV EDGECRAFTER_TASKS="det,pose"
ENV EDGECRAFTER_DEVICE="cuda"
ENV EDGECRAFTER_IMG_SIZE="640"
ENV EDGECRAFTER_CONF="0.25"
ENV EDGECRAFTER_REPO_DIR="/opt/EdgeCrafter"
ENV EDGECRAFTER_CACHE_DIR="/runpod-volume/models/edgecrafter"
ENV EDGECRAFTER_DET_CONFIG="/opt/EdgeCrafter/ecdetseg/configs/ecdet/ecdet_s.yml"
ENV EDGECRAFTER_DET_CHECKPOINT_URL="https://github.com/capsule2077/edgecrafter/releases/download/edgecrafterv1/ecdet_s.pth"
ENV EDGECRAFTER_DET_CHECKPOINT_PATH="/runpod-volume/models/edgecrafter/ecdet_s.pth"
ENV EDGECRAFTER_POSE_CONFIG="/opt/EdgeCrafter/ecpose/configs/ecpose/ecpose_s_coco.yml"
ENV EDGECRAFTER_POSE_CHECKPOINT_URL="https://github.com/capsule2077/edgecrafter/releases/download/edgecrafterv1/ecpose_s.pth"
ENV EDGECRAFTER_POSE_CHECKPOINT_PATH="/runpod-volume/models/edgecrafter/ecpose_s.pth"

# ------- Build Mode (lightweight blueprint processing; CPU-only) -------------
# Initial deployment uses the fallback contour pipeline so the image works
# WITHOUT SAM2 installed. Switch BUILD_SEGMENTATION_BACKEND to "sam2" (and set
# BUILD_SAM2_DEVICE / checkpoint) only after the fallback is proven stable.
ENV BUILD_SEGMENTATION_BACKEND="fallback"
ENV BUILD_MASK_OUTPUT="contour"
ENV BUILD_SEGMENT_ON_EXTRACT="true"
ENV BUILD_SEGMENT_EVERY_N="3"
ENV BUILD_SAM2_DEVICE="cuda"
ENV BUILD_SAM2_WEIGHTS="/app/models/sam2_b.pt"

# ------- Plan Mode selected-crop context -------------------------------------
# Rule-based crop context + virtualBlueprintPoints are ON by default and need no
# extra models. Depth / open-vocab / known-part-pose / assembly-state are
# OPTIONAL, DISABLED safe stubs (no extra deps in this image; they degrade to a
# clear "unavailable" signal if enabled without a backend). Point-E is never run.
ENV PLAN_CONTEXT_ENABLED="true"
ENV PLAN_DEPTH_ENABLED="false"
ENV PLAN_DEPTH_BACKEND="none"
ENV PLAN_DEPTH_SAMPLE_POINTS="120"
ENV PLAN_DEPTH_MAX_RES="384"
ENV PLAN_DEPTH_EVERY_N="3"
ENV PLAN_DEPTH_CACHE_TTL_MS="1500"
ENV PLAN_OPEN_VOCAB_ENABLED="false"
ENV PLAN_OPEN_VOCAB_BACKEND="none"
ENV PLAN_OPEN_VOCAB_PROMPTS="pcb board,cable,connector,screw,tool,battery,wire,arduino board"
ENV PLAN_KNOWN_PART_POSE_ENABLED="false"
ENV PLAN_KNOWN_PART_POSE_BACKEND="none"
ENV PLAN_ASSEMBLY_STATE_ENABLED="false"
ENV DEPTH_MODEL_ID="depth-anything-v2-small"
ENV DEPTH_DEVICE="cuda"
ENV DEPTH_CACHE_DIR="/runpod-volume/models/depth"

# ------- Risk-aware perception (deterministic engine + tracking) -------------
# Additive and OFF by default: when RISK_ENGINE_ENABLED=false the /detect and
# /ws/vision responses are byte-for-byte the legacy shape. When enabled, the
# deterministic engine (the safety signal) adds tracks/scene_graph/risks +
# schema_version. Per-session tracker state is keyed by session_id/camera_id
# with TTL eviction (Build Mode pattern). No weights, no GPU, no VLM here --
# the event-driven Gemini reasoner / GroundingDINO scanner are additive.
ENV RISK_ENGINE_ENABLED="false"
ENV RISK_TRACKING_ENABLED="true"
ENV RISK_SCENE_GRAPH_ENABLED="true"
ENV RISK_PROVENANCE_ENABLED="true"
ENV RISK_MODEL_VERSION="risk_engine.v1"
ENV RISK_MATRIX_PROFILE="/app/risk/risk_matrix_profile.json"
ENV RISK_NEAR_THRESHOLD="0.12"
ENV RISK_EDGE_THRESHOLD="0.04"
ENV TRACK_IOU_MATCH="0.3"
ENV TRACK_MAX_AGE_FRAMES="30"
ENV TRACK_MAX_PER_SESSION="300"
ENV SESSION_TTL_MS="30000"
ENV SESSION_MAX_ACTIVE="64"
# Privacy: blur persons/faces before any frame is persisted/sent to a VLM.
# OFF until the (later) VLM/evidence path exists; the deterministic engine
# needs no imagery. No emotion/biometric inference -- hazards/conditions only.
ENV PRIVACY_BLUR_ENABLED="false"
ENV PRIVACY_BLUR_RADIUS="24"
# Validation gate (validation/run_validation.py): min critical-hazard recall.
ENV VALIDATION_MIN_RECALL_CRITICAL="0.90"

# ------- Event-driven VLM reasoning (Gemini API) ----------------------------
# REAL adapter, but OFF by default and NEVER per-frame: /reason runs it on
# demand and /detect triggers it asynchronously (rate-limited, above
# REASONER_TRIGGER_LEVEL) and never waits for it. VLM output is an AI DRAFT
# only (produced_by=vlm_reasoner, requires_human_review=true, should_alert=false)
# -- it never becomes the safety authority. The deterministic engine remains the
# signal. REASONER_MODE=mock gives a CPU, weight-free contract for app integration.
# GEMINI_API_KEY is intentionally NOT set here (never bake secrets).
# Supported live modes: gemini | mock | disabled.
# Removed modes (degrade to unavailable): qwen_vl | deepseek_vl2.
ENV VLM_REASONER_ENABLED="true"
ENV REASONER_MODE="gemini"
# QWEN_VL_DEEP_* kept as inert future/deep/offline placeholders (not used live).
ENV QWEN_VL_DEEP_MODEL_ID="Qwen/Qwen2.5-VL-7B-Instruct"
ENV QWEN_VL_DEEP_ENABLED="false"
ENV REASONER_DEVICE="cuda"
ENV REASONER_MAX_IMAGE_SIDE="512"
ENV REASONER_TIMEOUT_MS="2500"
ENV REASONER_MIN_INTERVAL_MS="1500"
ENV REASONER_CACHE_TTL_MS="10000"
ENV REASONER_TRIGGER_LEVEL="YELLOW"
ENV REASONER_MAX_WORKERS="1"
ENV REASONER_MAX_SESSIONS="64"
ENV REASONER_MATCH_IOU_MIN="0.20"
ENV REASONER_MATCH_CENTER_DIST_MAX="0.20"
ENV REASONER_LINKED_RISK_TTL_MS="8000"
ENV REASONER_UNMATCHED_CANDIDATE_TTL_MS="5000"
ENV REASONER_LATEST_WINS="true"
ENV REASONER_PENDING_FRAME_MAX_AGE_MS="2500"
# Gemini-specific knobs (set GEMINI_API_KEY at deploy time, never here).
ENV GEMINI_MODEL_ID="gemini-2.5-flash"
ENV GEMINI_TIMEOUT_MS="12000"
ENV GEMINI_MAX_OUTPUT_TOKENS="512"
ENV GEMINI_TEMPERATURE="0"
ENV GEMINI_MAX_IMAGE_SIDE="512"
ENV GEMINI_MAX_DETECTED_LABELS="20"
ENV GEMINI_REQUEST_RETRIES="1"

# ------- Open-vocabulary scanner (GroundingDINO) -----------------------------
# Optional, OFF by default, NEVER per-frame. Candidate-only output (requires
# human review); never triggers official HSE alerts. Weights resolve at runtime
# (NOT baked, NOT downloaded at build).
ENV OPEN_VOCAB_SCANNER_ENABLED="false"
ENV OPEN_VOCAB_SCANNER_MODE="grounding_dino"
ENV GROUNDING_DINO_MODEL_ID="IDEA-Research/grounding-dino-tiny"
ENV GROUNDING_DINO_BOX_THRESHOLD="0.35"
ENV GROUNDING_DINO_TEXT_THRESHOLD="0.25"
ENV OPEN_VOCAB_SCAN_INTERVAL_MS="30000"
ENV GROUNDING_DINO_CACHE_DIR="/runpod-volume/models/groundingdino"

# ------- Worker hardening (auth / input guards / shutdown) -------------------
# WORKER_SHARED_SECRET is intentionally NOT set here (never bake secrets). Set it
# at deploy time so the worker rejects any request without the proxy's secret on
# every route except /health and /ping. Unset = compatibility/testing mode.
# (placeholder documented, not a default secret): WORKER_SHARED_SECRET=""
ENV WORKER_AUTH_HEADER="x-worker-secret"
ENV MAX_REQUEST_BYTES="10000000"
ENV MAX_IMAGE_MEGAPIXELS="16"
ENV GRACEFUL_DRAIN_MS="1500"

# ------- Event-triggered temporal VLM perception (GPU side) ------------------
# Additive. The detector runs every frame; the VLM is event-triggered + NON-
# BLOCKING (/detect never waits). Perception corrections are advisory (no human
# approval); safety/compliance drafts require human review. No weights baked.
ENV TEMPORAL_REASONING_ENABLED="true"
ENV TEMPORAL_MEMORY_WINDOW_FRAMES="45"
ENV TEMPORAL_MEMORY_TTL_MS="30000"
ENV TEMPORAL_MAX_ACTIVE_SESSIONS="64"
ENV TEMPORAL_STORE_KEYFRAMES="false"
ENV TEMPORAL_REASONING_TRIGGER_MIN_INTERVAL_MS="1500"
ENV TEMPORAL_REASONING_MAX_ASYNC_JOBS="1"
ENV TEMPORAL_LABEL_FLIP_WINDOW_FRAMES="8"
ENV SCENE_CONTEXT_ENABLED="true"
ENV SCENE_CONTEXT_REFRESH_MS="2000"
ENV SCENE_HINT_ENABLED="true"
ENV CONTEXTUAL_SUPPRESSION_ENABLED="true"
ENV SEMANTIC_CORRECTION_ENABLED="true"
ENV SEMANTIC_CORRECTION_LOW_CONF_THRESHOLD="0.35"
ENV OBJECT_EDGE_RISK_ENABLED="true"
ENV OBJECT_EDGE_DISTANCE_THRESHOLD="0.10"
ENV OBJECT_EDGE_HISTORY_FRAMES="6"
ENV REASONER_RESULT_STALE_MS="8000"
ENV REASONER_HUMAN_REVIEW_SCORE="10"
# Bounded GPU reasoner concurrency (separate from the CPU agent's limits).
ENV GPU_REASONER_MAX_INFLIGHT="1"

# ------- CPU agentic orchestration (/agent/*) --------------------------------
# CPU-only (no torch/cv2/ultralytics/transformers). Mounted in the SAME worker
# but on a SEPARATE bounded job queue so it can never block /detect. Mock by
# default: no real LLM key or DB required. Serious actions are drafts requiring
# human approval; memory backends are NOT durable -- use postgres/supabase in
# production (see docs/agentic_cpu_inside_runpod.md).
ENV AGENTIC_CPU_ENABLED="true"
ENV CPU_AGENT_MODE="mock"
ENV CPU_AGENT_MAX_INFLIGHT="2"
ENV CPU_AGENT_QUEUE_MAX="16"
ENV CPU_AGENT_JOB_TIMEOUT_MS="30000"
ENV CPU_AGENT_REQUIRE_APPROVAL="true"
ENV CPU_AGENT_ACTION_LOG_BACKEND="memory"
ENV CHECKPOINTER_BACKEND="memory"
ENV CPU_AGENT_LLM_PROVIDER="mock"
ENV CPU_AGENT_LLM_MODEL="mock"
ENV CPU_AGENT_DISABLE_ON_GPU_PRESSURE="true"
ENV CPU_AGENT_MAX_GPU_BUSY_RATIO="0.85"

# ------- DEIMv2 (legacy fallback) configuration ------------------------------
ENV DEIMV2_DEVICE="cuda"
ENV DEIMV2_BACKEND="official-deimv2-hf"
ENV DEIMV2_MODEL_ID="Intellindust/DEIMv2_DINOv3_S_COCO"
ENV DEIMV2_CONF="0.35"
ENV DEIMV2_IMG_SIZE="640"
ENV HF_HOME="/runpod-volume/.cache/huggingface"

# Build-time smoke test: verify every module that bootstrap.py -> import server
# needs is present in the image. Fails the build fast (not at runtime).
# Must NOT start uvicorn, download weights, or call /warmup.
RUN python - <<'PY'
import importlib, os, subprocess, sys
sys.path.insert(0, "/app")
import bitsandbytes as bnb  # noqa: F401
from transformers import AutoProcessor  # noqa: F401
print("bitsandbytes and transformers OK (used by grounding_dino_scanner + DEIMv2 detector paths)")
required = [
    "/app/worker_guards.py", "/app/worker_runtime.py", "/app/worker_security.py",
    "/app/server.py", "/app/bootstrap.py",
    "/app/shared", "/app/gpu_vision", "/app/temporal_reasoning", "/app/agentic_cpu",
]
for p in required:
    if not os.path.exists(p):
        raise FileNotFoundError(f"missing required runtime file: {p}")
for mod in ["worker_guards", "worker_runtime", "worker_security",
            "schema", "config_resolver", "vision_backend", "ws_vision",
            "shared", "gpu_vision", "temporal_reasoning", "agentic_cpu"]:
    importlib.import_module(mod)
    print("import OK:", mod)
# CPU agent import guard must run in a CLEAN subprocess so earlier imports of
# transformers/bitsandbytes/torch in this smoke test do not cause false positives.
probe = "\n".join([
    "import sys",
    "sys.path.insert(0, '/app')",
    "import agentic_cpu",
    "r = agentic_cpu.get_router()",
    "paths = sorted({route.path for route in r.routes})",
    "assert any(p.endswith('/health') for p in paths), paths",
    "forbidden = ['torch', 'torchvision', 'ultralytics', 'cv2', 'transformers']",
    "leaked = sorted(m for m in forbidden if m in sys.modules)",
    "print('LEAKED:' + ','.join(leaked))",
    "sys.exit(1 if leaked else 0)",
])
proc = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True)
out = (proc.stdout + proc.stderr).strip()
assert proc.returncode == 0, f"agentic_cpu leaked GPU deps in clean subprocess -> {out}"
print(proc.stdout.strip())
print("SafeLens worker startup import smoke test passed")
PY

# ------- Non-root runtime user (B10) -----------------------------------------
# Run as an unprivileged user. Writable runtime paths (startup log, Ultralytics
# config, HF cache, and the /runpod-volume model caches) are created and chowned
# so the user can write to them. NOTE: the mounted /runpod-volume must be
# writable by uid 10001 (see docs/runbook.md) -- weights resolve there at runtime.
RUN groupadd -r safelens && useradd -r -u 10001 -g safelens -m -d /home/safelens safelens \
    && mkdir -p /runpod-volume/models /runpod-volume/.cache/huggingface /tmp/Ultralytics \
    && chown -R safelens:safelens /app /runpod-volume /tmp/Ultralytics /home/safelens
ENV HOME="/home/safelens"
USER safelens

EXPOSE ${PORT}

# bootstrap.py starts server.py; falls back to a minimal health-only server if
# server.py fails to import (prevents silent container death).
CMD ["python", "-u", "/app/bootstrap.py"]
