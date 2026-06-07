# safelens-deimv2-worker/Dockerfile
# Builds the SafeLens vision live-server worker for RunPod load-balancing endpoints.
#
# Default backend: EdgeCrafter (ECDet-S boxes + optional ECPose-S poses).
# Legacy fallback: DEIMv2 (VISION_BACKEND=deimv2).
#
# Architecture: long-running FastAPI/uvicorn server (adapted from Kingo333/fluxrt-serverless).
# Model weights are NOT baked in -- EdgeCrafter checkpoints are downloaded at
# runtime into EDGECRAFTER_CACHE_DIR (a RunPod volume); DEIMv2 weights come from HF Hub.
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

# Copy worker code
COPY schema.py /app/schema.py
COPY edgecrafter_loader.py /app/edgecrafter_loader.py
COPY vision_backend.py /app/vision_backend.py
COPY deimv2_infer.py /app/deimv2_infer.py
COPY official_deimv2_loader.py /app/official_deimv2_loader.py
COPY server.py /app/server.py
COPY ws_vision.py /app/ws_vision.py
COPY bootstrap.py /app/bootstrap.py
COPY handler.py /app/handler.py

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
# edgecrafter (default) | deimv2 (legacy fallback)
ENV VISION_BACKEND="edgecrafter"

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

# ------- DEIMv2 (legacy fallback) configuration ------------------------------
ENV DEIMV2_DEVICE="cuda"
ENV DEIMV2_BACKEND="official-deimv2-hf"
ENV DEIMV2_MODEL_ID="Intellindust/DEIMv2_DINOv3_S_COCO"
ENV DEIMV2_CONF="0.35"
ENV DEIMV2_IMG_SIZE="640"
ENV HF_HOME="/runpod-volume/.cache/huggingface"

EXPOSE ${PORT}

# bootstrap.py starts server.py; falls back to a minimal health-only server if
# server.py fails to import (prevents silent container death).
CMD ["python", "-u", "/app/bootstrap.py"]
