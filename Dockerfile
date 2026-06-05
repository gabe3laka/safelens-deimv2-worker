# safelens-deimv2-worker/Dockerfile
# Builds the SafeLens DEIMv2 live-server worker for RunPod load-balancing endpoints.
#
# Architecture: long-running FastAPI/uvicorn server (adapted from Kingo333/fluxrt-serverless).
# Model weights are NOT baked in -- they are downloaded at cold-start from HuggingFace Hub.
#
# RunPod endpoint type: HTTP (load-balancing), not serverless queue.
# Health probe: GET /health or GET /ping (returns immediately, no model required).

FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y \
    git wget curl libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Python worker dependencies (includes fastapi + uvicorn for live-server mode)
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Clone DEIMv2 source (architecture modules needed on PYTHONPATH).
# The official DEIMv2 model is loaded via the engine.* package from this clone
# (PyTorchModelHubMixin custom class), NOT via transformers Auto classes.
ARG DEIMV2_REPO_URL=https://github.com/Intellindust-AI-Lab/DEIMv2.git
ARG DEIMV2_BRANCH=main
RUN git clone --depth 1 --branch ${DEIMV2_BRANCH} ${DEIMV2_REPO_URL} /opt/DEIMv2

# Install DEIMv2's own requirements
WORKDIR /opt/DEIMv2
RUN if [ -f requirements.txt ]; then pip install --no-cache-dir -r requirements.txt; fi

# Re-apply SafeLens pinned runtime deps AFTER upstream DEIMv2 deps.
# DEIMv2's requirements.txt may downgrade/overwrite transformers and friends.
# This force-reinstall layer guarantees the final, compatible versions win.
WORKDIR /app
RUN pip install --no-cache-dir --upgrade \
    "transformers>=4.46.0,<4.50.0" \
    "huggingface-hub>=0.26.0" \
    "safetensors>=0.4.5" \
    "tokenizers>=0.20.0" \
    "accelerate>=1.0.0" \
    "timm>=1.0.11" \
    "opencv-python-headless>=4.10.0.84"

# Copy worker code
COPY handler.py /app/handler.py
COPY deimv2_infer.py /app/deimv2_infer.py
COPY official_deimv2_loader.py /app/official_deimv2_loader.py
COPY schema.py /app/schema.py
COPY server.py /app/server.py
COPY bootstrap.py /app/bootstrap.py

# DEIMv2 source on PYTHONPATH so its engine.* modules are importable
ENV PYTHONPATH="/opt/DEIMv2:/app:${PYTHONPATH}"

# ------- RunPod HTTP endpoint configuration ----------------------------------

# Port the uvicorn server listens on. RunPod load-balancer forwards traffic here.
ENV PORT="8000"

# Uvicorn log level
ENV UVICORN_LOG_LEVEL="info"

# Set to "true" to skip model load on startup (diagnostic / smoke-test mode).
ENV SKIP_WARMUP="false"

# Set to "true" to start model load immediately on container start.
ENV AUTO_WARMUP="true"

# Warmup timeout in seconds before giving up and setting status=error.
ENV WARMUP_TIMEOUT_S="600"

# Log file written during startup (readable via GET /debug/startup)
ENV STARTUP_LOG="/tmp/safelens_startup.log"

# ------- DEIMv2 inference configuration -------------------------------------

# Device: "cuda" (default on RunPod GPU workers) or "cpu"
ENV DEIMV2_DEVICE="cuda"

# Detection backend:
#   official-deimv2-hf      -> official DEIMv2 PyTorchModelHubMixin loader (default)
#   transformers-fallback   -> facebook/detr-resnet-50 (pipeline validation only,
#                              clearly labelled FALLBACK, NOT DEIMv2)
ENV DEIMV2_BACKEND="official-deimv2-hf"

# Official DEIMv2-S HuggingFace model id (DINOv3 ViT-Tiny backbone, COCO).
# Override to switch model size, e.g.:
#   Intellindust/DEIMv2_DINOv3_S_COCO  (9.7M params, 50.9 AP, default)
#   Intellindust/DEIMv2_DINOv3_M_COCO  (18.1M params, 53.0 AP)
ENV DEIMV2_MODEL_ID="Intellindust/DEIMv2_DINOv3_S_COCO"

# Confidence threshold (0..1). Lower = more detections.
ENV DEIMV2_CONF="0.35"

# Square resize resolution before inference (DEIMv2-S eval size = 640).
ENV DEIMV2_IMG_SIZE="640"

# HuggingFace cache -- mount a RunPod volume here to persist weights.
ENV HF_HOME="/runpod-volume/.cache/huggingface"

EXPOSE ${PORT}

# bootstrap.py: starts server.py; falls back to minimal health-only server
# if server.py fails to import (prevents silent container death).
CMD ["python", "-u", "/app/bootstrap.py"]
