# safelens-deimv2-worker/Dockerfile
# Builds a RunPod-compatible serverless worker that runs DEIMv2 inference.
#
# Model weights are NOT baked in - they are loaded at cold-start from
# HuggingFace hub (configurable via DEIMV2_MODEL_ID env var).

FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y \
    git wget curl libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Python worker dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Clone DEIMv2 source (architecture modules needed on PYTHONPATH)
ARG DEIMV2_REPO_URL=https://github.com/Intellindust-AI-Lab/DEIMv2.git
ARG DEIMV2_BRANCH=main
RUN git clone --depth 1 --branch ${DEIMV2_BRANCH} ${DEIMV2_REPO_URL} /opt/DEIMv2

# Install DEIMv2's own requirements
WORKDIR /opt/DEIMv2
RUN if [ -f requirements.txt ]; then pip install --no-cache-dir -r requirements.txt; fi

# Copy worker code
WORKDIR /app
COPY handler.py      /app/handler.py
COPY deimv2_infer.py /app/deimv2_infer.py
COPY schema.py       /app/schema.py

# DEIMv2 source on PYTHONPATH so its modules are importable
ENV PYTHONPATH="/opt/DEIMv2:/app:${PYTHONPATH}"

# Device: "cuda" (default on RunPod GPU workers) or "cpu"
ENV DEIMV2_DEVICE="cuda"

# HuggingFace model id - override to switch model size:
#   Intellindust-AI-Lab/DEIMv2-S  (9.7M params, 50.9 AP, recommended)
#   Intellindust-AI-Lab/DEIMv2-N  (3.6M params, 43.0 AP, ultra-light)
#   Intellindust-AI-Lab/DEIMv2-M  (18.1M params, 53.0 AP)
#   Intellindust-AI-Lab/DEIMv2-L  (32.2M params, 56.0 AP)
ENV DEIMV2_MODEL_ID="Intellindust-AI-Lab/DEIMv2-S"

# Confidence threshold (0..1). Lower = more detections.
ENV DEIMV2_CONF="0.35"

# Shorter-side resize resolution before inference.
ENV DEIMV2_IMG_SIZE="640"

# HuggingFace cache - mount a RunPod volume here to persist weights
ENV HF_HOME="/runpod-volume/.cache/huggingface"

CMD ["python", "-u", "handler.py"]
