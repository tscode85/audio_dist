# file: docker/Dockerfile
# ---------------------------------------------------------------------------
# Offline GPU image for the acoustic-gap toolkit.
#
# BUILD this on a machine WITH internet access. It bakes in:
#   * all pinned pip dependencies (GPU torch/torchvision/torchaudio via the
#     CUDA 12.1 wheels — self-contained, they bundle their own CUDA runtime)
#   * the WavLM x-vector weights + the PANN (Cnn14) FAD checkpoint (vendored
#     into /opt/model_cache), so the runtime container needs NO network at all.
# Then `docker save` the image and load it on the air-gapped host.
#
# Runtime target: Ubuntu 24.04 host with the NVIDIA driver 580.x series and the
# NVIDIA Container Toolkit installed. The torch 2.1.2 cu121 wheels bundle the
# CUDA 12.1 runtime and only need the host driver (580 is far newer than the 530
# minimum for CUDA 12.1) supplied by the container toolkit — so the base image's
# own CUDA version is not what torch uses. The container userspace is independent
# of the host's Ubuntu version, so a 22.04-based base runs fine on a 24.04 host.
# ---------------------------------------------------------------------------
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MODEL_CACHE=/opt/model_cache \
    HF_HOME=/opt/model_cache/hf

# --- OS packages: Python + audio I/O (libsndfile for soundfile, ffmpeg for
#     librosa's mp3/m4a decoding) ---
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-dev \
        libsndfile1 ffmpeg ca-certificates git \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/local/bin/python

WORKDIR /app

# --- Python dependencies (pinned). Install GPU torch/torchvision/torchaudio
#     from the CUDA 12.1 index first, then the rest of the pinned requirements. ---
COPY requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip \
    && python -m pip install \
        --index-url https://download.pytorch.org/whl/cu121 \
        torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 \
    && python -m pip install -r /app/requirements.txt

# --- Application code ---
COPY . /app
RUN python -m pip install -e .

# --- Vendor model weights into the image (requires network AT BUILD TIME) ---
# Toggle with `--build-arg DOWNLOAD_MODELS=0` for a CPU/dev image without weights.
ARG DOWNLOAD_MODELS=1
ARG WAVLM_MODEL=microsoft/wavlm-base-plus-sv
ARG FAD_MODEL=pann
ARG FAD_SR=16000
RUN if [ "$DOWNLOAD_MODELS" = "1" ]; then \
        python setup/download_models.py \
            --cache-dir "$MODEL_CACHE" \
            --wavlm "$WAVLM_MODEL" \
            --fad-model "$FAD_MODEL" \
            --fad-sr "$FAD_SR" ; \
    else \
        echo "Skipping model download (DOWNLOAD_MODELS=$DOWNLOAD_MODELS)"; \
    fi

# --- Enforce offline mode at RUNTIME (set AFTER the download step above) ---
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_DATASETS_OFFLINE=1

COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# Default: run the CLI. Override the subcommand/args at `docker run` time.
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["run", "--config", "/app/config/offline_gpu.yaml"]
