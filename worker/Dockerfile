# immich-splat worker: Apple ml-sharp (GPU inference) + splat-transform (.spz).
#
# Build args:
#   TORCH_INDEX_URL  PyTorch wheel index. ml-sharp pins torch 2.8.0 / torchvision 0.23.0.
#                    Default is the cu126 index because those wheels still ship kernels for
#                    old GPUs (Pascal, sm_61, e.g. TITAN Xp / GTX 10xx). For newer GPUs
#                    (Turing and up) you may use https://download.pytorch.org/whl/cu128.
#                    Wheels bundle the CUDA libs, so no CUDA base image is needed; the host
#                    only needs the NVIDIA driver + container toolkit.
#   SHARP_REF        ml-sharp commit to install.
#   SPLAT_TRANSFORM_VERSION  @playcanvas/splat-transform version.

# ---- node runtime + splat-transform --------------------------------------------------
FROM node:22-slim AS node
ARG SPLAT_TRANSFORM_VERSION=3.12.0
RUN mkdir -p /opt/splat-transform \
    && cd /opt/splat-transform \
    && npm install --no-audit --no-fund --omit=dev @playcanvas/splat-transform@${SPLAT_TRANSFORM_VERSION} \
    && npm cache clean --force

# ---- python deps ---------------------------------------------------------------------
FROM python:3.13-slim AS build
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu126
ARG SHARP_REF=aed6527499ef91cba3b54c18d49a870f25947190
RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates build-essential \
    && rm -rf /var/lib/apt/lists/*
RUN python -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH PIP_NO_CACHE_DIR=1
RUN git clone https://github.com/apple/ml-sharp.git /opt/ml-sharp \
    && git -C /opt/ml-sharp checkout ${SHARP_REF} \
    && rm -rf /opt/ml-sharp/.git
WORKDIR /opt/ml-sharp
# torch first, from the chosen index (brings its own matching nvidia-* libs); then the rest
# of ml-sharp's pinned requirements without the nvidia-* pins (they are cu128 builds).
RUN pip install torch==2.8.0 torchvision==0.23.0 --index-url ${TORCH_INDEX_URL} \
    && sed -i '/^nvidia-/d' requirements.txt \
    && pip install -r requirements.txt
# Small extras in their own layer so changing them doesn't rebuild the ~6.5 GB torch layer.
RUN pip install nvidia-ml-py==13.580.82 psutil==7.1.0

# ---- final image ---------------------------------------------------------------------
FROM python:3.13-slim
COPY --from=node /usr/local/bin/node /usr/local/bin/node
COPY --from=node /opt/splat-transform /opt/splat-transform
RUN ln -s /opt/splat-transform/node_modules/.bin/splat-transform /usr/local/bin/splat-transform
COPY --from=build /opt/venv /opt/venv
COPY --from=build /opt/ml-sharp /opt/ml-sharp

RUN groupadd -g 1000 splat && useradd -u 1000 -g 1000 -m -d /home/splat -s /usr/sbin/nologin splat \
    && mkdir -p /data /cache/torch \
    && chown -R 1000:1000 /data /cache

ENV PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    HOME=/tmp \
    DATA_DIR=/data \
    TORCH_HOME=/cache/torch \
    KEEP_PLY=false \
    SHARP_DEVICE=default \
    POLL_INTERVAL=3
# Less GPU memory fragmentation: fp16 then fits in ~3.4 GiB instead of ~3.6 GiB.
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

WORKDIR /worker
COPY worker.py .
USER 1000:1000
# The model checkpoint (~1.2GB) downloads on first run into TORCH_HOME; mount /cache.
CMD ["python", "worker.py"]
