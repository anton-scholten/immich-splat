# GPU build. If you don't have an NVIDIA GPU, see the CPU notes at the
# bottom of README.md instead of using this file as-is.
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# ml-sharp needs Python 3.13, which isn't in Ubuntu 22.04's default repos.
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common \
        git \
        wget \
        ca-certificates \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.13 \
        python3.13-venv \
        python3.13-dev \
    && rm -rf /var/lib/apt/lists/*

RUN wget -qO- https://bootstrap.pypa.io/get-pip.py | python3.13

# Apple's ml-sharp (SHARP: Sharp Monocular View Synthesis in Less Than a Second)
# https://github.com/apple/ml-sharp
WORKDIR /opt/ml-sharp
RUN git clone --depth 1 https://github.com/apple/ml-sharp.git . \
    && python3.13 -m pip install --no-cache-dir -r requirements.txt

RUN python3.13 -m pip cache purge

WORKDIR /worker
COPY worker.py .

ENV DATA_DIR=/data
ENV POLL_INTERVAL=3
# The model checkpoint (~1.2GB) downloads on first run and is cached at
# ~/.cache/torch/hub/checkpoints/ — mount a volume there (see docker-compose.yml)
# so you don't re-download it every time the container restarts.

CMD ["python3.13", "worker.py"]
