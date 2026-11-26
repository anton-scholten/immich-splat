# --- stage 1: build the official SuperSplat viewer as a static site -------
FROM node:20-slim AS viewer-build

# Pin a specific release for reproducibility. Bump this if you want a newer
# viewer: https://github.com/playcanvas/supersplat-viewer/releases
ARG SUPERSPLAT_VIEWER_TAG=v1.44.0

RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
RUN git clone --depth 1 --branch ${SUPERSPLAT_VIEWER_TAG} \
    https://github.com/playcanvas/supersplat-viewer.git .
RUN npm ci
RUN npm run build

# The static viewer (index.html, index.js, index.css, wasm) is written to public/.
# dist/ is the npm module build and has no index.html, so serve public/.
# Default settings.json so the viewer's default `?settings=./settings.json`
# resolves even before any splat-specific settings exist.
COPY viewer_settings.json /src/public/settings.json


# --- stage 2: python app ----------------------------------------------------
FROM python:3.12-slim

# Non-root user. /data is the shared volume; it must be owned by this user so a
# fresh named volume inherits the ownership.
RUN groupadd --gid 1000 splat \
    && useradd --uid 1000 --gid 1000 --no-create-home --shell /usr/sbin/nologin splat \
    && mkdir -p /data && chown 1000:1000 /data

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .
COPY templates ./templates

COPY --from=viewer-build /src/public /app/viewer_dist

ENV DATA_DIR=/data
ENV VIEWER_DIST=/app/viewer_dist
# Read-only root filesystem: no bytecode writes, and HOME points at the /tmp tmpfs.
ENV PYTHONDONTWRITEBYTECODE=1
ENV HOME=/tmp

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD ["python", "-c", "import sys, urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8080/healthz', timeout=4).status == 200 else 1)"]

USER 1000:1000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
