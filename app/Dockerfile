# --- stage 1: build the official SuperSplat viewer as a static site -------
FROM node:20-slim AS viewer-build

# Pin a specific release for reproducibility. Bump this if you want a newer
# viewer: https://github.com/playcanvas/supersplat-viewer/releases
ARG SUPERSPLAT_VIEWER_TAG=v1.26.2

RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
RUN git clone --depth 1 --branch ${SUPERSPLAT_VIEWER_TAG} \
    https://github.com/playcanvas/supersplat-viewer.git .
RUN npm ci
RUN npm run build

# supply a default settings.json so the viewer's default `?settings=./settings.json`
# resolves even before any splat-specific settings exist
COPY viewer_settings.json /src/dist/settings.json


# --- stage 2: python app ----------------------------------------------------
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .
COPY templates ./templates

COPY --from=viewer-build /src/dist /app/viewer_dist

ENV DATA_DIR=/data
ENV VIEWER_DIST=/app/viewer_dist

EXPOSE 8080

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
