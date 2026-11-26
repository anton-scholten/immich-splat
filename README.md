# immich-splat

Tag a photo in Immich and get a 3D Gaussian splat of it, viewable in the browser.
The stack downloads the photo, runs Apple's [ml-sharp](https://github.com/apple/ml-sharp)
(SHARP) to build the splat, and serves it in a self-hosted
[SuperSplat viewer](https://github.com/playcanvas/supersplat-viewer) with the camera
placed where your original photo was taken.

```
Immich: tag "splat" added to asset
   │  Workflow: Asset tagged → Filter by tags = splat → Trigger Webhook
   │  POST (X-Webhook-Secret header)
   ▼
app (FastAPI) ──downloads original from Immich API──▶ /data/incoming/<job>/
   │  writes /data/jobs/<job>.json (status: queued) and /data/thumbs/<job>.jpg
   ▼
worker (GPU) ──polls /data/jobs── runs ml-sharp (model kept loaded in memory)
   │  writes /data/splats/<job>/scene.spz + settings.json
   │  job status: done
   ▼
app gallery (/) ──▶ SuperSplat viewer (scene.spz, per-job camera)
```

## Requirements

- Docker with Compose v2
- Immich **v3.0.0 or later** (Workflows; still a preview feature in Immich)
- An NVIDIA GPU on the Docker host with the
  [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
  (CPU fallback: see GPU notes)
- This stack must share a Docker network with your Immich server (see step 2)

## 1. Create an Immich API key

Immich: **Account Settings → API Keys → New API Key**. Grant only
`asset.download`. Add `asset.view` only if you want RAW or other formats that
Immich cannot serve as an original (the app then uses Immich's rendered preview).

## 2. Configure

```bash
cp .env.example .env
```

Edit `.env`:

- `IMMICH_URL` — default `http://immich-server:2283` works when this stack joins
  Immich's Docker network and the Immich server container is named `immich-server`.
  Otherwise use the Immich server's own URL.
- `IMMICH_API_KEY` — the key from step 1
- `WEBHOOK_SECRET` — generate with `openssl rand -hex 32`; you reuse it in step 4
- `IMMICH_NETWORK` — the Docker network Immich runs on (usually `immich_default`)
- `APP_BIND` — leave `127.0.0.1` and publish through a reverse proxy, or set another address
- `IMMICH_PUBLIC_URL` — optional. The Immich address you open in your browser
  (e.g. `https://photos.example.com`). Adds "Open in Immich" links to the gallery.

Compose refuses to start if `IMMICH_URL`, `IMMICH_API_KEY` or `WEBHOOK_SECRET` is empty.

## 3. Build and start

```bash
docker compose up -d --build
```

The app listens on `${APP_BIND}:8080`. The first job the worker runs downloads
ml-sharp's model checkpoint (~1.2 GB) into the `model_cache` volume; this happens once.

## 4. Wire up Immich

In Immich, first create a tag (e.g. `splat`). Then go to **Workflows** and create a workflow:

- **Trigger:** *Asset tagged*
- **Filter:** *Filter by tags* = `splat`
- **Action:** *Trigger Webhook*
  - URL: `http://immich-splat-app:8080/webhook/immich`
  - Method: `POST`
  - Header: name `X-Webhook-Secret`, value = your `WEBHOOK_SECRET`

Save and enable the workflow.

The tag filter matters: Immich workflows have only four triggers (*Asset created*,
*Asset metadata extracted*, *Asset OCR extracted*, *Asset tagged*). There is no
"added to album" trigger. Without the tag filter, every tag added to any photo
would send a job.

The URL uses the container name, so Immich's server must be on the same Docker
network as `immich-splat-app` (the app joins `IMMICH_NETWORK` for this).

## 5. Sending photos

Tagging a photo `splat` in Immich is the only way to send one. Pick your device:

**Immich web** (PC, phone or VR headset browser):
open the photo, select it, choose **Tag**, then pick `splat`.

**Immich Android and iOS apps:**
tags must be enabled first: **Account settings → Features → Tags**.
Then open the photo, select it, choose **Add tags**, then pick `splat`.
(menu names may differ slightly by Immich version)

The job then goes `queued` → `processing` → `done` and appears in the gallery.

## 6. Gallery

Open `http://<host>:8080` to see the gallery.

- **Thumbnails** of every photo you sent. Photos still being processed are dimmed
  with a progress ring. Tap a finished one to open it in the SuperSplat viewer.
  The viewer opens on the photo's own view; press play for camera motion, and drag
  to look around.
- **Sort** by date taken, date added, name, original size, splat size or processing
  time, ascending or descending. **Filter** by status (all, ready, in progress, failed).
- **Failed** jobs can be retried or deleted.
- **Stats bar** shows disk use, host CPU and RAM, GPU load and the worker state.

Jobs can be deleted from the gallery. This removes the job record, its thumbnail,
its incoming photo and its splat files. Jobs in `processing` cannot be deleted.

Use HTTPS through a reverse proxy to view it from a phone or VR headset. WebXR and
other phone/VR features need a secure context.

## Where things live

Everything the stack writes is under `/data` (the `splat_data` volume):

- `/data/incoming/<job_id>/original.*` — downloaded photo (deleted by the worker after success)
- `/data/jobs/<job_id>.json` — job record and status
- `/data/thumbs/<job_id>.jpg` — thumbnail for the gallery
- `/data/splats/<job_id>/scene.spz` — splat for the viewer
- `/data/splats/<job_id>/scene.ply` — only if `KEEP_PLY=true`
- `/data/splats/<job_id>/settings.json` — per-job viewer settings (camera)
- `/data/status/worker.json` — worker heartbeat (state, GPU and CPU load) for the stats bar

The model checkpoint is in `/cache` (the `model_cache` volume).

## GPU notes

- The worker needs the NVIDIA Container Toolkit on the host. The PyTorch wheels
  bundle their own CUDA runtime, so only the NVIDIA driver is needed on the host. The
  image uses the CUDA 12.6 wheels (any CUDA 12 driver, 525 or newer; tested with 580),
  which still support older Pascal GPUs (GTX 10xx, TITAN Xp). For Turing or newer
  you can build with
  `--build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128`.
- **CPU fallback:** set `SHARP_DEVICE=cpu`, and remove the `deploy:` block from the
  `worker` service in `docker-compose.yml`. It works, but each photo takes much longer.

## Caveats

- **Preview API.** Immich workflows are preview. Trigger and action names, and the
  webhook body, may change in later Immich versions. The app parses the body Immich's core
  plugin sends today: `{type, trigger, data: {asset: {...}}}`.
- **Model license.** The ml-sharp code is under Apple's license. The released model
  weights have a separate license (`LICENSE_MODEL` in the
  [ml-sharp repo](https://github.com/apple/ml-sharp)). Read it before using results
  beyond personal experimentation.
- **Viewer version.** The SuperSplat viewer is pinned to v1.44.0 by the `SUPERSPLAT_VIEWER_TAG`
  build arg in `app/Dockerfile`.
