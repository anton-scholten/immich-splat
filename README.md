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
   │  writes /data/jobs/<job>.json (status: queued)
   ▼
worker (GPU) ──polls /data/jobs── runs ml-sharp (model kept loaded in memory)
   │  writes /data/splats/<job>/scene.spz + settings.json + thumb.jpg
   │  job status: done
   ▼
app status page ──▶ /jobs/<job>  or  SuperSplat viewer (scene.spz, per-job camera)
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

## 5. Use it

Tag a photo `splat`. In the app, open the status page at `http://<host>:8080`
(or `/jobs/<id>` for one job: it waits, then opens the viewer by itself). Jobs go `queued` → `processing` → `done`. When a job
is `done` the viewer link opens SuperSplat with `scene.spz` and the per-job camera,
which starts paused on the same viewpoint as the original photo. Press play for a
camera fly-around; drag to look around.

Use HTTPS through a reverse proxy to view it from a phone or VR headset. WebXR and
other phone/VR features need a secure context.

## Where things live

Everything the stack writes is under `/data` (the `splat_data` volume):

- `/data/incoming/<job_id>/original.*` — downloaded photo (deleted by the worker after success)
- `/data/jobs/<job_id>.json` — job record and status
- `/data/splats/<job_id>/scene.spz` — splat for the viewer
- `/data/splats/<job_id>/scene.ply` — only if `KEEP_PLY=true`
- `/data/splats/<job_id>/settings.json` — per-job viewer settings (camera)
- `/data/splats/<job_id>/thumb.jpg` — thumbnail for the status page

The model checkpoint is in `/cache` (the `model_cache` volume).

Jobs can be deleted from the status page. This removes the job record, its incoming
photo and its splat files. Jobs in `processing` cannot be deleted.

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
