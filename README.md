# immich-splat

Add a photo to an album (or tag) in Immich → this stack downloads it, runs
Apple's [ml-sharp](https://github.com/apple/ml-sharp) (SHARP) to turn it into
a 3D Gaussian splat, and lets you view it in a self-hosted
[SuperSplat viewer](https://github.com/playcanvas/supersplat-viewer).

```
Immich (Workflow: album/tag trigger -> webhook)
   │  POST asset id
   ▼
app (FastAPI) ──downloads original via Immich API──▶ /data/incoming/<job>/
   │  writes job.json (status: queued)
   ▼
worker (GPU) ──polls /data/jobs──▶ runs `sharp predict` ──▶ /data/splats/<job>/scene.ply
   │  updates job.json (status: done)
   ▼
app status page (http://<host>:8080) ──▶ "View Splat" ──▶ /viewer/index.html?content=...
```

## Requirements

- Docker + Docker Compose v2
- An NVIDIA GPU on the Docker host, with the
  [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
  installed (the worker image is CUDA-based)
- Immich **v3.0.0 or later**, for the Workflows feature (currently in preview)

## 1. Create an Immich API key

In Immich: **Account Settings → API Keys → New API Key**. Grant at least
`asset.read` and `asset.download`.

## 2. Configure this app

```bash
cp .env.example .env
```

Edit `.env`:

- `IMMICH_URL` — your Immich server URL
- `IMMICH_API_KEY` — the key from step 1
- `WEBHOOK_SECRET` — make up a long random string, you'll reuse it in step 4
- `PUBLIC_BASE_URL` — how *you* reach this app in a browser (used to build
  the splat-viewer links on the status page)

## 3. Build and start

```bash
docker compose up -d --build
```

First start of the `worker` container will download ml-sharp's model
checkpoint (~1.2GB) the first time a job actually runs; it's cached in the
`model_cache` volume so this only happens once.

Visit `http://<host>:8080` — you should see an empty status page.

## 4. Wire up Immich

In Immich, go to **Workflows** and create a new workflow. Either trigger
works equally well here:

- **Trigger: Asset added to album** — create an album (e.g. `Send to Splat`),
  set that as the trigger. "Clicking the button" in daily use is then just
  using Immich's normal *Add to Album* action on a photo.
- **Trigger: Tag added to asset** — create a tag (e.g. `splat`) and trigger
  on that instead.

Add an action: **Trigger Webhook**

- URL: `http://<host>:8080/webhook/immich`
- Method: `POST`
- Header: name `X-Webhook-Secret`, value = the `WEBHOOK_SECRET` from your `.env`

Save the workflow and enable it.

## 5. Try it

Add a photo to the album (or tag) you configured. Refresh
`http://<host>:8080` — you should see a `queued` card, then `processing`,
then a **View Splat** link once the worker finishes.

## Where things live

Everything is under the `splat_data` Docker volume:

- `/data/incoming/<job_id>/original.*` — the downloaded photo
- `/data/jobs/<job_id>.json` — job status (`queued` / `processing` / `done` / `error`)
- `/data/splats/<job_id>/scene.ply` — the resulting Gaussian splat

## Notes and caveats

- **Immich's webhook payload shape.** Workflows/webhooks are a preview
  feature in Immich v3.0.0 and the exact JSON body isn't formally documented
  yet. `app/main.py`'s `extract_asset_ids()` tries several likely shapes
  (`assetId`, `asset.id`, `assetIds`, etc.) and logs the raw payload either
  way (`docker compose logs -f app`). If Immich sends something this doesn't
  recognize, check the logs for the raw body and tweak that one function —
  everything downstream just needs a list of asset UUIDs.
- **Stuck jobs.** If the worker container is killed mid-job, that job stays
  at `processing` forever (no crash-recovery/locking in this MVP). Just
  delete the offending `/data/jobs/<id>.json` file and re-trigger, or wire
  up a simple stale-job sweep if this bugs you.
- **Model license.** ml-sharp's code is Apple's own license, and the
  released model *weights* have a separate license
  (`LICENSE_MODEL` in the [ml-sharp repo](https://github.com/apple/ml-sharp)).
  Read it before using this for anything beyond personal experimentation.
- **CPU-only host.** This repo's `worker/Dockerfile` assumes a CUDA GPU. To
  run on CPU instead: change the base image to something like
  `python:3.13-slim`, drop the `deploy.resources.reservations` block in
  `docker-compose.yml`, and drop the CUDA-specific apt packages — `sharp
  predict` runs fine on CPU per Apple's README, just much slower. (Only the
  optional `--render` trajectory-video mode requires CUDA specifically; we
  don't use it since SuperSplat renders the splat live in the browser.)
- **Viewer version pin.** The SuperSplat viewer is cloned and built at a
  pinned tag (`SUPERSPLAT_VIEWER_TAG` build arg in `app/Dockerfile`,
  currently `v1.26.2`). Bump it there if you want a newer viewer release.
