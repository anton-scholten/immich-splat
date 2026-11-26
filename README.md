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
- `APP_PORT` — host port for the gallery, default `8080`. Change it if that port is
  already taken; the Immich webhook URL below does not change.
- `IMMICH_PUBLIC_URL` — optional. The Immich address you open in your browser
  (e.g. `https://photos.example.com`). Adds "Open in Immich" links to the gallery.

Compose refuses to start if `IMMICH_URL`, `IMMICH_API_KEY` or `WEBHOOK_SECRET` is empty.

**Keep your secrets out of git.** `.env`, any other `*.env`, personal
`docker-compose.override.yml` files and key files are git-ignored. If you contribute,
also enable the repo's pre-commit hook once per clone; it blocks commits that add
those files or lines that look like real keys:

```bash
git config core.hooksPath .githooks
```

A GitHub Actions check (`.github/workflows/secret-scan.yml`) runs gitleaks over the
full history on every push and pull request.

## 3. Build and start

```bash
docker compose up -d --build
```

The gallery is at `http://${APP_BIND}:${APP_PORT}` (default `http://127.0.0.1:8080`). The first job the worker runs downloads
ml-sharp's model checkpoint (~1.2 GB) into the `model_cache` volume; this happens once.

## 4. Wire up Immich

**Turn on tags first.** Immich hides tags until you enable them, per user account:
in Immich web open **Account Settings → Features → Tags**, switch **Enable** on
(also switch **Sidebar** on to get a *Tags* page in the left menu), and save.
The setting is stored on the server, so the phone apps pick it up too.

**Create the tag.** Open **Tags** in the sidebar and choose **Create tag**, name it
`splat`. (Or select any photo, press `t` or choose **Tag**, type `splat` and create it
there.) The workflow's tag filter below can only pick a tag that already exists, so
create it before the workflow. Tags added before the workflow exists send nothing.

Then go to **Workflows** and create a workflow:

- **Trigger:** *Asset tagged*
- **Filter:** *Filter by tags* = `splat`
- **Action:** *Trigger Webhook*
  - URL: `http://immich-splat-app:8080/webhook/immich` (always `8080`: Immich talks to
    the container directly over the Docker network, not through `APP_PORT`)
  - Method: `POST`
  - Header: name `X-Webhook-Secret`, value = your `WEBHOOK_SECRET`

Save and enable the workflow.

Why a tag and not an album: Immich workflows have only four triggers (*Asset created*,
*Asset metadata extracted*, *Asset OCR extracted*, *Asset tagged*). There is no
"added to album" trigger, so adding a photo to an album cannot send it. The tag
filter matters too: without it, every tag added to any photo would send a job.

The URL uses the container name, so Immich's server must be on the same Docker
network as `immich-splat-app` (the app joins `IMMICH_NETWORK` for this).

## 5. Sending photos

Tagging a photo `splat` in Immich is the only way to send one. Pick your device:

Tags must be turned on for your account first (step 4). Then:

**Immich web** (PC, phone or VR headset browser):
select the photo(s), choose **Tag** (in the ⋮ menu if it is not shown), then pick `splat`.

**Immich Android and iOS apps:**
select the photo(s), choose **Add tags**, then pick `splat`.
(menu names may differ slightly by Immich version)

The job then goes `queued` → `processing` → `done` and appears in the gallery.

## 6. Gallery

Open `http://<host>:<APP_PORT>` (default port `8080`) to see the gallery.

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
- **CPU fallback:** set `SHARP_DEVICE=cpu` in `.env`, and drop the GPU request with a
  `docker-compose.override.yml` next to `docker-compose.yml` (Compose loads it
  automatically, and git ignores it). It works, but each photo takes much longer.

  ```yaml
  services:
    worker:
      deploy: !reset {}
  ```

## Updating

Your settings live in `.env` and any `docker-compose.override.yml`, which git ignores,
so updating never touches them. From the repo folder:

```bash
git pull --rebase --autostash
```

```bash
docker compose up -d --build
```

`--autostash` sets aside any local edits to tracked files, pulls, then puts your edits
back. If it reports a conflict, fix the files it names and run `git stash drop`
once you are happy. Your photos, splats and the model cache are in Docker volumes and
survive the rebuild. To make plain `git pull` always do this:

```bash
git config pull.rebase true && git config rebase.autoStash true
```

Unused old image layers can pile up after many updates (the worker image is large);
`docker image prune` removes the dangling ones.

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
