# immich-splat

Tag a photo `splat` in Immich and get a 3D Gaussian splat of it in your browser,
made by Apple's [ml-sharp](https://github.com/apple/ml-sharp) and shown in the
[SuperSplat viewer](https://github.com/playcanvas/supersplat-viewer).

```
Immich: tag "splat" → workflow webhook → app (downloads photo) → worker (GPU) → gallery → viewer
```

## Requirements

- Docker with Compose v2, on the same machine/Docker network as Immich
- Immich **v3.0.0+** (Workflows, still a preview feature)
- NVIDIA GPU with **6 GB VRAM recommended (4 GB minimum)**, driver 525+, and the
  [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
- About 15 GB disk for the images and model

| GPU VRAM | Mode (chosen automatically) | Time per photo* |
|---|---|---|
| 10 GB+ | fp32 on the GPU | ~7 s |
| 6–8 GB | fp32, the two big encoders kept in RAM and copied in when used | ~10 s |
| 4 GB | fp16, only if nothing else uses the GPU (needs ~3.7 GiB free) | ~12 s |
| less | CPU, ~8 GB RAM | 1 min (6-core) to 20 min (old CPU) |

\*Measured on a TITAN Xp (Pascal); a GTX 1660 is roughly 2× slower. The first photo
after a start takes longer while the model loads.

## Setup

1. **API key.** Immich → Account Settings → API Keys → New. Permission: `asset.download`
   (add `asset.view` for RAW files).
2. **Configure.**
   ```bash
   cp .env.example .env
   ```
   Fill in `IMMICH_API_KEY` and `WEBHOOK_SECRET` (`openssl rand -hex 32`). Check
   `IMMICH_URL` and `IMMICH_NETWORK` match your Immich setup. Other options are
   explained in `.env.example`.
3. **Start.**
   ```bash
   docker compose up -d --build
   ```
   The first job downloads the model (~1.2 GB) once.
4. **Enable tags in Immich.** Account Settings → Features → Tags → Enable (and Sidebar).
   Then create a tag named `splat` (Tags page → Create tag).
5. **Create the workflow.** Immich → Workflows → new:
   - Trigger: *Asset tagged*
   - Filter: *Filter by tags* = `splat`
   - Action: *Trigger Webhook*, URL `http://immich-splat-app:8080/webhook/immich`,
     method `POST`, header `X-Webhook-Secret` = your `WEBHOOK_SECRET`

   The URL always uses port 8080 (container to container). Immich has no
   "added to album" trigger, so albums can't be used; the tag filter stops other
   tags from sending photos.

## Use

Tag any photo `splat` (web: select → **Tag**; phone app: select → **Add tags**).
Open the gallery at `http://<host>:8080`:

- Photos being processed are dimmed with a progress ring; tap a finished one to
  open it in 3D (it starts on the photo's own view; drag to look around).
- Sort, filter, choose whether splats open in the same or a new tab, retry failed
  jobs, delete jobs, and see disk/CPU/GPU use.

**From other devices:** the gallery listens on `127.0.0.1` only. Set `APP_BIND=0.0.0.0`
in `.env` and restart (there is no login, so keep it on your home network), or put it
behind your reverse proxy with HTTPS (needed for VR mode). Change `APP_PORT` if 8080 is taken.

## Update

```bash
git pull --rebase --autostash
```

```bash
docker compose up -d --build
```

Your `.env` and data volumes are kept. `docker image prune` frees old image layers.

## GPU and CPU options

- `SHARP_PRECISION=auto` (default) picks the mode from the table. Force one with
  `fp32`, `fp32-offload` or `fp16`.
- If a photo fails (out of memory, or NaN results, which GTX 16xx cards give in fp16),
  the worker steps down by itself and remembers it: fp32 → fp32-offload → fp16 → CPU.
  Each job's `precision` field in `/api/jobs` shows what it ran on.
- For Turing or newer GPUs, `--build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128`
  gives newer CUDA wheels; the default (cu126) also supports older Pascal cards.
- **CPU only:** set `SHARP_DEVICE=cpu` and add a `docker-compose.override.yml`:
  ```yaml
  services:
    worker:
      deploy: !reset {}
  ```

## Notes

- Data lives in the `splat_data` volume (`/data`: jobs, thumbnails, splats); the model in `model_cache`.
- `.env` and override files are git-ignored. Contributors: `git config core.hooksPath .githooks`
  enables a pre-commit secret check.
- ml-sharp's model weights have their own license (`LICENSE_MODEL` in its repo).
