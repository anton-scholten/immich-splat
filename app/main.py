"""
immich-splat: receiver + status UI

Receives a webhook from an Immich Workflow ("Trigger Webhook" action),
downloads the original photo from the Immich API, drops a job file on the
shared /data volume for the worker container to pick up, and serves a small
status page plus the SuperSplat viewer so you can watch a splat once it's
ready.
"""

import json
import logging
import mimetypes
import os
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, Header, Request
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("immich-splat")

# --- configuration -----------------------------------------------------

DATA_DIR = Path(os.environ.get("DATA_DIR", "/data"))
JOBS_DIR = DATA_DIR / "jobs"
INCOMING_DIR = DATA_DIR / "incoming"
SPLATS_DIR = DATA_DIR / "splats"
for d in (JOBS_DIR, INCOMING_DIR, SPLATS_DIR):
    d.mkdir(parents=True, exist_ok=True)

IMMICH_URL = os.environ.get("IMMICH_URL", "").rstrip("/")
IMMICH_API_KEY = os.environ.get("IMMICH_API_KEY", "")
WEBHOOK_SECRET = os.environ.get("WEBHOOK_SECRET", "")
PUBLIC_BASE_URL = os.environ.get("PUBLIC_BASE_URL", "http://localhost:8080").rstrip("/")

VIEWER_DIST = Path(os.environ.get("VIEWER_DIST", "/app/viewer_dist"))

app = FastAPI(title="immich-splat")
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))

app.mount("/splats", StaticFiles(directory=str(SPLATS_DIR)), name="splats")
app.mount("/incoming", StaticFiles(directory=str(INCOMING_DIR)), name="incoming")
if VIEWER_DIST.exists():
    app.mount("/viewer", StaticFiles(directory=str(VIEWER_DIST), html=True), name="viewer")


# --- job storage (flat JSON files, no DB needed for a single-host stack) -

def _job_path(job_id: str) -> Path:
    return JOBS_DIR / f"{job_id}.json"


def write_job(job: dict[str, Any]) -> None:
    """Atomic write: write to a temp file then rename, so the worker never
    reads a half-written job file."""
    path = _job_path(job["id"])
    fd, tmp_path = tempfile.mkstemp(dir=str(JOBS_DIR), suffix=".tmp")
    with os.fdopen(fd, "w") as f:
        json.dump(job, f, indent=2)
    os.replace(tmp_path, path)


def read_job(job_id: str) -> dict[str, Any] | None:
    path = _job_path(job_id)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def list_jobs() -> list[dict[str, Any]]:
    jobs = []
    for path in JOBS_DIR.glob("*.json"):
        try:
            jobs.append(json.loads(path.read_text()))
        except (json.JSONDecodeError, OSError):
            continue
    jobs.sort(key=lambda j: j.get("created_at", ""), reverse=True)
    return jobs


# --- extracting an asset id out of an Immich workflow webhook payload ---
#
# Immich's Workflows feature (v3.0.0, preview) is new and its webhook body
# shape isn't nailed down yet. We try several likely shapes defensively and
# log the raw payload so you can adjust this if Immich changes the format.

def extract_asset_ids(payload: Any) -> list[str]:
    if payload is None:
        return []

    if isinstance(payload, list):
        ids = []
        for item in payload:
            ids.extend(extract_asset_ids(item))
        return ids

    if isinstance(payload, str):
        # sometimes people configure the webhook body as a raw asset id string
        return [payload] if _looks_like_uuid(payload) else []

    if not isinstance(payload, dict):
        return []

    # direct keys
    for key in ("assetId", "asset_id", "id"):
        val = payload.get(key)
        if isinstance(val, str) and _looks_like_uuid(val):
            return [val]

    # array of ids
    for key in ("assetIds", "asset_ids"):
        val = payload.get(key)
        if isinstance(val, list):
            return [v for v in val if isinstance(v, str) and _looks_like_uuid(v)]

    # nested single asset object
    for key in ("asset", "data"):
        val = payload.get(key)
        if isinstance(val, dict):
            found = extract_asset_ids(val)
            if found:
                return found

    # array of asset objects
    for key in ("assets", "data"):
        val = payload.get(key)
        if isinstance(val, list):
            found = extract_asset_ids(val)
            if found:
                return found

    return []


def _looks_like_uuid(s: str) -> bool:
    return len(s) == 36 and s.count("-") == 4


# --- Immich API -----------------------------------------------------------

async def download_original(asset_id: str, dest_dir: Path) -> Path:
    """Download the original asset bytes from Immich into dest_dir, returning
    the path to the saved file."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    url = f"{IMMICH_URL}/api/assets/{asset_id}/original"
    headers = {"x-api-key": IMMICH_API_KEY}

    async with httpx.AsyncClient(timeout=60.0) as client:
        async with client.stream("GET", url, headers=headers) as resp:
            resp.raise_for_status()
            ext = _guess_extension(resp.headers.get("content-type"), resp.headers.get("content-disposition"))
            out_path = dest_dir / f"original{ext}"
            with open(out_path, "wb") as f:
                async for chunk in resp.aiter_bytes(chunk_size=1 << 16):
                    f.write(chunk)
    return out_path


def _guess_extension(content_type: str | None, content_disposition: str | None) -> str:
    if content_disposition and "filename=" in content_disposition:
        fname = content_disposition.split("filename=")[-1].strip('"; ')
        suffix = Path(fname).suffix
        if suffix:
            return suffix
    if content_type:
        guessed = mimetypes.guess_extension(content_type.split(";")[0].strip())
        if guessed:
            return guessed
    return ".jpg"


# --- routes ----------------------------------------------------------------

@app.post("/webhook/immich")
async def immich_webhook(request: Request, x_webhook_secret: str | None = Header(default=None)):
    raw_body = await request.body()
    log.info("Incoming webhook, raw body: %s", raw_body[:2000])

    if WEBHOOK_SECRET:
        if x_webhook_secret != WEBHOOK_SECRET:
            log.warning("Webhook rejected: bad/missing X-Webhook-Secret header")
            return JSONResponse({"error": "unauthorized"}, status_code=401)

    try:
        payload = json.loads(raw_body) if raw_body else {}
    except json.JSONDecodeError:
        payload = {}

    asset_ids = extract_asset_ids(payload)
    if not asset_ids:
        log.warning(
            "No asset id found in webhook payload. Raw payload logged above; "
            "you may need to adjust extract_asset_ids() in main.py to match "
            "Immich's actual webhook shape."
        )
        return JSONResponse(
            {"error": "no asset id found in payload", "payload": payload},
            status_code=400,
        )

    created = []
    for asset_id in asset_ids:
        job_id = str(uuid.uuid4())
        job_dir = INCOMING_DIR / job_id
        try:
            saved_path = await download_original(asset_id, job_dir)
        except httpx.HTTPError as e:
            log.error("Failed to download asset %s from Immich: %s", asset_id, e)
            continue

        job = {
            "id": job_id,
            "asset_id": asset_id,
            "status": "queued",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "incoming_path": str(saved_path),
            "thumb_url": f"/incoming/{job_id}/{saved_path.name}",
            "error": None,
            "ply_url": None,
        }
        write_job(job)
        created.append(job_id)
        log.info("Created job %s for asset %s", job_id, asset_id)

    return {"created_jobs": created}


@app.get("/api/jobs")
async def api_jobs():
    jobs = list_jobs()
    for j in jobs:
        if j.get("status") == "done" and j.get("ply_url"):
            j["viewer_url"] = f"/viewer/index.html?content={j['ply_url']}"
    return jobs


@app.get("/", response_class=HTMLResponse)
async def status_page(request: Request):
    return templates.TemplateResponse("status.html", {"request": request})


@app.get("/healthz", response_class=PlainTextResponse)
async def healthz():
    return "ok"
