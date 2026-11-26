"""
immich-splat: receiver + status UI

Receives a webhook from an Immich workflow, downloads the photo from the
Immich API in a background task, drops a job file on the shared /data volume
for the worker container, and serves a status page, the SuperSplat viewer and
the finished splats.
"""

import hmac
import json
import logging
import os
import shutil
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote

import httpx
from fastapi import BackgroundTasks, FastAPI, Header, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("immich-splat")

# --- configuration -----------------------------------------------------

IMMICH_URL = os.environ.get("IMMICH_URL", "").strip().rstrip("/")
IMMICH_API_KEY = os.environ.get("IMMICH_API_KEY", "").strip()
WEBHOOK_SECRET = os.environ.get("WEBHOOK_SECRET", "")

_missing = [
    name
    for name, value in (
        ("WEBHOOK_SECRET", WEBHOOK_SECRET),
        ("IMMICH_URL", IMMICH_URL),
        ("IMMICH_API_KEY", IMMICH_API_KEY),
    )
    if not value
]
if _missing:
    raise RuntimeError(
        "Refusing to start: required env var(s) empty or unset: " + ", ".join(_missing)
    )

DATA_DIR = Path(os.environ.get("DATA_DIR", "/data"))
JOBS_DIR = DATA_DIR / "jobs"
INCOMING_DIR = DATA_DIR / "incoming"
SPLATS_DIR = DATA_DIR / "splats"
for d in (JOBS_DIR, INCOMING_DIR, SPLATS_DIR):
    d.mkdir(parents=True, exist_ok=True)

VIEWER_DIST = Path(os.environ.get("VIEWER_DIST", "/app/viewer_dist"))

MAX_DOWNLOAD_BYTES = 300 * 1024 * 1024
# Formats the worker can open directly. Everything else uses an Immich-rendered JPEG.
SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".heic", ".heif", ".webp", ".tif", ".tiff", ".bmp"}
BLOCKING_STATUSES = {"queued", "processing", "done"}

app = FastAPI(title="immich-splat")
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))

app.mount("/splats", StaticFiles(directory=str(SPLATS_DIR)), name="splats")
if VIEWER_DIST.exists():
    app.mount("/viewer", StaticFiles(directory=str(VIEWER_DIST), html=True), name="viewer")

# asset_id -> job_id for downloads still running (no job file exists yet)
_downloading: dict[str, str] = {}


def make_client() -> httpx.AsyncClient:
    """Client factory; tests replace it to inject a fake transport."""
    return httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=10.0))


# --- job storage (flat JSON files, no DB needed for a single-host stack) -

def parse_job_id(job_id: str) -> str:
    """Return canonical uuid string or 404. Also blocks path traversal."""
    try:
        parsed = uuid.UUID(job_id)
    except ValueError:
        raise HTTPException(status_code=404, detail="job not found")
    if str(parsed) != job_id.lower():
        raise HTTPException(status_code=404, detail="job not found")
    return str(parsed)


def _job_path(job_id: str) -> Path:
    return JOBS_DIR / f"{job_id}.json"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_job(job: dict[str, Any]) -> None:
    """Atomic write: temp file in the same dir, then rename, so the worker
    never reads a half-written job file."""
    fd, tmp_path = tempfile.mkstemp(dir=str(JOBS_DIR), suffix=".tmp")
    with os.fdopen(fd, "w") as f:
        json.dump(job, f, indent=2)
    os.replace(tmp_path, _job_path(job["id"]))


def read_job(job_id: str) -> dict[str, Any] | None:
    try:
        return json.loads(_job_path(job_id).read_text())
    except (OSError, json.JSONDecodeError):
        return None


def list_jobs() -> list[dict[str, Any]]:
    jobs = []
    for path in JOBS_DIR.glob("*.json"):
        try:
            jobs.append(json.loads(path.read_text()))
        except (OSError, json.JSONDecodeError):
            continue
    jobs.sort(key=lambda j: j.get("created_at", ""), reverse=True)
    return jobs


def with_viewer_url(job: dict[str, Any]) -> dict[str, Any]:
    if job.get("status") == "done":
        jid = job["id"]
        splat = job.get("splat_url") or f"/splats/{jid}/scene.spz"
        settings = job.get("settings_url") or f"/splats/{jid}/settings.json"
        thumb = job.get("thumb_url") or f"/splats/{jid}/thumb.jpg"
        job["viewer_url"] = (
            f"/viewer/index.html?content={quote(splat, safe='')}"
            f"&settings={quote(settings, safe='')}&poster={quote(thumb, safe='')}"
            # Start paused on the photo's own view; the viewer's default figure-8
            # camera motion swings too far for a single-image splat.
            "&noanim"
        )
    return job


def find_blocking_job(asset_id: str) -> tuple[str, str] | None:
    """(job_id, status) of an existing job that should not be repeated."""
    if asset_id in _downloading:
        return _downloading[asset_id], "downloading"
    for job in list_jobs():
        if job.get("asset_id") == asset_id and job.get("status") in BLOCKING_STATUSES:
            return job["id"], job["status"]
    return None


# --- downloading from Immich ------------------------------------------------

class DownloadError(Exception):
    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


def _ext_from_disposition(value: str | None) -> str:
    if not value or "filename=" not in value:
        return ""
    name = value.split("filename=", 1)[1].split(";", 1)[0].strip().strip('"')
    return Path(name).suffix.lower()


async def _stream_to(client: httpx.AsyncClient, url: str, dest: Path, params: dict | None = None) -> str:
    """Stream a GET response to dest, aborting above MAX_DOWNLOAD_BYTES.
    Returns the file extension from Content-Disposition ("" if none)."""
    headers = {"x-api-key": IMMICH_API_KEY}
    async with client.stream("GET", url, headers=headers, params=params) as resp:
        if resp.status_code >= 400:
            raise DownloadError(f"Immich returned HTTP {resp.status_code}", resp.status_code)
        ext = _ext_from_disposition(resp.headers.get("content-disposition"))
        total = 0
        with open(dest, "wb") as f:
            async for chunk in resp.aiter_bytes(chunk_size=1 << 16):
                total += len(chunk)
                if total > MAX_DOWNLOAD_BYTES:
                    raise DownloadError(
                        f"download larger than {MAX_DOWNLOAD_BYTES // (1024 * 1024)} MB, aborted"
                    )
                f.write(chunk)
    if total == 0:
        raise DownloadError("Immich returned an empty file")
    return ext


async def fetch_asset(asset_id: str, filename: str | None, job_dir: Path) -> tuple[Path, str]:
    """Download into job_dir. Returns (final path, image_kind)."""
    job_dir.mkdir(parents=True, exist_ok=True)
    ext = Path(filename).suffix.lower() if filename else ""
    base = f"{IMMICH_URL}/api/assets/{asset_id}"
    part = job_dir / "download.part"

    async with make_client() as client:
        if not ext:
            # No filename (bare curl body): take the original and read the
            # extension from Content-Disposition, so asset.download is enough.
            ext = await _stream_to(client, f"{base}/original", part)
            if ext in SUPPORTED_EXTS:
                os.replace(part, job_dir / f"original{ext}")
                return job_dir / f"original{ext}", "original"
            part.unlink(missing_ok=True)
        if ext in SUPPORTED_EXTS:
            await _stream_to(client, f"{base}/original", part)
            final, kind = job_dir / f"original{ext}", "original"
        else:
            try:
                await _stream_to(client, f"{base}/thumbnail", part, {"size": "fullsize"})
            except DownloadError as e:
                if e.status_code is None or e.status_code < 400 or e.status_code >= 500:
                    raise
                await _stream_to(client, f"{base}/thumbnail", part, {"size": "preview"})
            final, kind = job_dir / "original.jpg", "preview"
    os.replace(part, final)
    return final, kind


async def download_and_queue(job_id: str, asset_id: str, filename: str | None, focal: float | None) -> None:
    job_dir = INCOMING_DIR / job_id
    job: dict[str, Any] = {
        "id": job_id,
        "asset_id": asset_id,
        "status": "queued",
        "source": "immich-webhook",
        "original_filename": filename,
        "focal_length_mm": focal,
        "image_kind": None,
        "incoming_path": None,
        "created_at": now_iso(),
        "updated_at": now_iso(),
        "error": None,
    }
    try:
        path, kind = await fetch_asset(asset_id, filename, job_dir)
        job["incoming_path"] = str(path)
        job["image_kind"] = kind
        log.info("Job %s queued for asset %s (%s)", job_id, asset_id, kind)
    except Exception as e:  # any failure becomes an error job the user can see
        if isinstance(e, DownloadError):
            msg = str(e)
        elif isinstance(e, httpx.HTTPError):
            msg = f"download failed: {type(e).__name__}"
        else:
            log.exception("Unexpected error downloading asset %s", asset_id)
            msg = f"download failed: {type(e).__name__}"
        log.error("Job %s: %s (asset %s)", job_id, msg, asset_id)
        shutil.rmtree(job_dir, ignore_errors=True)
        job["status"] = "error"
        job["error"] = msg
        job["updated_at"] = now_iso()
    try:
        write_job(job)
    finally:
        _downloading.pop(asset_id, None)


# --- webhook -----------------------------------------------------------------

def parse_payload(payload: Any) -> tuple[str, str | None, str | None, float | None]:
    """Return (asset_id, type, filename, focal_length). Raises ValueError."""
    if not isinstance(payload, dict):
        raise ValueError("body must be a JSON object")
    data = payload.get("data")
    asset = data.get("asset") if isinstance(data, dict) else None
    if isinstance(asset, dict):
        raw_id = asset.get("id")
        asset_type = asset.get("type")
        filename = asset.get("originalFileName")
        exif = asset.get("exifInfo")
        focal = exif.get("focalLength") if isinstance(exif, dict) else None
    else:
        # bare body for manual curl tests: {"assetId": "..."}
        raw_id = payload.get("assetId")
        asset_type, filename, focal = None, None, None
    if not isinstance(raw_id, str):
        raise ValueError("no asset id in payload")
    try:
        asset_id = str(uuid.UUID(raw_id))
    except ValueError:
        raise ValueError("asset id is not a uuid")
    if not isinstance(filename, str):
        filename = None
    if isinstance(focal, bool) or not isinstance(focal, (int, float)):
        focal = None
    return asset_id, asset_type, filename, focal


@app.post("/webhook/immich")
async def immich_webhook(
    request: Request,
    background: BackgroundTasks,
    x_webhook_secret: str | None = Header(default=None),
):
    if not x_webhook_secret or not hmac.compare_digest(
        x_webhook_secret.encode(), WEBHOOK_SECRET.encode()
    ):
        log.warning("Webhook rejected: bad or missing X-Webhook-Secret")
        return JSONResponse({"error": "unauthorized"}, status_code=401)

    raw_body = await request.body()
    log.debug("Webhook raw body: %s", raw_body[:4000])
    try:
        payload = json.loads(raw_body) if raw_body else {}
        asset_id, asset_type, filename, focal = parse_payload(payload)
    except (json.JSONDecodeError, ValueError) as e:
        log.warning("Webhook rejected: %s", e)
        return JSONResponse({"error": str(e)}, status_code=400)

    log.info("Webhook: asset=%s type=%s file=%s", asset_id, asset_type, filename)

    if asset_type is not None and asset_type != "IMAGE":
        return {"skipped": f"asset type {asset_type} is not IMAGE"}

    existing = find_blocking_job(asset_id)
    if existing:
        return {"job_id": existing[0], "status": existing[1], "duplicate": True}

    job_id = str(uuid.uuid4())
    _downloading[asset_id] = job_id
    background.add_task(download_and_queue, job_id, asset_id, filename, focal)
    return JSONResponse({"job_id": job_id, "status": "downloading"}, status_code=202)


# --- API ---------------------------------------------------------------------

@app.get("/api/jobs")
async def api_jobs():
    return [with_viewer_url(j) for j in list_jobs()]


@app.get("/api/jobs/{job_id}")
async def api_job(job_id: str):
    job = read_job(parse_job_id(job_id))
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    return with_viewer_url(job)


@app.delete("/api/jobs/{job_id}")
async def api_delete_job(job_id: str):
    job_id = parse_job_id(job_id)
    job = read_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    if job.get("status") == "processing":
        raise HTTPException(status_code=409, detail="job is processing")
    _job_path(job_id).unlink(missing_ok=True)
    shutil.rmtree(INCOMING_DIR / job_id, ignore_errors=True)
    shutil.rmtree(SPLATS_DIR / job_id, ignore_errors=True)
    return {"deleted": job_id}


# --- pages -------------------------------------------------------------------

JOB_PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Splat job</title>
<style>
body{font-family:system-ui,sans-serif;background:#111;color:#eee;display:grid;place-items:center;min-height:100vh;margin:0;text-align:center}
.err{color:#f66}a{color:#8cf}
</style></head><body>
<main><h1 id="s">Waiting for photo...</h1><p id="d"></p><p><a href="/">All jobs</a></p></main>
<script>
const id = "__JOB_ID__";
const s = document.getElementById("s"), d = document.getElementById("d");
let misses = 0;
async function poll() {
  try {
    const r = await fetch("/api/jobs/" + id, {cache: "no-store"});
    if (r.status === 404) {
      misses++;
      s.textContent = misses > 30 ? "Job not found" : "Downloading photo from Immich...";
    } else if (r.ok) {
      const j = await r.json();
      if (j.status === "done" && j.viewer_url) { location.replace(j.viewer_url); return; }
      if (j.status === "error") {
        s.textContent = "Failed"; s.className = "err"; d.textContent = j.error || "";
        return;
      }
      s.textContent = j.status === "processing" ? "Generating splat..." : "Queued...";
    }
  } catch (e) { s.textContent = "Connection lost, retrying..."; }
  setTimeout(poll, 2000);
}
poll();
</script></body></html>
"""


@app.get("/jobs/{job_id}", response_class=HTMLResponse)
async def job_page(job_id: str):
    return JOB_PAGE.replace("__JOB_ID__", parse_job_id(job_id))


@app.get("/", response_class=HTMLResponse)
async def status_page(request: Request):
    return templates.TemplateResponse("status.html", {"request": request})


@app.get("/healthz", response_class=PlainTextResponse)
async def healthz():
    return "ok"
