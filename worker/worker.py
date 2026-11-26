"""
immich-splat worker

Polls /data/jobs for queued jobs. The ml-sharp model is loaded once at startup;
each job runs it in-process, converts the PLY to .spz with splat-transform
(rotated 180 deg about X so the scene is y-up), and writes the per-job viewer
settings.json and thumb.jpg.
"""

import json
import logging
import math
import os
import shutil
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pillow_heif
import torch
from PIL import Image

from sharp.cli.predict import DEFAULT_MODEL_URL, predict_image
from sharp.models import PredictorParams, create_predictor
from sharp.utils import io
from sharp.utils.gaussians import save_ply

pillow_heif.register_heif_opener()  # so Pillow can open .heif/.heic too

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("immich-splat-worker")

DATA_DIR = Path(os.environ.get("DATA_DIR", "/data"))
JOBS_DIR = DATA_DIR / "jobs"
INCOMING_DIR = DATA_DIR / "incoming"
SPLATS_DIR = DATA_DIR / "splats"

POLL_INTERVAL = float(os.environ.get("POLL_INTERVAL", "3"))
KEEP_PLY = os.environ.get("KEEP_PLY", "false").lower() in ("1", "true", "yes")
SHARP_DEVICE = os.environ.get("SHARP_DEVICE", "default")
SPLAT_TRANSFORM_TIMEOUT = 600
THUMB_SIZE = 512


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_job(job: dict) -> None:
    """Atomic write: temp file in the same dir, then replace."""
    job["updated_at"] = now()
    fd, tmp = tempfile.mkstemp(dir=str(JOBS_DIR), suffix=".tmp")
    with os.fdopen(fd, "w") as f:
        json.dump(job, f, indent=2)
    os.replace(tmp, JOBS_DIR / f"{job['id']}.json")


def read_job(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def jobs_with_status(status: str) -> list[dict]:
    jobs = []
    for path in sorted(JOBS_DIR.glob("*.json")):
        job = read_job(path)
        if job and job.get("status") == status:
            jobs.append(job)
    jobs.sort(key=lambda j: j.get("created_at", ""))
    return jobs


def load_model(device: torch.device):
    log.info("Loading model (downloads ~1.2GB on first run into %s)", os.environ.get("TORCH_HOME"))
    state_dict = torch.hub.load_state_dict_from_url(DEFAULT_MODEL_URL, progress=False)
    predictor = create_predictor(PredictorParams())
    predictor.load_state_dict(state_dict)
    predictor.eval()
    predictor.to(device)
    return predictor


def focal_px_from_mm(width: int, height: int, focal_mm: float) -> float:
    """Same rule as sharp.utils.io.load_rgb when EXIF has no 35mm focal length."""
    if focal_mm < 10.0:
        focal_mm *= 8.4  # crude phone-sensor -> 35mm approximation, as in ml-sharp
    return float(io.convert_focallength(width, height, focal_mm))


def make_settings(f_px: float, width: int, height: int, depth: float) -> dict:
    """SuperSplat viewer settings v2: camera at origin looking at the scene centre.

    The viewer applies `fov` to the longer viewport axis (supersplat-viewer
    src/viewer.ts: horizontalFov = width > height), so use the longer image side.
    """
    fov = math.degrees(2 * math.atan(max(width, height) / (2 * f_px)))
    fov = min(max(fov, 10.0), 120.0)  # authoring range in schemas/ranges.ts
    return {
        "version": 2,
        "tonemapping": "none",
        "highPrecisionRendering": False,
        "background": {"color": [0, 0, 0]},
        "postEffectSettings": {
            "sharpness": {"enabled": False, "amount": 0},
            "bloom": {"enabled": False, "intensity": 0.1, "blurLevel": 2},
            "grading": {"enabled": False, "brightness": 1, "contrast": 1, "saturation": 1, "tint": [1, 1, 1]},
            "vignette": {"enabled": False, "intensity": 0.5, "inner": 0.3, "outer": 0.75, "curvature": 1},
            "fringing": {"enabled": False, "intensity": 0.5},
        },
        "animTracks": [],
        "cameras": [
            {"initial": {"position": [0, 0, 0], "target": [0, 0, -depth], "fov": round(fov, 2)}}
        ],
        "annotations": [],
        "startMode": "default",
    }


def write_thumb(image: np.ndarray, path: Path) -> None:
    img = Image.fromarray(image).convert("RGB")  # already EXIF-rotated by load_rgb
    img.thumbnail((THUMB_SIZE, THUMB_SIZE), Image.LANCZOS)
    img.save(path, "JPEG", quality=85)


def run_splat_transform(ply: Path, spz: Path) -> None:
    # -r 180,0,0: rotate 180 deg about X (OpenCV y-down/z-forward -> y-up, scene at -z)
    cmd = ["splat-transform", "-w", str(ply), "-r", "180,0,0", str(spz)]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=SPLAT_TRANSFORM_TIMEOUT)
    if result.returncode != 0:
        raise RuntimeError("splat-transform failed: " + (result.stderr or result.stdout)[-500:])


def process(job: dict, predictor, device: torch.device) -> None:
    job_id = job["id"]
    image_path = Path(job["incoming_path"])
    out_dir = SPLATS_DIR / job_id
    out_dir.mkdir(parents=True, exist_ok=True)

    image, _, f_px = io.load_rgb(image_path)
    height, width = image.shape[:2]
    if job.get("image_kind") == "preview" and job.get("focal_length_mm"):
        f_px = focal_px_from_mm(width, height, float(job["focal_length_mm"]))
    log.info("Job %s: %dx%d f_px=%.1f", job_id, width, height, f_px)

    gaussians = predict_image(predictor, image, f_px, device)
    depth = float(gaussians.mean_vectors[..., 2].flatten().median().item())  # OpenCV z, forward
    count = int(gaussians.mean_vectors.shape[0] * gaussians.mean_vectors.shape[1])

    ply = out_dir / "scene.ply"
    save_ply(gaussians, f_px, (height, width), ply)
    run_splat_transform(ply, out_dir / "scene.spz")
    if not KEEP_PLY:
        ply.unlink(missing_ok=True)

    (out_dir / "settings.json").write_text(json.dumps(make_settings(f_px, width, height, depth), indent=2))
    write_thumb(image, out_dir / "thumb.jpg")

    shutil.rmtree(INCOMING_DIR / job_id, ignore_errors=True)

    job.update(
        status="done",
        error=None,
        gaussians=count,
        thumb_url=f"/splats/{job_id}/thumb.jpg",
        splat_url=f"/splats/{job_id}/scene.spz",
        settings_url=f"/splats/{job_id}/settings.json",
    )
    if KEEP_PLY:
        job["ply_url"] = f"/splats/{job_id}/scene.ply"


def run_job(job: dict, predictor, device: torch.device) -> None:
    started = time.monotonic()
    job["status"] = "processing"
    job["started_at"] = now()
    job["error"] = None
    write_job(job)
    log.info("Job %s: processing asset %s", job["id"], job.get("asset_id"))
    try:
        process(job, predictor, device)
    except Exception as e:  # keep the loop alive no matter what
        log.exception("Job %s failed", job["id"])
        shutil.rmtree(SPLATS_DIR / job["id"], ignore_errors=True)
        job["status"] = "error"
        job["error"] = f"{type(e).__name__}: {e}"[:300]
    job["finished_at"] = now()
    job["duration_s"] = round(time.monotonic() - started, 2)
    write_job(job)
    log.info("Job %s: %s in %.1fs", job["id"], job["status"], job["duration_s"])


def main() -> None:
    for d in (JOBS_DIR, INCOMING_DIR, SPLATS_DIR):
        d.mkdir(parents=True, exist_ok=True)

    # Crash recovery: anything left in "processing" goes back in the queue.
    for job in jobs_with_status("processing"):
        log.warning("Job %s was processing at startup, re-queueing", job["id"])
        job["status"] = "queued"
        write_job(job)

    if SHARP_DEVICE == "default":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(SHARP_DEVICE)
    log.info("Using device %s", device)
    predictor = load_model(device)

    log.info("Worker ready, polling %s every %ss", JOBS_DIR, POLL_INTERVAL)
    while True:
        try:
            queued = jobs_with_status("queued")
            for job in queued:
                run_job(job, predictor, device)
            if not queued:
                time.sleep(POLL_INTERVAL)
        except Exception:
            log.exception("Poll loop error")
            time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()
