"""
immich-splat worker

Polls /data/jobs for jobs with status "queued", runs Apple's ml-sharp
(`sharp predict`) on the downloaded image, and drops the resulting Gaussian
splat .ply into /data/splats/<job_id>/scene.ply.

Deliberately dependency-free (stdlib only) so it doesn't fight with
ml-sharp's own pinned requirements.
"""

import json
import logging
import os
import shutil
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("immich-splat-worker")

DATA_DIR = Path(os.environ.get("DATA_DIR", "/data"))
JOBS_DIR = DATA_DIR / "jobs"
INCOMING_DIR = DATA_DIR / "incoming"
SPLATS_DIR = DATA_DIR / "splats"
for d in (JOBS_DIR, INCOMING_DIR, SPLATS_DIR):
    d.mkdir(parents=True, exist_ok=True)

POLL_INTERVAL = float(os.environ.get("POLL_INTERVAL", "3"))
SHARP_TIMEOUT_SECONDS = int(os.environ.get("SHARP_TIMEOUT_SECONDS", "900"))
SHARP_EXTRA_ARGS = os.environ.get("SHARP_EXTRA_ARGS", "").split()


def _job_path(job_id: str) -> Path:
    return JOBS_DIR / f"{job_id}.json"


def write_job(job: dict) -> None:
    job["updated_at"] = datetime.now(timezone.utc).isoformat()
    path = _job_path(job["id"])
    fd, tmp_path = tempfile.mkstemp(dir=str(JOBS_DIR), suffix=".tmp")
    with os.fdopen(fd, "w") as f:
        json.dump(job, f, indent=2)
    os.replace(tmp_path, path)


def read_job(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def find_queued_jobs() -> list[dict]:
    jobs = []
    for path in JOBS_DIR.glob("*.json"):
        job = read_job(path)
        if job and job.get("status") == "queued":
            jobs.append(job)
    return jobs


def run_sharp(job: dict) -> None:
    job_id = job["id"]
    input_dir = Path(job["incoming_path"]).parent  # sharp wants a directory of images
    output_dir = SPLATS_DIR / job_id / "raw"
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = ["sharp", "predict", "-i", str(input_dir), "-o", str(output_dir), *SHARP_EXTRA_ARGS]
    log.info("Job %s: running %s", job_id, " ".join(cmd))

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=SHARP_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        job["status"] = "error"
        job["error"] = f"sharp predict timed out after {SHARP_TIMEOUT_SECONDS}s"
        write_job(job)
        log.error("Job %s: timed out", job_id)
        return

    if result.returncode != 0:
        job["status"] = "error"
        job["error"] = (result.stderr or result.stdout or "sharp predict failed")[-2000:]
        write_job(job)
        log.error("Job %s: sharp predict failed (rc=%s)\n%s", job_id, result.returncode, result.stderr)
        return

    ply_files = sorted(output_dir.rglob("*.ply"))
    if not ply_files:
        job["status"] = "error"
        job["error"] = "sharp predict succeeded but produced no .ply file"
        write_job(job)
        log.error("Job %s: no .ply output found in %s", job_id, output_dir)
        return

    final_dir = SPLATS_DIR / job_id
    final_dir.mkdir(parents=True, exist_ok=True)
    dest = final_dir / "scene.ply"
    shutil.copy2(ply_files[0], dest)

    job["status"] = "done"
    job["ply_url"] = f"/splats/{job_id}/scene.ply"
    job["error"] = None
    write_job(job)
    log.info("Job %s: done -> %s", job_id, dest)


def main() -> None:
    log.info("immich-splat worker started, polling %s every %ss", JOBS_DIR, POLL_INTERVAL)
    while True:
        for job in find_queued_jobs():
            job["status"] = "processing"
            write_job(job)
            log.info("Job %s: processing asset %s", job["id"], job.get("asset_id"))
            try:
                run_sharp(job)
            except Exception as e:  # keep the loop alive no matter what
                job["status"] = "error"
                job["error"] = f"unexpected worker error: {e}"
                write_job(job)
                log.exception("Job %s: unexpected error", job["id"])
        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()
