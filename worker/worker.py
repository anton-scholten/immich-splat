"""
immich-splat worker

Polls /data/jobs for queued jobs. The ml-sharp model is loaded once at startup;
each job runs it in-process, converts the PLY to .spz with splat-transform
(rotated 180 deg about X so the scene is y-up), and writes the per-job viewer
settings.json. A background thread writes /data/status/worker.json every 2 s.
"""

import gc
import json
import logging
import math
import os
import shutil
import subprocess
import tempfile
import threading
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import psutil
import pillow_heif
import torch
import torch.nn.functional as F
from PIL import Image

from sharp.cli.predict import DEFAULT_MODEL_URL
from sharp.models import PredictorParams, create_predictor
from sharp.utils import io
from sharp.utils.gaussians import Gaussians3D, save_ply, unproject_gaussians

pillow_heif.register_heif_opener()  # so Pillow can open .heif/.heic too

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("immich-splat-worker")

DATA_DIR = Path(os.environ.get("DATA_DIR", "/data"))
JOBS_DIR = DATA_DIR / "jobs"
INCOMING_DIR = DATA_DIR / "incoming"
SPLATS_DIR = DATA_DIR / "splats"
STATUS_DIR = DATA_DIR / "status"
CACHE_DIR = Path("/cache")

POLL_INTERVAL = float(os.environ.get("POLL_INTERVAL", "3"))
KEEP_PLY = os.environ.get("KEEP_PLY", "false").lower() in ("1", "true", "yes")
SHARP_DEVICE = os.environ.get("SHARP_DEVICE", "default")
SHARP_PRECISION = os.environ.get("SHARP_PRECISION", "auto").lower()  # auto | fp32 | fp32-offload | fp16
SPLAT_TRANSFORM_TIMEOUT = 600
HEARTBEAT_INTERVAL = 2
STAGES = ("loading", "predicting", "saving", "converting")

# Shared between the main thread (jobs) and the heartbeat thread (status file).
state_lock = threading.Lock()
status = {"state": "loading_model", "current_job": None, "device": "cpu"}
recent = deque(maxlen=10)  # last 10 successful jobs: {"total": s, "stages": {name: s}}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_job(job: dict) -> None:
    """Atomic write: temp file in the same dir, then replace."""
    job["updated_at"] = now()
    write_json_atomic(JOBS_DIR / f"{job['id']}.json", job)


def write_json_atomic(path: Path, data: dict) -> None:
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")  # unique name per call
    with os.fdopen(fd, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


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


GIB = 1024**3
# Measured with ml-sharp's fixed 1536x1536 input (PyTorch memory, plus ~0.25 GiB CUDA overhead):
#   fp32          weights 2.6 GiB, peak 7.5 GiB    fastest
#   fp32-offload  weights 0.2 GiB, needs ~4.2 GiB  the two ViT encoders (650M of 700M params)
#                 stay in RAM and are copied in only while they run; ~10% slower
#   fp16          weights 1.4 GiB, needs ~3.4 GiB  last GPU option; NaN on GTX 16xx
MODES = ("fp32", "fp32-offload", "fp16", "cpu")
FP32_MIN_VRAM = 10 * GIB
OFFLOAD_MIN_VRAM = 4.6 * GIB


class Model:
    """The ml-sharp predictor and the mode it runs in (see MODES)."""

    def __init__(self, device: torch.device):
        self.device = device
        self.mode = self._initial_mode()
        self.tried: set[str] = set()
        self._load()

    def _initial_mode(self) -> str:
        if self.device.type != "cuda":
            return "cpu"
        if SHARP_PRECISION in MODES[:3]:
            return SHARP_PRECISION
        total = torch.cuda.get_device_properties(self.device).total_memory
        mode = "fp32" if total >= FP32_MIN_VRAM else "fp32-offload" if total >= OFFLOAD_MIN_VRAM else "fp16"
        log.info("GPU has %.1f GiB; starting in %s (override with SHARP_PRECISION)", total / GIB, mode)
        return mode

    @property
    def label(self) -> str:
        """What the job ran on, e.g. "cuda-fp32-offload" or "cpu"."""
        return "cpu-fp32" if self.mode == "cpu" else f"cuda-{self.mode}"

    def _load(self) -> None:
        self.predictor = None
        free_gpu_memory()
        log.info("Loading model (downloads ~1.2GB on first run into %s)", os.environ.get("TORCH_HOME"))
        state_dict = torch.hub.load_state_dict_from_url(DEFAULT_MODEL_URL, progress=False)
        predictor = create_predictor(PredictorParams())
        predictor.load_state_dict(state_dict)
        del state_dict
        predictor.eval()
        if self.mode == "cpu":
            self.device = torch.device("cpu")
            with state_lock:
                status["device"] = "cpu"
        elif self.mode == "fp16":
            if is_gtx16(self.device):
                log.info("GTX 16xx GPU: enabling the known fp16 NaN workarounds")
                enable_fp16_safety()
            predictor.half()  # on the CPU, so the fp32 copy never hits the GPU
            predictor.register_forward_pre_hook(
                lambda mod, args: tuple(a.half() if torch.is_tensor(a) and a.is_floating_point() else a for a in args)
            )
            upcast_attention()
        elif self.mode == "fp32-offload":
            enc = predictor.monodepth_model.monodepth_predictor.encoder
            for vit in (enc.patch_encoder, enc.image_encoder):
                offload_to_ram(vit, self.device)
        predictor.to(self.device)  # offloaded parts ignore this: their hooks keep them in RAM
        self.predictor = predictor
        self.tried.add(self.mode)
        log.info("Model ready: %s", self.label)

    def next_fallback(self, problem: str) -> bool:
        """Pick the next mode after a failed prediction; True if there is one to try.

        "oom": move one step down MODES.  "nan" in fp16: first retry with the fp16
        workarounds, then fp32-offload (if not yet tried), then CPU. The new mode is kept
        for later jobs, so a GPU that can't do it only costs one failed try.
        """
        if self.mode == "cpu":
            return False
        if problem == "nan" and self.mode == "fp16" and not fp16_safety_on():
            enable_fp16_safety()
            log.warning("fp16 gave NaN/inf on this GPU; retrying with the fp16 workarounds on")
            return True
        if problem == "oom":
            nxt = MODES[MODES.index(self.mode) + 1]
        else:
            nxt = next((m for m in ("fp32-offload", "cpu") if m not in self.tried), "cpu")
        reason = "not enough GPU memory" if problem == "oom" else "NaN/inf results"
        log.warning("%s in %s; switching to %s for this and later photos", reason, self.mode, nxt)
        self.mode = nxt
        self._load()
        return True

    @torch.no_grad()
    def predict(self, image, f_px: float) -> Gaussians3D:
        """Same steps as sharp.cli.predict.predict_image, except the model output is
        converted to fp32 BEFORE unprojection. ml-sharp's unprojection squares the
        Gaussian scales in the input dtype, which overflows fp16 for far-away content
        (inf -> "linalg.svd ... non-finite values")."""
        size = 1536  # ml-sharp's fixed internal resolution
        dev = self.device
        x = torch.from_numpy(image.copy()).float().to(dev).permute(2, 0, 1) / 255.0
        _, height, width = x.shape
        disparity_factor = torch.tensor([f_px / width]).float().to(dev)
        x = F.interpolate(x[None], size=(size, size), mode="bilinear", align_corners=True)

        ndc = self.predictor(x, disparity_factor)
        del x
        ndc = Gaussians3D(*[t.float() for t in ndc])
        ndc = drop_nonfinite(ndc, "model output")

        intrinsics = torch.tensor(
            [[f_px, 0, width / 2, 0], [0, f_px, height / 2, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
            dtype=torch.float32, device=dev,
        )
        intrinsics[0] *= size / width
        intrinsics[1] *= size / height
        gaussians = unproject_gaussians(ndc, torch.eye(4, device=dev), intrinsics, (size, size))
        return drop_nonfinite(Gaussians3D(*[t.float() for t in gaussians]), "unprojection")


MAX_DROPPED_FRACTION = 0.01


class NonFiniteOutput(RuntimeError):
    """Too many NaN/inf Gaussians: the result is unusable on this device/precision."""


def drop_nonfinite(g: Gaussians3D, where: str) -> Gaussians3D:
    """Remove Gaussians with any NaN/inf value instead of failing the whole job.
    A handful is harmless; more than 1% means the result is broken, so raise."""
    ok = None
    for t in g:
        finite = torch.isfinite(t.reshape(t.shape[0], t.shape[1], -1)).all(dim=-1)  # [batch, N]
        ok = finite if ok is None else ok & finite
    bad = int((~ok).sum())
    if bad == 0:
        return g
    total = ok.numel()
    if bad > MAX_DROPPED_FRACTION * total:
        raise NonFiniteOutput(f"{bad} of {total} Gaussians are NaN/inf after {where}; result unusable")
    log.warning("Dropping %d of %d non-finite Gaussians after %s", bad, total, where)
    keep = ok[0]  # batch size is 1
    return Gaussians3D(*[t[:, keep] for t in g])


def is_gtx16(device: torch.device) -> bool:
    """GTX 16xx (Turing without tensor cores) is known to give NaN in fp16
    (pytorch/pytorch#58123); same check as AUTOMATIC1111's stable-diffusion-webui."""
    return (torch.cuda.get_device_capability(device) == (7, 5)
            and torch.cuda.get_device_name(device).startswith("NVIDIA GeForce GTX 16"))


def enable_fp16_safety() -> None:
    """cuDNN benchmark mode lets these cards pick fp16 conv kernels that work
    (stable-diffusion-webui PR 4407); fp32 accumulation keeps big matmul sums finite.
    Costs about 0.06 GiB extra and no speed."""
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False


def fp16_safety_on() -> bool:
    return torch.backends.cudnn.benchmark and not torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction


_sdpa = F.scaled_dot_product_attention


def _sdpa_fp32(q, k, v, *args, **kwargs):
    if q.dtype == torch.float16:
        return _sdpa(q.float(), k.float(), v.float(), *args, **kwargs).half()
    return _sdpa(q, k, v, *args, **kwargs)


def upcast_attention() -> None:
    """Run attention (the ViT encoders' softmax) in fp32 while the weights stay fp16.
    Measured: +0.12 GiB peak, and faster on Pascal, whose fp16 math is slow."""
    F.scaled_dot_product_attention = _sdpa_fp32


def offload_to_ram(module: torch.nn.Module, device: torch.device) -> None:
    """Keep this module's weights in (pinned) RAM; copy them to the GPU only for its
    forward pass and drop the GPU copy right after. No copy back: weights don't change."""
    tensors = list(module.named_parameters()) + list(module.named_buffers())
    ram = {}
    for name, t in tensors:
        data = t.data.float().cpu()
        try:
            data = data.pin_memory()  # faster copies; needs locked memory, skip if refused
        except RuntimeError:
            pass
        ram[name] = data
        t.data = data

    def to_gpu(mod, args):
        for name, t in tensors:
            t.data = ram[name].to(device, non_blocking=True)

    def to_ram(mod, args, out):
        for name, t in tensors:
            t.data = ram[name]

    module.register_forward_pre_hook(to_gpu)
    module.register_forward_hook(to_ram)
    module._apply = lambda fn, *a, **k: module  # predictor.to(device) must not move it


def free_gpu_memory() -> None:
    """Give cached GPU memory back so idle VRAM is just the model weights."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def exif_has_focal(path: Path) -> bool:
    """True if the file's EXIF has a focal length that load_rgb would use."""
    try:
        with Image.open(path) as im:
            exif = io.extract_exif(im)
    except Exception:
        return False
    return any(exif.get(k) for k in ("FocalLengthIn35mmFilm", "FocalLenIn35mmFilm", "FocalLength"))


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


def seed_recent() -> None:
    """After a restart, seed the duration average from existing done jobs (no per-stage data)."""
    done = [j for j in jobs_with_status("done") if j.get("duration_s")]
    done.sort(key=lambda j: j.get("finished_at") or "")
    for j in done[-recent.maxlen:]:
        recent.append({"total": float(j["duration_s"]), "stages": {}})


def averages() -> tuple[float | None, dict]:
    with state_lock:
        items = list(recent)
    avg = round(sum(r["total"] for r in items) / len(items), 2) if items else None
    stage_avg = {}
    for name in STAGES:
        vals = [r["stages"][name] for r in items if name in r["stages"]]
        stage_avg[name] = round(sum(vals) / len(vals), 2) if vals else None
    return avg, stage_avg


class GpuStats:
    """NVML when available, else torch.cuda.mem_get_info (util null). Never raises."""

    def __init__(self, device: torch.device):
        self.device = device
        self.nvml = None
        if device.type != "cuda":
            return
        try:
            import pynvml

            pynvml.nvmlInit()
            index = device.index or 0
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            self.nvml = pynvml
        except Exception as e:
            log.warning("NVML unavailable (%s), falling back to torch.cuda.mem_get_info", e)

    def read(self) -> dict | None:
        if self.device.type != "cuda":
            return None
        try:
            if self.nvml:
                name = self.nvml.nvmlDeviceGetName(self.handle)
                if isinstance(name, bytes):
                    name = name.decode()
                util = self.nvml.nvmlDeviceGetUtilizationRates(self.handle)
                mem = self.nvml.nvmlDeviceGetMemoryInfo(self.handle)
                return {"name": name, "util_percent": int(util.gpu),
                        "mem_used_bytes": int(mem.used), "mem_total_bytes": int(mem.total)}
            free, total = torch.cuda.mem_get_info(self.device)
            return {"name": torch.cuda.get_device_name(self.device), "util_percent": None,
                    "mem_used_bytes": int(total - free), "mem_total_bytes": int(total)}
        except Exception:
            log.exception("GPU stats failed")
            return None


def dir_size(path: Path) -> int:
    total = 0
    for root, _, files in os.walk(path):
        for name in files:
            try:
                total += os.lstat(os.path.join(root, name)).st_size
            except OSError:
                pass
    return total


def heartbeat_loop(device: torch.device) -> None:
    """Write /data/status/worker.json every 2 s. Must never die."""
    gpu = GpuStats(device)
    proc = psutil.Process()
    proc.cpu_percent(None)  # prime; later calls measure since the previous call
    queue_length, queue_at = 0, 0.0
    cache_bytes, cache_at = None, 0.0
    while True:
        try:
            t = time.monotonic()
            if t - queue_at >= 5:
                queue_length, queue_at = len(jobs_with_status("queued")), t
            if cache_bytes is None or t - cache_at >= 60:
                cache_bytes, cache_at = dir_size(CACHE_DIR), t
            avg, stage_avg = averages()
            with state_lock:
                snap = dict(status)
            write_json_atomic(STATUS_DIR / "worker.json", {
                "updated_at": now(),
                "state": snap["state"],
                "current_job": snap["current_job"],
                "queue_length": queue_length,
                "device": device.type,
                "gpu": gpu.read(),
                "process": {"cpu_percent": proc.cpu_percent(None), "rss_bytes": proc.memory_info().rss},
                "avg_duration_s": avg,
                "stage_avg_s": stage_avg,
                "model_cache_bytes": cache_bytes,
            })
        except Exception:
            log.exception("Heartbeat error")
        time.sleep(HEARTBEAT_INTERVAL)


def set_stage(job: dict, name: str, timings: dict, marks: dict) -> None:
    """Close the previous stage's timing, start `name`, and write the job."""
    t = time.monotonic()
    if marks.get("name"):
        timings[marks["name"]] = t - marks["t"]
    marks["name"], marks["t"] = name, t
    job["stage"] = name
    job["stage_started_at"] = now()
    write_job(job)


def run_splat_transform(ply: Path, spz: Path) -> None:
    # -r 180,0,0: rotate 180 deg about X (OpenCV y-down/z-forward -> y-up, scene at -z)
    cmd = ["splat-transform", "-w", str(ply), "-r", "180,0,0", str(spz)]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=SPLAT_TRANSFORM_TIMEOUT)
    if result.returncode != 0:
        raise RuntimeError("splat-transform failed: " + (result.stderr or result.stdout)[-500:])


def predict_with_fallback(model: Model, image, f_px: float):
    """Predict, stepping down GPU precision / to the CPU on out-of-memory or NaN results."""
    while True:
        try:
            return model.predict(image, f_px)
        except torch.OutOfMemoryError:
            problem = "oom"  # handled below, outside the except block, so its tensors are freed
        except NonFiniteOutput:
            problem = "nan"
        free_gpu_memory()
        if not model.next_fallback(problem):
            if problem == "oom":
                raise RuntimeError("Not enough memory to run the model, even on the CPU (needs about 8 GB RAM).")
            raise RuntimeError("The model produced NaN/inf values even on the CPU; this photo can't be converted.")


def process(job: dict, model: Model, timings: dict) -> None:
    job_id = job["id"]
    image_path = Path(job["incoming_path"])
    out_dir = SPLATS_DIR / job_id
    out_dir.mkdir(parents=True, exist_ok=True)
    marks: dict = {}

    set_stage(job, "loading", timings, marks)
    image, _, f_px = io.load_rgb(image_path)
    height, width = image.shape[:2]
    # load_rgb falls back to 30mm when the file has no focal length (Immich previews
    # never do). Immich's own EXIF value is better than that guess.
    if job.get("focal_length_mm") and (job.get("image_kind") == "preview" or not exif_has_focal(image_path)):
        f_px = focal_px_from_mm(width, height, float(job["focal_length_mm"]))
    job["width"], job["height"] = width, height
    log.info("Job %s: %dx%d f_px=%.1f", job_id, width, height, f_px)

    set_stage(job, "predicting", timings, marks)
    gaussians = predict_with_fallback(model, image, f_px)
    job["precision"] = model.label
    depth = float(gaussians.mean_vectors[..., 2].flatten().median().item())  # OpenCV z, forward
    count = int(gaussians.mean_vectors.shape[0] * gaussians.mean_vectors.shape[1])

    set_stage(job, "saving", timings, marks)
    ply = out_dir / "scene.ply"
    save_ply(gaussians, f_px, (height, width), ply)

    set_stage(job, "converting", timings, marks)
    spz = out_dir / "scene.spz"
    run_splat_transform(ply, spz)
    if not KEEP_PLY:
        ply.unlink(missing_ok=True)
    (out_dir / "settings.json").write_text(json.dumps(make_settings(f_px, width, height, depth), indent=2))
    timings[marks["name"]] = time.monotonic() - marks["t"]

    shutil.rmtree(INCOMING_DIR / job_id, ignore_errors=True)

    job.update(
        status="done",
        error=None,
        gaussians=count,
        splat_bytes=spz.stat().st_size,
        splat_url=f"/splats/{job_id}/scene.spz",
        settings_url=f"/splats/{job_id}/settings.json",
    )
    if KEEP_PLY:
        job["ply_url"] = f"/splats/{job_id}/scene.ply"


def run_job(job: dict, model: Model) -> None:
    started = time.monotonic()
    job["status"] = "processing"
    job["started_at"] = now()
    job["error"] = None
    job["stage"] = None
    write_job(job)
    with state_lock:
        status.update(state="busy", current_job=job["id"])
    log.info("Job %s: processing asset %s", job["id"], job.get("asset_id"))
    timings: dict = {}
    try:
        process(job, model, timings)
    except Exception as e:  # keep the loop alive no matter what
        log.exception("Job %s failed", job["id"])
        shutil.rmtree(SPLATS_DIR / job["id"], ignore_errors=True)
        job["status"] = "error"
        job["error"] = f"{type(e).__name__}: {e}"[:300]
    free_gpu_memory()  # after the except block, so the failed job's tensors are released too
    job["finished_at"] = now()
    job["duration_s"] = round(time.monotonic() - started, 2)
    job.pop("stage", None)
    job.pop("stage_started_at", None)
    write_job(job)
    with state_lock:
        if job["status"] == "done":
            recent.append({"total": job["duration_s"], "stages": timings})
        status.update(state="idle", current_job=None)
    log.info("Job %s: %s in %.1fs", job["id"], job["status"], job["duration_s"])


def main() -> None:
    for d in (JOBS_DIR, INCOMING_DIR, SPLATS_DIR, STATUS_DIR):
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
    seed_recent()
    status["device"] = device.type
    threading.Thread(target=heartbeat_loop, args=(device,), daemon=True).start()
    model = Model(device)
    free_gpu_memory()
    with state_lock:
        status["state"] = "idle"

    log.info("Worker ready, polling %s every %ss", JOBS_DIR, POLL_INTERVAL)
    while True:
        try:
            queued = jobs_with_status("queued")
            for job in queued:
                run_job(job, model)
            if not queued:
                time.sleep(POLL_INTERVAL)
        except Exception:
            log.exception("Poll loop error")
            time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()
