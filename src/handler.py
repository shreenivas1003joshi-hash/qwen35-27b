"""
RunPod serverless handler for vLLM.

Startup sequence (once per worker lifetime):
  1. Spawn the vLLM OpenAI-compatible HTTP server as a child process,
     loading the model directly from config.yaml.
  2. Poll /health until vLLM is fully loaded and all CUDA kernels are compiled.
  3. Signal RunPod that this worker is ready → runpod.serverless.start().

Per-request:
  - Forward the job payload to the local vLLM server (localhost:8001).
  - Stream chunks back to RunPod as they arrive.
  - The model never reloads; it stays in VRAM for the entire worker lifetime.
"""

import sys
import json
import asyncio
import subprocess
import traceback
import os

import runpod
from runpod import RunPodLogger

log = RunPodLogger()

CONFIG_PATH = os.environ.get("CONFIG_PATH", "/app/src/config.yaml")
VLLM_PORT   = int(os.environ.get("VLLM_PORT", 8001))
VLLM_URL    = f"http://localhost:{VLLM_PORT}"
READY_TIMEOUT = int(os.environ.get("VLLM_READY_TIMEOUT", 900))  # seconds

# Set at startup once vLLM is launched; the handler rewrites the "model" field
# in every request to this value so the name always matches what vLLM registered.
_SERVED_MODEL_NAME: str = ""


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------

class _IncompleteDownloadError(Exception):
    """Raised when the model directory exists but shards are missing/empty."""
    def __init__(self, path: str, model_id: str | None):
        self.path     = path
        self.model_id = model_id
        super().__init__(f"Incomplete model at {path!r} (model_id={model_id!r})")


# ---------------------------------------------------------------------------
# Auto-download helper
# ---------------------------------------------------------------------------

def _dir_size_gb(path: str) -> float:
    """Return the total size of all files under *path* in GB (best-effort)."""
    total = 0
    try:
        for root, _, files in os.walk(path):
            for f in files:
                fp = os.path.join(root, f)
                try:
                    # Follow symlinks so we count the real blob size
                    total += os.path.getsize(fp)
                except OSError:
                    pass
    except OSError:
        pass
    return total / (1024 ** 3)


def _report_volume_usage(volume_root: str) -> None:
    """Log a top-level disk usage breakdown so the user can see what's taking space."""
    if not os.path.isdir(volume_root):
        return
    log.info(f"── Volume usage breakdown: {volume_root} ──")
    try:
        entries = sorted(os.scandir(volume_root), key=lambda e: e.name)
        for entry in entries:
            try:
                gb = _dir_size_gb(entry.path) if entry.is_dir(follow_symlinks=True) else (
                    entry.stat(follow_symlinks=True).st_size / (1024 ** 3)
                )
                log.info(f"  {gb:7.2f} GB  {entry.name}")
            except OSError:
                log.info(f"  {'?':>7}      {entry.name}")
    except OSError as exc:
        log.warn(f"Could not scan {volume_root}: {exc}")
    log.info("────────────────────────────────────────────")


def _purge_wrong_path_cache(hf_home: str) -> None:
    """
    Delete stale cache directories that the old buggy code created at the wrong path.

    When cache_dir=HF_HOME was passed to snapshot_download (instead of HF_HOME/hub/),
    huggingface_hub wrote blobs to  HF_HOME/models--{name}/blobs/  instead of
    HF_HOME/hub/models--{name}/blobs/.  After multiple failed retries these accumulate
    and silently consume tens of GB of quota.

    This function removes every  HF_HOME/models--*/  directory that sits directly
    under HF_HOME (not under HF_HOME/hub/).
    """
    import shutil as _shutil
    if not os.path.isdir(hf_home):
        return
    freed = 0.0
    try:
        for entry in os.scandir(hf_home):
            if entry.is_dir() and entry.name.startswith("models--"):
                gb = _dir_size_gb(entry.path)
                log.info(
                    f"Removing stale wrong-path cache: {entry.path}  ({gb:.2f} GB)"
                )
                try:
                    _shutil.rmtree(entry.path)
                    freed += gb
                except Exception as exc:
                    log.warn(f"Could not remove {entry.path}: {exc}")
    except OSError as exc:
        log.warn(f"Could not scan {hf_home} for stale caches: {exc}")
    if freed > 0:
        log.info(f"Freed {freed:.2f} GB from wrong-path cache directories.")


def _purge_incomplete_snapshot(snapshot_path: str) -> None:
    """
    Delete a broken/incomplete HuggingFace snapshot directory and its blobs.

    When a snapshot download is cut short (disk quota), a partial snapshot
    directory is left behind containing symlinks to blobs.  Those blobs consume
    real disk space and must be removed before a fresh download will fit.

    We delete:
      • The snapshot directory itself  (symlinks → freed)
      • Any blobs referenced ONLY by this snapshot (unreferenced after deletion)
        i.e. files in  …/blobs/  whose link count drops to 1 (just the blob itself).

    Everything is best-effort; errors are logged but do not abort startup.
    """
    import shutil as _shutil

    if not snapshot_path or not os.path.isdir(snapshot_path):
        return

    log.info(f"Purging incomplete snapshot to free disk space: {snapshot_path}")

    # Collect all blob paths (the real files that symlinks point to)
    blobs_to_check: list[str] = []
    try:
        for entry in os.scandir(snapshot_path):
            if entry.is_symlink():
                real = os.path.realpath(entry.path)
                if os.path.isfile(real):
                    blobs_to_check.append(real)
    except Exception as exc:
        log.warn(f"Could not scan snapshot dir: {exc}")

    # Remove the snapshot directory (this unlinks all symlinks inside)
    try:
        _shutil.rmtree(snapshot_path)
        log.info(f"Removed snapshot dir: {snapshot_path}")
    except Exception as exc:
        log.warn(f"Could not remove snapshot dir: {exc}")
        return  # don't try to remove blobs if rmtree failed

    # Remove orphaned blobs (link count == 1 means nothing else references them)
    freed = 0
    for blob in blobs_to_check:
        try:
            if os.path.exists(blob) and os.stat(blob).st_nlink == 1:
                size = os.path.getsize(blob)
                os.remove(blob)
                freed += size
        except Exception as exc:
            log.warn(f"Could not remove blob {blob}: {exc}")

    freed_gb = freed / (1024 ** 3)
    log.info(f"Freed {freed_gb:.2f} GB by removing orphaned blobs.")


def _check_write_quota(path: str) -> bool:
    """
    Return True if we can actually write to *path*.

    statvfs().f_bavail on NFS / RunPod network volumes reports the total pool
    size, not the per-volume quota — it can show hundreds of TB even when the
    user's 50 GB quota is exhausted.  Writing a tiny test file is the only
    reliable way to detect EDQUOT (errno 122) early.
    """
    os.makedirs(path, exist_ok=True)
    test = os.path.join(path, ".quota_check")
    try:
        with open(test, "w") as f:
            f.write("x")
        os.remove(test)
        return True
    except OSError as exc:
        import errno as _errno
        if exc.errno == _errno.EDQUOT:
            log.error(
                "DISK QUOTA EXCEEDED on the RunPod network volume.\n"
                "The volume is full (statvfs shows the shared pool size, not your quota).\n"
                "Fix: RunPod dashboard → Storage → increase the volume size, then redeploy."
            )
        else:
            log.error(f"Cannot write to {path!r}: {exc}")
        return False


def download_model_to_volume(model_id: str) -> None:
    """
    Download (or resume) a HuggingFace model to the network volume.

    cache_dir is set to $HF_HOME/hub/ so the layout matches what vLLM expects
    when HF_HOME is set as an environment variable:
        $HF_HOME/hub/models--{org}--{name}/snapshots/{hash}/
    Passing cache_dir=$HF_HOME (without hub/) would create a *different* path
    and none of the already-downloaded blobs would be reused.

    snapshot_download is idempotent — files already present are skipped.
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        log.error("huggingface_hub is not installed — cannot auto-download.")
        sys.exit(1)

    hf_home  = os.environ.get("HF_HOME", "/runpod-volume/huggingface-cache")
    hub_cache = os.path.join(hf_home, "hub")   # must match HF_HOME convention
    token    = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    # Verify we can actually write before attempting a multi-GB download
    if not _check_write_quota(hub_cache):
        sys.exit(1)

    # Report usable space via shutil (more accurate than statvfs on quota filesystems)
    try:
        import shutil as _shutil
        total, used, free = _shutil.disk_usage(hub_cache)
        free_gb  = free  / (1024 ** 3)
        total_gb = total / (1024 ** 3)
        log.info(
            f"Volume space: {free_gb:.1f} GB free / {total_gb:.1f} GB total  "
            f"(cache: {hub_cache})"
        )
        if free_gb < 20:
            log.warn(
                f"Only {free_gb:.1f} GB free — large models may fail mid-download. "
                f"Consider increasing the volume size in the RunPod dashboard."
            )
    except Exception:
        pass

    log.info(
        f"Starting snapshot_download for {model_id!r}  →  cache_dir={hub_cache}\n"
        f"Only missing files will be fetched (resume-safe)."
    )

    kwargs: dict = dict(
        repo_id         = model_id,
        ignore_patterns = ["*.pt", "*.bin", "original/*"],
        cache_dir       = hub_cache,
    )
    if token:
        kwargs["token"] = token

    try:
        local_dir = snapshot_download(**kwargs)
        log.info(f"Download complete → {local_dir}")
    except OSError as exc:
        import errno as _errno
        if exc.errno == _errno.EDQUOT:
            log.error(
                "Download failed: disk quota exceeded mid-transfer.\n"
                "Increase your RunPod volume size and redeploy — "
                "already-downloaded blobs will be reused automatically."
            )
        else:
            log.error(f"snapshot_download failed: {exc}")
        sys.exit(1)
    except Exception as exc:
        log.error(f"snapshot_download failed: {exc}")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Startup helpers
# ---------------------------------------------------------------------------

def available_gpus() -> int:
    """Return the number of CUDA GPUs visible to this process."""
    try:
        import torch
        n = torch.cuda.device_count()
        return n if n > 0 else 1
    except Exception:
        return 1


def resolve_tensor_parallel_size() -> int:
    """
    Determine the tensor-parallel-size to pass to vLLM:
      1. TENSOR_PARALLEL_SIZE env var (highest priority — set in RunPod endpoint config)
      2. Value from config.yaml
      3. Falls back to the number of available GPUs

    If the requested size exceeds available GPUs it is clamped automatically
    so the worker never crashes with a 'World size > available GPUs' error.
    """
    import yaml

    # 1. Explicit env override
    if os.environ.get("TENSOR_PARALLEL_SIZE"):
        requested = int(os.environ["TENSOR_PARALLEL_SIZE"])
    else:
        # 2. Read from config.yaml
        try:
            with open(CONFIG_PATH) as f:
                config = yaml.safe_load(f) or {}
            requested = int(config.get("tensor-parallel-size", 1))
        except Exception:
            requested = 1

    gpus = available_gpus()
    if requested > gpus:
        log.warn(
            f"tensor-parallel-size={requested} exceeds available GPUs ({gpus}). "
            f"Clamping to {gpus}."
        )
        return gpus

    return requested


def _has_config_json(path: str) -> bool:
    return os.path.isfile(os.path.join(path, "config.json"))


def _read_model_config(path: str) -> dict:
    try:
        with open(os.path.join(path, "config.json")) as f:
            import json
            return json.load(f)
    except Exception:
        return {}


def validate_model_dir(path: str, expected_model_id: str | None = None) -> None:
    """
    Validate the model directory before handing it to vLLM.

    Checks (in order):
      1. config.json is present.
      2. The model_type / architectures in config.json are logged so mismatches
         are immediately visible — catches 'wrong model in the volume' errors
         before the EngineCore crashes with an opaque weights mismatch.
      3. At least one weight shard (.safetensors / .bin) is present.
      4. No shard resolves to a zero-byte or broken-symlink blob (partial DL).
      5. Total size is sanity-checked: warns if < 1 GB (suspiciously small).
    """
    WEIGHT_EXTS = (".safetensors", ".bin", ".pt")

    if not _has_config_json(path):
        log.error(f"Model directory {path!r} has no config.json — path is wrong.")
        sys.exit(1)

    # ── Read and log the model identity from config.json ──────────────────
    cfg = _read_model_config(path)
    model_type    = cfg.get("model_type", "unknown")
    architectures = cfg.get("architectures", [])
    hf_name       = cfg.get("_name_or_path", cfg.get("name_or_path", ""))
    num_layers    = cfg.get("num_hidden_layers", cfg.get("num_layers", "?"))

    log.info(
        f"config.json  →  model_type={model_type!r}  "
        f"architectures={architectures}  "
        f"layers={num_layers}  "
        f"name_or_path={hf_name!r}"
    )

    if expected_model_id and hf_name and expected_model_id.lower() not in hf_name.lower():
        log.error(
            f"WRONG MODEL IN VOLUME.\n"
            f"  Expected : {expected_model_id}\n"
            f"  Found    : {hf_name}  (model_type={model_type!r})\n"
            f"  Path     : {path}\n"
            f"The volume contains a different model. "
            f"Update MODEL_PATH / HF_HOME to point at {expected_model_id}, "
            f"or update 'model:' in config.yaml to match what is in the volume."
        )
        sys.exit(1)

    # ── Shard index check (catches partial downloads immediately) ──────────
    # model.safetensors.index.json lists every shard the full model needs.
    # If it is present and some shards are missing, the download was cut short.
    import json as _json

    index_path = os.path.join(path, "model.safetensors.index.json")
    real_index = os.path.realpath(index_path)
    expected_shards: set[str] = set()

    if os.path.isfile(real_index):
        try:
            with open(real_index) as f:
                index_data = _json.load(f)
            expected_shards = set(index_data.get("weight_map", {}).values())
        except Exception as exc:
            log.warn(f"Could not read shard index: {exc}")

    if expected_shards:
        missing, bad_size = [], []
        for shard in sorted(expected_shards):
            shard_path = os.path.join(path, shard)
            real_shard = os.path.realpath(shard_path)
            if not os.path.exists(real_shard):
                missing.append(shard)
            elif os.path.getsize(real_shard) == 0:
                bad_size.append(shard)

        if missing or bad_size:
            problems = (
                [f"  MISSING : {s}" for s in missing]
                + [f"  EMPTY   : {s}" for s in bad_size]
            )
            log.warn(
                f"INCOMPLETE DOWNLOAD — {len(missing)} missing + {len(bad_size)} empty "
                f"shards out of {len(expected_shards)} total.\n"
                + "\n".join(problems)
            )
            # Signal caller that the model needs (re-)downloading
            raise _IncompleteDownloadError(path, expected_model_id)

    # ── Check whatever weight files ARE present ────────────────────────────
    entries      = os.listdir(path)
    weight_files = [f for f in entries if any(f.endswith(e) for e in WEIGHT_EXTS)]

    if not weight_files:
        log.warn(f"No weight files (.safetensors / .bin) found in {path!r} — will download.")
        raise _IncompleteDownloadError(path, expected_model_id)

    bad = []
    total_bytes = 0
    for fname in sorted(weight_files):
        fpath = os.path.join(path, fname)
        real  = os.path.realpath(fpath)
        if not os.path.exists(real):
            bad.append(f"  {fname} → broken symlink ({real})")
        else:
            size = os.path.getsize(real)
            if size == 0:
                bad.append(f"  {fname} → 0 bytes")
            else:
                total_bytes += size

    if bad:
        log.error(
            f"Corrupt/empty weight files in {path!r}:\n" + "\n".join(bad) + "\n"
            f"Delete the snapshot and re-download."
        )
        sys.exit(1)

    total_gb = total_bytes / (1024 ** 3)

    # Warn if we have no index file but the total size looks too small
    if not expected_shards and total_gb < 10.0:
        log.warn(
            f"Only {total_gb:.1f} GB found across {len(weight_files)} shard(s). "
            f"This seems too small — the download may be incomplete."
        )

    log.info(
        f"Model validation OK — {len(weight_files)} shards, "
        f"{total_gb:.1f} GB  →  {path}"
    )


def _resolve_hf_snapshot(hf_home: str, model_id: str) -> str | None:
    """
    Given an HF_HOME directory and a model ID like "Qwen/Qwen3.5-27B",
    return the exact snapshot path so vLLM can load weights directly
    without any network calls.

    HF cache layout:
      <hf_home>/hub/models--<org>--<name>/
        refs/main          ← contains the commit hash
        snapshots/<hash>/  ← actual model files
    """
    if "/" not in model_id:
        return None
    org, name = model_id.split("/", 1)
    hub_dir = os.path.join(hf_home, "hub")
    model_cache = os.path.join(hub_dir, f"models--{org}--{name}")

    if not os.path.isdir(model_cache):
        return None

    # Prefer the hash recorded in refs/main
    refs_main = os.path.join(model_cache, "refs", "main")
    if os.path.isfile(refs_main):
        try:
            commit = open(refs_main).read().strip()
            snap = os.path.join(model_cache, "snapshots", commit)
            if _has_config_json(snap):
                return snap
        except OSError:
            pass

    # Fall back: pick newest snapshot directory that has a config.json
    snaps_dir = os.path.join(model_cache, "snapshots")
    if os.path.isdir(snaps_dir):
        for snap in sorted(os.listdir(snaps_dir), reverse=True):
            snap_path = os.path.join(snaps_dir, snap)
            if _has_config_json(snap_path):
                return snap_path

    return None


def _find_model_on_volume(volume_root: str, hint: str) -> str | None:
    """
    Search common layouts under `volume_root` for a model directory.

    Search order:
      1. <volume_root>/<hint>  or  <volume_root>/<short_name>
      2. <volume_root>/models/<hint | short_name>
      3. HF snapshot via refs/main inside <volume_root>/hub/…
      4. HF snapshot via refs/main inside <volume_root>/hf-cache/hub/…
      5. Any immediate child that has a config.json
    """
    short_name = hint.split("/")[-1] if "/" in hint else hint

    # Plain directory candidates
    candidates = [
        os.path.join(volume_root, hint),
        os.path.join(volume_root, short_name),
        os.path.join(volume_root, "models", hint),
        os.path.join(volume_root, "models", short_name),
    ]
    for path in candidates:
        if _has_config_json(path):
            return path

    # HF cache layouts — check common cache directory names under the volume root
    for hf_home in [
        volume_root,
        os.path.join(volume_root, "huggingface-cache"),
        os.path.join(volume_root, "hf-cache"),
        os.path.join(volume_root, "huggingface"),
    ]:
        snap = _resolve_hf_snapshot(hf_home, hint)
        if snap:
            return snap

    # Last resort: any direct child with config.json
    if os.path.isdir(volume_root):
        for child in sorted(os.listdir(volume_root)):
            child_path = os.path.join(volume_root, child)
            if os.path.isdir(child_path) and _has_config_json(child_path):
                return child_path

    return None


def resolve_model_path() -> str | None:
    """
    Return the exact local path to pass as --model to vLLM, or None.

    Priority:
      1. MODEL_PATH env var — direct path to model dir with config.json.
      2. HF_HOME env var  — resolve snapshot from the HF cache on the volume.
      3. Auto-scan /runpod-volume (or VOLUME_PATH) for the model in config.yaml.
    """
    import yaml

    # Read model hint from env or config.yaml
    try:
        with open(CONFIG_PATH) as f:
            config = yaml.safe_load(f) or {}
        model_hint = os.environ.get("MODEL_NAME") or config.get("model", "")
    except Exception:
        model_hint = os.environ.get("MODEL_NAME", "")

    # 1. Explicit local path
    explicit = os.environ.get("MODEL_PATH", "").strip()
    if explicit:
        if _has_config_json(explicit):
            log.info(f"Model source: MODEL_PATH  →  {explicit}")
            return explicit
        log.warn(f"MODEL_PATH={explicit!r} has no config.json — will try other locations.")

    # 2. HF_HOME — resolve to the exact snapshot to avoid ANY network call
    hf_home = os.environ.get("HF_HOME", "").strip()
    if hf_home:
        snap = _resolve_hf_snapshot(hf_home, model_hint)
        if snap:
            log.info(f"Model source: HF snapshot  →  {snap}")
            return snap
        log.warn(
            f"HF_HOME={hf_home!r} is set but no snapshot found for {model_hint!r}. "
            f"Expected: {hf_home}/hub/models--<org>--<name>/snapshots/<hash>/"
        )

    # 3. Auto-scan the volume
    volume_root = os.environ.get("VOLUME_PATH", "/runpod-volume")
    if os.path.isdir(volume_root):
        log.info(f"Scanning {volume_root!r} for {model_hint!r} …")
        found = _find_model_on_volume(volume_root, model_hint)
        if found:
            log.info(f"Model source: auto-discovered  →  {found}")
            return found
        log.warn(
            f"Model not found on volume. "
            f"Set MODEL_PATH=<path> or HF_HOME=<hf-cache dir> in your RunPod env vars."
        )
    else:
        log.warn(f"Volume {volume_root!r} not mounted — model will be downloaded from HuggingFace.")

    return None


def build_vllm_env(model_path: str | None) -> dict:
    """
    Build the subprocess environment for vLLM.
    When a local model path is resolved, HF_HUB_OFFLINE=1 is set so vLLM
    never touches the network — which prevents 'Disk quota exceeded' errors
    caused by HuggingFace trying to write update metadata.
    """
    env = os.environ.copy()

    hf_home = os.environ.get("HF_HOME", "").strip()
    if hf_home:
        env["HF_HOME"] = hf_home
        env.setdefault("HF_DATASETS_CACHE", os.path.join(hf_home, "datasets"))
        log.info(f"HF_HOME  →  {hf_home}")

    if model_path:
        # Model is fully local — block all network access to HF Hub.
        # This prevents 'Disk quota exceeded' from metadata writes.
        env["HF_HUB_OFFLINE"] = "1"
        env["TRANSFORMERS_OFFLINE"] = "1"
        log.info("Offline mode: HF_HUB_OFFLINE=1, TRANSFORMERS_OFFLINE=1")
    return env


def start_vllm_server() -> subprocess.Popen:
    """
    Launch vLLM's OpenAI-compatible HTTP server as a child process.
    - tensor-parallel-size is clamped to available GPUs automatically.
    - Model path resolved to exact local snapshot; HF network disabled when local.
    - Model directory is validated before vLLM starts so broken downloads
      produce a clear error instead of a cryptic EngineCore crash.
    """
    tp         = resolve_tensor_parallel_size()
    model_path = resolve_model_path()
    env        = build_vllm_env(model_path)

    # Validate before handing off to vLLM.
    # Passes the expected model ID so a wrong-model-in-volume mismatch is
    # caught here with a clear message instead of inside the EngineCore.
    if model_path:
        import yaml
        try:
            with open(CONFIG_PATH) as f:
                _cfg = yaml.safe_load(f) or {}
            expected_id = os.environ.get("MODEL_NAME") or _cfg.get("model")
        except Exception:
            expected_id = os.environ.get("MODEL_NAME")
        validate_model_dir(model_path, expected_model_id=expected_id)

    # The human-readable model name used in API requests (the "model" field).
    # When vLLM is given a local path via --model, it registers under that path.
    # --served-model-name gives it a clean HuggingFace-style name instead so
    # client requests with e.g. "model": "Qwen/Qwen3.5-27B" work correctly.
    served_model_name = (
        os.environ.get("MODEL_NAME")
        or (expected_id if model_path else None)
        or "default"
    )

    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--config", CONFIG_PATH,
        "--tensor-parallel-size", str(tp),
        "--served-model-name", served_model_name,
    ]

    if model_path:
        cmd += ["--model", model_path]

    global _SERVED_MODEL_NAME
    _SERVED_MODEL_NAME = served_model_name

    log.info(f"Available GPUs: {available_gpus()} | tensor-parallel-size: {tp}")
    log.info(f"Served model name: {served_model_name}")
    log.info(f"vLLM command: {' '.join(cmd)}")
    return subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stderr, env=env)


async def wait_for_vllm(proc: subprocess.Popen, timeout: int = READY_TIMEOUT) -> None:
    """
    Block until vLLM is truly ready to serve inference — not just HTTP-up.

    vLLM's /health returns 200 as soon as the HTTP server starts, long before
    model weights are in VRAM.  /v1/models returns the model entry as soon as
    it is *configured*, not when weights are loaded.  The only reliable signal
    is a successful inference response.

    Sequence (all phases share the same deadline):
      1. Poll /health  — wait for the HTTP server to start.
      2. Poll /v1/models — wait for the model to be configured.
      3. Warmup inference — send max_tokens=1 and wait for a 200 response.
         This blocks until weights are in VRAM and CUDA graphs are compiled,
         guaranteeing zero load latency on the first real request.
    """
    import aiohttp

    log.info(f"Waiting for vLLM to be ready (timeout={timeout}s) …")
    loop     = asyncio.get_event_loop()
    deadline = loop.time() + timeout

    def _check_proc() -> None:
        if proc.poll() is not None:
            log.error(f"vLLM process exited early with code {proc.returncode}.")
            sys.exit(1)

    # ── Phase 1: HTTP server alive ────────────────────────────────────────────
    while loop.time() < deadline:
        _check_proc()
        try:
            async with aiohttp.ClientSession() as s:
                async with s.get(
                    f"{VLLM_URL}/health",
                    timeout=aiohttp.ClientTimeout(total=3),
                ) as r:
                    if r.status == 200:
                        log.info("vLLM HTTP server is up — waiting for model weights …")
                        break
        except Exception:
            pass
        await asyncio.sleep(2)
    else:
        log.error(f"vLLM HTTP server did not start within {timeout}s.")
        proc.kill()
        sys.exit(1)

    # ── Phase 2: Model entry registered ──────────────────────────────────────
    # (This passes quickly — it only confirms the model is configured, not loaded.)
    while loop.time() < deadline:
        _check_proc()
        try:
            async with aiohttp.ClientSession() as s:
                async with s.get(
                    f"{VLLM_URL}/v1/models",
                    timeout=aiohttp.ClientTimeout(total=5),
                ) as r:
                    if r.status == 200:
                        data = await r.json()
                        if data.get("data"):
                            ids = [m["id"] for m in data["data"]]
                            log.info(f"Model entry registered: {ids}")
                            break
        except Exception:
            pass
        await asyncio.sleep(3)
    else:
        log.error(f"vLLM model did not register within {timeout}s.")
        proc.kill()
        sys.exit(1)

    # ── Phase 3: Warmup inference ─────────────────────────────────────────────
    # This is the ONLY reliable signal that weights are in VRAM and CUDA graphs
    # are compiled.  We keep retrying until deadline (not a separate 120s cap)
    # so large models that take 200-300s to load don't time out prematurely.
    log.info("Sending warmup inference request (this blocks until model is in VRAM) …")
    warmup_body = {
        "model":      _SERVED_MODEL_NAME or "default",
        "messages":   [{"role": "user", "content": "Hi"}],
        "max_tokens": 1,
        "stream":     False,
    }
    warmup_logged = False
    while loop.time() < deadline:
        _check_proc()
        remaining = int(deadline - loop.time())
        try:
            async with aiohttp.ClientSession() as s:
                async with s.post(
                    f"{VLLM_URL}/v1/chat/completions",
                    json=warmup_body,
                    # Give each single attempt up to the full remaining time so
                    # the request isn't cancelled mid-flight while weights load.
                    timeout=aiohttp.ClientTimeout(total=max(remaining, 10)),
                ) as r:
                    if r.status == 200:
                        log.info("Warmup complete — model is in VRAM, worker is ready.")
                        return
                    text = await r.text()
                    if not warmup_logged:
                        log.info(
                            f"Warmup queued (HTTP {r.status}) — "
                            f"model still loading, {remaining}s remaining …"
                        )
                        warmup_logged = True
        except asyncio.TimeoutError:
            log.info(f"Warmup still waiting for model ({remaining}s remaining) …")
            warmup_logged = False  # reset so next loop logs again
        except Exception as exc:
            log.warn(f"Warmup request error: {exc}")
        await asyncio.sleep(5)

    log.error(f"Model did not become ready within {timeout}s.")
    proc.kill()
    sys.exit(1)


# ---------------------------------------------------------------------------
# Request routing
# ---------------------------------------------------------------------------

def resolve_route_and_body(job_input: dict) -> tuple[str, dict]:
    """
    Accepts two input shapes:

    1. Explicit routing (wraps any vLLM endpoint):
         { "openai_route": "/v1/chat/completions", "openai_input": { … } }

    2. Bare OpenAI body (auto-detected):
         { "messages": […], "model": "…", … }   → /v1/chat/completions
         { "prompt": "…",   "model": "…", … }   → /v1/completions
    """
    if "openai_route" in job_input:
        route = job_input["openai_route"]
        body  = job_input.get("openai_input", {})
        return route, body

    if "messages" in job_input:
        return "/v1/chat/completions", job_input

    if "prompt" in job_input:
        return "/v1/completions", job_input

    # Default: treat the whole payload as a chat completions body
    return "/v1/chat/completions", job_input


# ---------------------------------------------------------------------------
# RunPod handler
# ---------------------------------------------------------------------------

async def handler(job):
    import aiohttp

    job_input = job.get("input", {})
    route, body = resolve_route_and_body(job_input)

    # Always overwrite "model" with the name vLLM is actually serving.
    # Without this, requests that omit "model" or use the HuggingFace ID get a
    # 404 when vLLM was loaded from a local path and registered under that path.
    if _SERVED_MODEL_NAME:
        body = {**body, "model": _SERVED_MODEL_NAME}

    url = f"{VLLM_URL}{route}"
    is_stream = body.get("stream", False)

    try:
        timeout = aiohttp.ClientTimeout(total=None)  # let vLLM control timing
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(url, json=body) as resp:

                if resp.status != 200:
                    error_text = await resp.text()
                    yield {"error": f"vLLM HTTP {resp.status}: {error_text}"}
                    return

                if is_stream:
                    async for raw in resp.content:
                        line = raw.decode("utf-8").strip()
                        if not line.startswith("data: "):
                            continue
                        data = line[len("data: "):]
                        if data == "[DONE]":
                            break
                        try:
                            yield json.loads(data)
                        except json.JSONDecodeError:
                            pass
                else:
                    yield await resp.json()

    except Exception as exc:
        err = str(exc)
        log.error(f"Handler error: {err}\n{traceback.format_exc()}")
        if "CUDA" in err or "cuda" in err:
            log.error("CUDA error detected — restarting worker.")
            sys.exit(1)
        yield {"error": err}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # ---------------------------------------------------------------------------
    # Pre-flight: ensure the model is fully present on the volume.
    # ---------------------------------------------------------------------------
    _hf_home    = os.environ.get("HF_HOME", "/runpod-volume/huggingface-cache")
    _vol_root   = os.environ.get("VOLUME_PATH", "/runpod-volume")

    # Show what's on the volume so quota surprises are immediately visible in logs
    _report_volume_usage(_vol_root)

    # Remove stale wrong-path cache dirs left by old buggy code
    # (cache_dir=HF_HOME instead of HF_HOME/hub → models--* accumulate directly
    #  under HF_HOME and silently consume quota across retries)
    _purge_wrong_path_cache(_hf_home)

    _model_path = resolve_model_path()

    # Read model ID: env var wins, then fall back to config.yaml
    _model_id = os.environ.get("MODEL_NAME", "")
    if not _model_id:
        try:
            import yaml as _yaml
            with open(CONFIG_PATH) as _f:
                _model_id = (_yaml.safe_load(_f) or {}).get("model", "")
        except Exception:
            _model_id = ""

    if _model_path:
        try:
            validate_model_dir(_model_path, _model_id)
        except _IncompleteDownloadError as _exc:
            log.info(
                "Incomplete model detected — purging broken snapshot to free space, "
                "then re-downloading missing shards …"
            )
            _purge_incomplete_snapshot(_exc.path)
            download_model_to_volume(_model_id or "unknown/model")
            _model_path = resolve_model_path()
    else:
        # Model not found at all on the volume — download it from scratch.
        if _model_id:
            log.info(f"Model {_model_id!r} not found on volume — downloading …")
            download_model_to_volume(_model_id)
            _model_path = resolve_model_path()
        else:
            log.warn(
                "No model path resolved and MODEL_NAME is not set. "
                "vLLM will use the model specified in config.yaml directly."
            )

    # 1. Start vLLM (model loads here — ONCE for the entire worker lifetime)
    vllm_proc = start_vllm_server()

    # 2. Block until the model is loaded and vLLM is accepting requests.
    #    RunPod will NOT receive any jobs until this returns.
    asyncio.run(wait_for_vllm(vllm_proc))

    # 3. Worker is hot — hand control to RunPod's job loop.
    runpod.serverless.start(
        {
            "handler": handler,
            "return_aggregate_stream": True,
        }
    )
