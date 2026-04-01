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


def validate_model_dir(path: str) -> None:
    """
    Verify the model directory is complete enough for vLLM to load.
    Raises SystemExit with a human-readable message on failure so the
    worker exits cleanly instead of crashing deep inside the EngineCore.

    Checks:
      1. config.json is present.
      2. At least one weight shard (.safetensors / .bin) is present.
      3. No weight file resolves to a zero-byte or missing blob
         (catches partial downloads caused by disk-quota errors).
    """
    WEIGHT_EXTS = (".safetensors", ".bin", ".pt")

    if not _has_config_json(path):
        log.error(f"Model directory {path!r} has no config.json. The path is wrong.")
        sys.exit(1)

    entries = os.listdir(path)
    weight_files = [f for f in entries if any(f.endswith(e) for e in WEIGHT_EXTS)]

    if not weight_files:
        log.error(
            f"No weight files (.safetensors/.bin) found in {path!r}. "
            f"The model may not have been downloaded yet."
        )
        sys.exit(1)

    bad = []
    for fname in weight_files:
        fpath = os.path.join(path, fname)
        # HF cache uses symlinks; resolve to the actual blob
        real = os.path.realpath(fpath)
        if not os.path.exists(real):
            bad.append(f"{fname} → broken symlink ({real})")
        elif os.path.getsize(real) == 0:
            bad.append(f"{fname} → 0 bytes")

    if bad:
        log.error(
            f"Incomplete model download detected in {path!r}. "
            f"The previous download was interrupted (disk quota exceeded?). "
            f"Delete the partial files and re-download the model.\n"
            f"Affected files:\n  " + "\n  ".join(bad)
        )
        sys.exit(1)

    total_gb = sum(
        os.path.getsize(os.path.realpath(os.path.join(path, f)))
        for f in weight_files
        if os.path.exists(os.path.realpath(os.path.join(path, f)))
    ) / (1024 ** 3)
    log.info(
        f"Model validation OK: {len(weight_files)} weight files, "
        f"{total_gb:.1f} GB total  →  {path}"
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

    # HF cache layouts — check both <volume_root> and <volume_root>/hf-cache as hf_home
    for hf_home in [volume_root, os.path.join(volume_root, "hf-cache")]:
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

    # Validate before handing off to vLLM — catches incomplete downloads early
    if model_path:
        validate_model_dir(model_path)

    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--config", CONFIG_PATH,
        "--tensor-parallel-size", str(tp),
    ]

    if model_path:
        cmd += ["--model", model_path]

    log.info(f"Available GPUs: {available_gpus()} | tensor-parallel-size: {tp}")
    log.info(f"vLLM command: {' '.join(cmd)}")
    return subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stderr, env=env)


async def wait_for_vllm(proc: subprocess.Popen, timeout: int = READY_TIMEOUT) -> None:
    """
    Poll GET /health until vLLM responds 200 (model fully loaded + warmed up).
    Exits the process if vLLM crashes before becoming ready.
    """
    import aiohttp

    log.info(f"Waiting for vLLM to be ready (timeout={timeout}s) …")
    deadline = asyncio.get_event_loop().time() + timeout

    while asyncio.get_event_loop().time() < deadline:
        # Check if vLLM process died unexpectedly
        if proc.poll() is not None:
            log.error(f"vLLM process exited early with code {proc.returncode}.")
            sys.exit(1)

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{VLLM_URL}/health",
                    timeout=aiohttp.ClientTimeout(total=3),
                ) as resp:
                    if resp.status == 200:
                        log.info("vLLM is ready — worker accepting jobs.")
                        return
        except Exception:
            pass  # Server not up yet, keep polling

        await asyncio.sleep(2)

    log.error(f"vLLM did not become ready within {timeout}s. Exiting.")
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
