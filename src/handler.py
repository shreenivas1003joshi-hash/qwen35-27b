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


def _find_model_on_volume(volume_root: str, hint: str) -> str | None:
    """
    Search `volume_root` for a directory that contains config.json.

    Search order:
      1. Direct match:           <volume_root>/<hint>
      2. Nested models folder:   <volume_root>/models/<hint>
      3. HF snapshot cache:      <volume_root>/hub/models--<org>--<name>/snapshots/<hash>/
      4. Any immediate child of  <volume_root>  that has a config.json
    """
    # Normalise hint  "Qwen/Qwen3.5-27B"  →  "Qwen3.5-27B"
    short_name = hint.split("/")[-1] if "/" in hint else hint

    candidates = [
        os.path.join(volume_root, hint),
        os.path.join(volume_root, short_name),
        os.path.join(volume_root, "models", hint),
        os.path.join(volume_root, "models", short_name),
    ]

    # HuggingFace snapshot cache layout:
    #   <volume_root>/hub/models--Qwen--Qwen3.5-27B/snapshots/<hash>/
    if "/" in hint:
        org, name = hint.split("/", 1)
        hf_model_dir = os.path.join(volume_root, "hub", f"models--{org}--{name}", "snapshots")
        if os.path.isdir(hf_model_dir):
            for snap in sorted(os.listdir(hf_model_dir)):
                candidates.append(os.path.join(hf_model_dir, snap))

    for path in candidates:
        if _has_config_json(path):
            return path

    # Last resort: any immediate child with config.json
    if os.path.isdir(volume_root):
        for child in sorted(os.listdir(volume_root)):
            child_path = os.path.join(volume_root, child)
            if os.path.isdir(child_path) and _has_config_json(child_path):
                return child_path

    return None


def _log_volume_tree(volume_root: str, depth: int = 2) -> None:
    """Log the directory tree of the volume so the correct path is visible in logs."""
    if not os.path.isdir(volume_root):
        log.info(f"  (volume root {volume_root!r} does not exist)")
        return
    for root, dirs, files in os.walk(volume_root):
        level = root.replace(volume_root, "").count(os.sep)
        if level >= depth:
            dirs.clear()
            continue
        indent = "  " * level
        log.info(f"  {indent}{os.path.basename(root)}/")
        if level == depth - 1:
            for f in files[:6]:
                log.info(f"  {indent}  {f}")
            if len(files) > 6:
                log.info(f"  {indent}  … ({len(files) - 6} more files)")


def resolve_model_path() -> str | None:
    """
    Return the final model path to pass to vLLM, or None to let config.yaml decide.

    Priority:
      1. MODEL_PATH env var — must point directly at a dir with config.json.
      2. Auto-discover on VOLUME_PATH (default /runpod-volume) using the model
         name from MODEL_NAME env var or config.yaml.
      3. HF_HOME env var — handled separately via env; no path override needed.
    """
    import yaml

    volume_root = os.environ.get("VOLUME_PATH", "/runpod-volume")

    # 1. Explicit path
    explicit = os.environ.get("MODEL_PATH", "").strip()
    if explicit:
        if _has_config_json(explicit):
            log.info(f"Model source: MODEL_PATH  →  {explicit}")
            return explicit
        # Path given but wrong — search inside it as a volume root too
        log.warn(
            f"MODEL_PATH={explicit!r} has no config.json. "
            f"Scanning it and {volume_root!r} for the model …"
        )
        log.info(f"Contents of {explicit}:")
        _log_volume_tree(explicit, depth=3)

    # Read model name from env or config.yaml
    try:
        with open(CONFIG_PATH) as f:
            config = yaml.safe_load(f) or {}
        model_hint = os.environ.get("MODEL_NAME") or config.get("model", "")
    except Exception:
        model_hint = os.environ.get("MODEL_NAME", "")

    # 2. Auto-discover on the volume
    if os.path.isdir(volume_root):
        log.info(f"Scanning volume {volume_root!r} for model {model_hint!r} …")
        _log_volume_tree(volume_root, depth=2)

        found = _find_model_on_volume(volume_root, model_hint)
        if found:
            log.info(f"Model source: auto-discovered on volume  →  {found}")
            return found

        log.warn(
            f"Could not find a directory with config.json on {volume_root!r}. "
            f"Set MODEL_PATH=<exact path> or HF_HOME=<hf-cache path> in your RunPod env vars."
        )
    else:
        log.warn(f"Volume root {volume_root!r} is not mounted. Falling back to HuggingFace download.")

    return None  # fall back to config.yaml model (HF download)


def build_vllm_env() -> dict:
    """Forward HF_HOME to the vLLM subprocess if set."""
    env = os.environ.copy()
    hf_home = os.environ.get("HF_HOME", "").strip()
    if hf_home:
        env["HF_HOME"] = hf_home
        env.setdefault("HF_DATASETS_CACHE", os.path.join(hf_home, "datasets"))
        log.info(f"HF_HOME  →  {hf_home}")
    return env


def start_vllm_server() -> subprocess.Popen:
    """
    Launch vLLM's OpenAI-compatible HTTP server as a child process.
    - tensor-parallel-size is clamped to available GPUs automatically.
    - Model path is resolved from volume; falls back to HF download.
    """
    tp         = resolve_tensor_parallel_size()
    env        = build_vllm_env()
    model_path = resolve_model_path()

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
