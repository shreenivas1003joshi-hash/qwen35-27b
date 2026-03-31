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

def start_vllm_server() -> subprocess.Popen:
    """
    Launch vLLM's OpenAI-compatible HTTP server as a child process.
    All server stdout/stderr is forwarded to this process's streams so RunPod
    captures the logs.
    """
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--config", CONFIG_PATH,
    ]
    log.info(f"Starting vLLM server: {' '.join(cmd)}")
    return subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stderr)


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
