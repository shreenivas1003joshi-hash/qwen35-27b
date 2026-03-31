import sys
import multiprocessing
import traceback
import runpod
from runpod import RunPodLogger

log = RunPodLogger()

vllm_engine = None
openai_engine = None


async def handler(job):
    try:
        from utils import JobInput
        job_input = JobInput(job["input"])
        engine = openai_engine if job_input.openai_route else vllm_engine
        results_generator = engine.generate(job_input)
        async for batch in results_generator:
            yield batch
    except Exception as e:
        error_str = str(e)
        full_traceback = traceback.format_exc()

        log.error(f"Error during inference: {error_str}")
        log.error(f"Full traceback:\n{full_traceback}")

        # CUDA errors = worker is broken, exit to let RunPod spin up a healthy one
        if "CUDA" in error_str or "cuda" in error_str:
            log.error("Terminating worker due to CUDA/GPU error")
            sys.exit(1)

        yield {"error": error_str}


# Only run in main process to prevent re-initialization when vLLM spawns worker subprocesses
if __name__ == "__main__" or multiprocessing.current_process().name == "MainProcess":

    runpod.serverless.start(
        {
            "handler": handler,
            "concurrency_modifier": 1,
            "return_aggregate_stream": True,
        }
    )
