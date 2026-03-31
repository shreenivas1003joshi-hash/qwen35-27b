"""
Optional model pre-downloader.

Run this as a build step (RUN python /app/src/download_model.py) to bake the
model weights into the Docker image, or execute it at container start before
the handler so that the first RunPod job doesn't have to wait for a download.

Required environment variables
--------------------------------
MODEL_NAME            – HuggingFace model ID, e.g. "Qwen/Qwen3.5-27B"
HF_TOKEN              – HuggingFace access token (needed for gated models)

Optional environment variables
--------------------------------
HF_HOME               – cache root (default: /root/.cache/huggingface)
CONFIG_PATH           – path to config.yaml (default: /app/src/config.yaml)
"""
import os
import sys
import yaml

from huggingface_hub import snapshot_download


def main() -> None:
    config_path = os.environ.get("CONFIG_PATH", "/app/src/config.yaml")
    config: dict = {}
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = yaml.safe_load(f) or {}

    model_name: str = os.environ.get("MODEL_NAME") or config.get("model", "")
    if not model_name:
        print(
            "ERROR: MODEL_NAME env var is not set and 'model' is missing from config.yaml.",
            file=sys.stderr,
        )
        sys.exit(1)

    token: str | None = os.environ.get("HF_TOKEN") or os.environ.get(
        "HUGGING_FACE_HUB_TOKEN"
    )

    print(f"Downloading model: {model_name}", flush=True)
    snapshot_download(
        repo_id=model_name,
        token=token,
        ignore_patterns=["*.pt", "*.bin", "original/*"],  # prefer safetensors
    )
    print(f"Model '{model_name}' downloaded successfully.", flush=True)


if __name__ == "__main__":
    main()
