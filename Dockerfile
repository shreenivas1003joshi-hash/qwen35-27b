FROM vllm/vllm-openai:nightly

WORKDIR /app

# Install RunPod SDK + aiohttp (for the HTTP proxy) on top of the vLLM base image
COPY builder/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Copy handler and config
COPY src/handler.py      /app/src/handler.py
COPY src/config.yaml     /app/src/config.yaml
COPY src/download_model.py /app/src/download_model.py

ENV PYTHONUNBUFFERED=1
ENV VLLM_LOGGING_LEVEL=INFO

# Override the vLLM image's default entrypoint.
# handler.py starts the vLLM server internally and waits for it to be
# fully loaded before RunPod accepts any jobs.
# Use shell form so the shell resolves python3 from PATH correctly,
# the same way the vLLM base image does internally.
ENTRYPOINT ["/bin/bash", "-c"]
CMD ["python3 -u /app/src/handler.py"]
