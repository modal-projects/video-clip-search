import logging
import os
import subprocess
import time
import requests

import modal
import modal.experimental

"""
Modal inference server template.

This app exposes vLLM's native HTTP routes directly via
modal.experimental.http_server (e.g. /pooling, /health).
"""

app = modal.App("video-clip-search-query-inference")
logger = logging.getLogger(__name__)

MINUTES = 60
VLLM_PORT = 8000
VLLM_MAX_MODEL_LEN = 4096 * 10


vllm_image = (
    modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.13")
    .entrypoint([])
    .apt_install("ffmpeg")
    .uv_pip_install(
        "vllm==0.18.0",
        "huggingface-hub==0.36.0",
        "qwen-vl-utils==0.0.14",
        "requests==2.32.3",
    )
    .run_commands(
        "pip install torchcodec==0.10.0 --index-url=https://download.pytorch.org/whl/cu129"
    )
    .env({"HF_XET_HIGH_PERFORMANCE": "1", "FORCE_QWENVL_VIDEO_READER": "torchcodec"})
)

hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)


def wait_for_vllm_server():
    for _ in range(300):  # up to ~5 minutes
        try:
            r = requests.get(f"http://localhost:{VLLM_PORT}/health", timeout=5)
            if r.status_code == 200:
                logger.info("vLLM server is ready")
                return
            logger.info("vLLM server returned %s, retrying...", r.status_code)
        except requests.ConnectionError:
            pass
        time.sleep(1)
    raise RuntimeError("vLLM server failed to start")


@app.cls(
    gpu="L40S",
    image=vllm_image,
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
    },
    timeout=10 * MINUTES,
    scaledown_window=15 * MINUTES,
    min_containers=1,
    max_containers=5,
)
@modal.concurrent(target_inputs=8)
@modal.experimental.http_server(port=VLLM_PORT, proxy_regions=["eu-west"])
class QueryInferenceServer:
    @modal.enter()
    def startup(self):
        model_name = "TomoroAI/tomoro-colqwen3-embed-4b"
        logger.info("Starting vLLM server")
        self.process = subprocess.Popen(
            [
                "vllm",
                "serve",
                model_name,
                "--runner",
                "pooling",
                "--max-model-len",
                str(VLLM_MAX_MODEL_LEN),
                "--dtype",
                "bfloat16",
                "--gpu-memory-utilization",
                "0.9",
                "--attention-backend",
                "flashinfer",
                "--host",
                "0.0.0.0",
                "--port",
                str(VLLM_PORT),
            ],
        )
        wait_for_vllm_server()

        # Warmup server for first request
        warmup_response = requests.post(
            f"http://localhost:{VLLM_PORT}/pooling",
            json={"model": model_name, "input": ["warm up"]},
            timeout=60,
        )
        warmup_response.raise_for_status()
        logger.info("Server startup completed")

    @modal.exit()
    def stop(self):
        if getattr(self, "process", None) is not None:
            self.process.terminate()
