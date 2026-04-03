import logging
import subprocess
import time
import requests

import modal


app = modal.App("video-clip-search-query-inference")
logger = logging.getLogger(__name__)

MODEL_NAME = "TomoroAI/tomoro-colqwen3-embed-4b"
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
        "fastapi==0.135.1",
        "uvicorn==0.32.1",
        "requests==2.32.3",
    )
    .run_commands(
        "pip install torchcodec==0.10.0 --index-url=https://download.pytorch.org/whl/cu129"
    )
    .env({"HF_XET_HIGH_PERFORMANCE": "1", "FORCE_QWENVL_VIDEO_READER": "torchcodec"})
)

hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)


with vllm_image.imports():
    from fastapi import FastAPI, HTTPException, Request


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


def create_inference_payload(req: dict) -> dict:
    query_type = str(req.get("type", "")).lower()
    if query_type == "text":
        text = req.get("text", "")
        if not text:
            raise HTTPException(
                status_code=400, detail="text is required for type='text'"
            )
        return {"model": MODEL_NAME, "input": [text]}

    if query_type == "image":
        image_url = req.get("image_url", "")
        if not image_url:
            raise HTTPException(
                status_code=400, detail="image_url is required for type='image'"
            )
        return {
            "model": MODEL_NAME,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image_url}},
                        {"type": "text", "text": "Represent this image."},
                    ],
                }
            ],
        }

    if query_type == "video":
        video_url = req.get("video_url", "")
        if not video_url:
            raise HTTPException(
                status_code=400, detail="video_url is required for type='video'"
            )
        return {
            "model": MODEL_NAME,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "video_url", "video_url": {"url": video_url}},
                        {"type": "text", "text": "Represent this video."},
                    ],
                }
            ],
        }

    raise HTTPException(
        status_code=400, detail="type must be one of: text, image, video"
    )


@app.function(
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
@modal.concurrent(max_inputs=10)
@modal.asgi_app()
def query_inference_server():
    logger.info("Starting vLLM server")
    subprocess.Popen(
        [
            "vllm",
            "serve",
            MODEL_NAME,
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

    # Warmup keeps first user request fast and validates request/response shape.
    warmup_response = requests.post(
        f"http://localhost:{VLLM_PORT}/pooling",
        json={"model": MODEL_NAME, "input": ["warm up"]},
        timeout=60,
    )
    warmup_response.raise_for_status()

    api_server = FastAPI()

    @api_server.post("/embed")
    async def embed(request: Request):
        req = await request.json()
        payload = create_inference_payload(req)
        response = requests.post(
            f"http://localhost:{VLLM_PORT}/pooling",
            json=payload,
            timeout=120,
        )
        if not response.ok:
            raise HTTPException(status_code=response.status_code, detail=response.text)
        data = response.json()
        embedding = data["data"][0]["data"]
        return {
            "embedding": embedding,
            "total_token_embeddings": len(embedding),
        }

    logger.info("Server startup completed")
    return api_server
