import glob
import logging
import os
import re
import subprocess
import time

import modal
import requests


app = modal.App("video-clip-search-servers")
logger = logging.getLogger(__name__)

MODEL_NAME = "Qwen/Qwen3-VL-Embedding-8B"
MINUTES = 60
EMBEDDING_STORE_DIR = "/root/embeddings"


# --- Images ---

vllm_image = (
    modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.13")
    .entrypoint([])
    .apt_install("ffmpeg")
    .uv_pip_install(
        "vllm==0.16.0",
        "huggingface-hub==0.36.0",
        "qwen-vl-utils==0.0.14",
        "torchcodec==0.9.0",
        "fastapi==0.135.1",
        "pandas==3.0.1",
        "requests==2.32.3",
        "numpy==2.0.0",
        "pyarrow==23.0.1",
        "cupy-cuda12x==14.0.0",
    )
    .env({"HF_XET_HIGH_PERFORMANCE": "1", "FORCE_QWENVL_VIDEO_READER": "torchcodec"})
)

# --- Volumes ---

hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)
embedding_store_vol = modal.Volume.from_name("dance-video-embeddings")

with vllm_image.imports():
    import pandas as pd
    import cupy as cp
    from fastapi import FastAPI, Request, HTTPException
    from fastapi.responses import JSONResponse

# ---------------------------------------------------------------------------
# GPU Embedding Server (vLLM serve)
# ---------------------------------------------------------------------------

VLLM_PORT = 8000


@app.function(
    gpu="L40S",
    image=vllm_image,
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
        EMBEDDING_STORE_DIR: embedding_store_vol,
    },
    timeout=10 * MINUTES,
    scaledown_window=15 * MINUTES,
    min_containers=1,
    max_containers=5,
)
@modal.concurrent(max_inputs=10)
@modal.asgi_app()
def video_search_server():
    """Serves a /search endpoint backed by vLLM embeddings.

    Clients POST {"text": "..."} and receive {"url": "...", "score": float}.
    Embeddings are loaded from parquet files in EMBEDDING_STORE_DIR on startup.
    """
    VLLM_MAX_MODEL_LEN = 4096 * 10

    logger.info("Loading embeddings")
    parquet_files = glob.glob(os.path.join(EMBEDDING_STORE_DIR, "embeddings_*.parquet"))

    if len(parquet_files) == 0:
        raise ValueError("No embeddings found in store")

    # sort parquet files by batch index
    parquet_files.sort(
        key=lambda p: int(re.search(r"embeddings_(\d+)\.parquet", p).group(1))
    )
    df = pd.concat([pd.read_parquet(f) for f in parquet_files], ignore_index=True)

    embedding_matrix = cp.array(df["embedding"].tolist())
    embedding_urls = df["url"].tolist()

    logger.info("Starting vLLM server")
    subprocess.Popen(
        [
            "vllm",
            "serve",
            "Qwen/Qwen3-VL-Embedding-8B",
            "--runner",
            "pooling",
            "--trust-remote-code",
            "--max-model-len",
            str(VLLM_MAX_MODEL_LEN),
            "--dtype",
            "bfloat16",
            "--attention-backend",
            "flashinfer",
            "--host",
            "0.0.0.0",
            "--port",
            str(VLLM_PORT),
        ],
    )

    wait_for_vllm_server()

    def get_text_embedding(text: str) -> list[float]:
        response = requests.post(
            f"http://localhost:{VLLM_PORT}/v1/embeddings",
            json={
                "model": MODEL_NAME,
                "messages": [
                    {
                        "role": "system",
                        "content": [
                            {"type": "text", "text": "Represent the user's input."}
                        ],
                    },
                    {"role": "user", "content": [{"type": "text", "text": text}]},
                ],
                "encoding_format": "float",
            },
        )
        response.raise_for_status()
        return response.json()["data"][0]["embedding"]

    get_text_embedding("warm up")

    api_server = FastAPI()

    @api_server.post("/search")
    async def search(request: Request):
        query = await request.json()
        query_text = query.get("text", "")
        if not query_text:
            raise HTTPException(status_code=400, detail="Query text is required")

        query_embedding = get_text_embedding(query_text)
        query_vec = cp.array(query_embedding)

        # At scale, replace with a database or ANN search
        similarities = embedding_matrix @ query_vec
        best_idx = int(cp.argmax(similarities))

        return JSONResponse(
            content={
                "url": embedding_urls[best_idx],
                "score": float(similarities[best_idx]),
            }
        )

    logger.info("Server startup completed")
    return api_server


def wait_for_vllm_server():
    for _ in range(300):  # up to ~5 minutes
        try:
            r = requests.get(f"http://localhost:{VLLM_PORT}/health")
            if r.status_code == 200:
                logger.info("vLLM server is ready")
                return
            logger.info(f"vLLM server returned {r.status_code}, retrying...")
        except requests.ConnectionError:
            pass
        time.sleep(1)
    raise RuntimeError("vLLM server failed to start")
