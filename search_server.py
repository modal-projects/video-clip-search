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

MODEL_NAME = "TomoroAI/tomoro-colqwen3-embed-4b"
MINUTES = 60
EMBEDDING_STORE_DIR = "/root/embeddings"


# --- Images ---

vllm_image = (
    modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.13")
    .entrypoint([])
    .apt_install("ffmpeg")
    .uv_pip_install(
        "vllm==0.18.0",
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
embedding_store_vol = modal.Volume.from_name("colqwen3-video-embeddings2")

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
    gpu="A100-80GB",
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
    VLLM_MAX_MODEL_LEN = 4096 * 2

    logger.info("Loading embeddings")
    parquet_files = glob.glob(os.path.join(EMBEDDING_STORE_DIR, "embeddings_*.parquet"))

    if len(parquet_files) == 0:
        raise ValueError("No embeddings found in store")

    # sort parquet files by batch index
    parquet_files.sort(
        key=lambda p: int(re.search(r"embeddings_(\d+)\.parquet", p).group(1))
    )
    df = pd.concat([pd.read_parquet(f) for f in parquet_files], ignore_index=True)
    df = df.sort_values(["url", "token_index"]).reset_index(drop=True)

    # Multi-vector: one row per token per video
    all_embeddings = cp.array(
        df["embedding"].tolist(), dtype=cp.float32
    )  # (total_tokens, 320)

    # Build per-document offset table for MaxSim scoring
    doc_urls = []
    doc_offsets = []  # (start, end) indices into all_embeddings
    cursor = 0
    for url, group in df.groupby("url", sort=False):
        n = len(group)
        doc_urls.append(url)
        doc_offsets.append((cursor, cursor + n))
        cursor += n

    logger.info(
        f"Loaded {len(doc_urls)} documents, {all_embeddings.shape[0]} total token embeddings"
    )

    logger.info("Starting vLLM server")
    subprocess.Popen(
        [
            "vllm",
            "serve",
            "TomoroAI/tomoro-colqwen3-embed-4b",
            "--runner",
            "pooling",
            "--trust-remote-code",
            "--max-model-len",
            str(VLLM_MAX_MODEL_LEN),
            "--dtype",
            "bfloat16",
            "--gpu-memory-utilization",
            "0.75",
            "--attention-backend",
            "flashinfer",
            "--host",
            "0.0.0.0",
            "--port",
            str(VLLM_PORT),
        ],
    )

    wait_for_vllm_server()

    def get_query_embedding(query: dict) -> list[list[float]]:
        """Returns multi-vector embedding: list of per-token vectors (num_tokens x 320)."""
        query_type = str(query.get("type", "text")).lower()
        if query_type == "text":
            text = query.get("text", "")
            if not text:
                raise HTTPException(status_code=400, detail="text is required for type='text'")
            payload = {"model": MODEL_NAME, "input": [text]}
        elif query_type == "image":
            image_url = query.get("image_url", "")
            if not image_url:
                raise HTTPException(
                    status_code=400, detail="image_url is required for type='image'"
                )
            payload = {
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
        elif query_type == "video":
            video_url = query.get("video_url", "")
            if not video_url:
                raise HTTPException(
                    status_code=400, detail="video_url is required for type='video'"
                )
            payload = {
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
        else:
            raise HTTPException(
                status_code=400,
                detail="type must be one of: text, image, video",
            )

        response = requests.post(
            f"http://localhost:{VLLM_PORT}/pooling",
            json=payload,
        )
        response.raise_for_status()
        return response.json()["data"][0]["data"]

    get_query_embedding({"type": "text", "text": "warm up"})

    api_server = FastAPI()

    @api_server.post("/search")
    async def search(request: Request):
        query = await request.json()
        query_embedding = get_query_embedding(query)
        query_vecs = cp.array(query_embedding, dtype=cp.float32)  # (M, 320)

        # MaxSim scoring: for each query token, find max similarity with any doc token, then sum
        # Single large GEMM for all pairwise token similarities
        sim_matrix = query_vecs @ all_embeddings.T  # (M, total_tokens)

        # Score each document by slicing its token columns
        scores = cp.empty(len(doc_offsets), dtype=cp.float32)
        for i, (start, end) in enumerate(doc_offsets):
            scores[i] = sim_matrix[:, start:end].max(axis=1).sum()

        best_idx = int(cp.argmax(scores))

        return JSONResponse(
            content={
                "url": doc_urls[best_idx],
                "score": float(scores[best_idx]),
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
