import logging

import modal


logger = logging.getLogger(__name__)

app = modal.App("video-clip-search-servers")

MODEL_NAME = "Qwen/Qwen3-VL-Embedding-8B"
MINUTES = 60
EMBEDDING_STORE_DIR = "/root/embeddings"

EMBED_URL = (
    "https://modal-labs-adamkelch-dev--video-clip-search-servers-embe-134493.modal.run"
)


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
        "fastapi",
    )
    .env({"HF_XET_HIGH_PERFORMANCE": "1", "FORCE_QWENVL_VIDEO_READER": "torchcodec"})
)

query_router_image = modal.Image.debian_slim().uv_pip_install(
    "fastapi", "requests", "numpy", "pandas", "pyarrow"
)

# --- Volumes ---

hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)
embedding_store_vol = modal.Volume.from_name(
    "video-clip-embeddings", create_if_missing=True
)

# ---------------------------------------------------------------------------
# GPU Embedding Server (vLLM serve)
# ---------------------------------------------------------------------------

VLLM_PORT = 8000
VLLM_MAX_MODEL_LEN = 4096 * 2


@app.function(
    gpu=["A100", "A100-80GB", "H100"],
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
@modal.web_server(port=VLLM_PORT, startup_timeout=300)
def embedding_server():
    """Run vLLM OpenAI-compatible embedding server. Clients use /v1/embeddings with
    body: {"model": MODEL_NAME, "input": [{"type": "text", "text": "..."}]}
    or {"type": "video_url", "video_url": {"url": "https://..."}}
    """
    import subprocess

    subprocess.Popen(
        [
            "vllm",
            "serve",
            MODEL_NAME,
            "--task",
            "embed",
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
        check=True,
    )


# ---------------------------------------------------------------------------
# Query Router
# ---------------------------------------------------------------------------


@app.function(
    image=query_router_image,
    volumes={EMBEDDING_STORE_DIR: embedding_store_vol},
    scaledown_window=30 * MINUTES,
    min_containers=1,
)
@modal.asgi_app()
def query_router():
    import glob
    import os
    import re

    import numpy as np
    import pandas as pd
    import requests
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import JSONResponse

    # Sort parquet files by batch index
    parquet_files = glob.glob(os.path.join(EMBEDDING_STORE_DIR, "embeddings_*.parquet"))
    parquet_files.sort(
        key=lambda p: int(re.search(r"embeddings_(\d+)\.parquet", p).group(1))
    )

    print(f"Found {len(parquet_files)} parquet files")

    if not parquet_files:
        raise ValueError("Embeddings store is empty")


    df = pd.concat([pd.read_parquet(f) for f in parquet_files], ignore_index=True)

    logger.info(f"Loaded {len(df)} embeddings from store")
    embedding_matrix = np.array(df["embedding"].tolist())
    embedding_urls = df["url"].tolist()
    # vectors are already normalized — dot product == cosine similarity

    def get_text_embedding(text: str) -> list[float]:
        response = requests.post(
            f"{EMBED_URL}/v1/embeddings",
            json={
                "model": MODEL_NAME,
                "input": [{"type": "text", "text": text}],
            },
            timeout=60,
        )
        response.raise_for_status()
        return response.json()["data"][0]["embedding"]

    web_app = FastAPI()

    @web_app.post("/search")
    async def search(request: Request):
        query = await request.json()
        query_type = query.get("type")
        logger.info(f"Search query: type={query_type!r} text={query.get('text', '')!r}")

        if query_type == "video":
            raise HTTPException(
                status_code=400, detail="Video queries are not yet supported"
            )
        elif query_type != "text":
            raise HTTPException(status_code=400, detail="Invalid query type")

        query_embedding = get_text_embedding(query.get("text", ""))

        query_vec = np.array(query_embedding)
        query_vec = query_vec / np.linalg.norm(query_vec)
        similarities = embedding_matrix @ query_vec

        best_idx = int(np.argmax(similarities))

        return JSONResponse(
            content={
                "url": embedding_urls[best_idx],
                "score": float(similarities[best_idx]),
            }
        )

    return web_app
