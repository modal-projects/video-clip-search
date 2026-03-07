from io import StringIO
import logging
import os
import subprocess
import tempfile
import time
import requests
import modal

logger = logging.getLogger(__name__)

from typing import Any, Dict, List, Optional

app = modal.App("video-clip-search")

MODEL_NAME = "Qwen/Qwen3-VL-Embedding-8B"
MINUTES = 60
INSTRUCTION = "Represent the user's input."

AIST_DANCE_BASIC_2MBPS_LINK_LIST_FILE_PATH = (
    "https://aistdancedb.ongaaccel.jp/data/video_refined/2M/refined_2M_sBM_url.csv"
)
AIST_DANCE_ADVANCED_2MBPS_LINK_LIST_FILE_PATH = (
    "https://aistdancedb.ongaaccel.jp/data/video_refined/2M/refined_2M_sFM_url.csv"
)
EMBED_BATCH_SIZE = 50
CLIPS_FILE_NAME = "dance-clips.csv"
DATASET_DIR = "/root/dance-clips"
EMBEDDING_STORE_DIR = "/root/embeddings"
EMBED_URL = (
    "https://modal-labs-adamkelch-dev--video-clip-search-embeddingser-bca6c3.modal.run"
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
    .env(
        {
            "HF_XET_HIGH_PERFORMANCE": "1",
            "FORCE_QWENVL_VIDEO_READER": "torchcodec",
        }
    )
)

orchestrator_image = modal.Image.debian_slim().uv_pip_install("pandas", "requests")

query_router_image = modal.Image.debian_slim().uv_pip_install(
    "fastapi", "requests", "numpy"
)

# --- Volumes ---

hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)
dance_clips_vol = modal.Volume.from_name("dance-clips", create_if_missing=True)
embedding_store_vol = modal.Volume.from_name(
    "dance-clip-embedding-store", create_if_missing=True
)

# ---------------------------------------------------------------------------
# GPU Embedding Server
# ---------------------------------------------------------------------------


def format_input_to_conversation(
    input_dict: Dict[str, Any], instruction: str = INSTRUCTION
) -> List[Dict]:
    content = []

    if ("text" in input_dict and "video" in input_dict) or (
        "text" not in input_dict and "video" not in input_dict
    ):
        raise ValueError("Exactly one of text or video must be provided")

    text = input_dict.get("text")
    video = input_dict.get("video")

    if text:
        content.append({"type": "text", "text": text})

    if video:
        content.append({"type": "video", "video": video})

    return [
        {"role": "system", "content": [{"type": "text", "text": instruction}]},
        {"role": "user", "content": content},
    ]


def prepare_vllm_inputs(
    input_dict: Dict[str, Any],
    llm,
    instruction: str = INSTRUCTION,
) -> Dict[str, Any]:
    from qwen_vl_utils import process_vision_info

    conversation = format_input_to_conversation(input_dict, instruction)

    prompt_text = llm.llm_engine.tokenizer.apply_chat_template(
        conversation, tokenize=False, add_generation_prompt=True
    )

    _, video_inputs, video_kwargs = process_vision_info(
        conversation,
        image_patch_size=16,
        return_video_kwargs=True,
        return_video_metadata=True,
    )

    mm_data = {}
    if video_inputs:
        mm_data["video"] = video_inputs

    return {
        "prompt": prompt_text,
        "multi_modal_data": mm_data,
        "mm_processor_kwargs": video_kwargs,
    }


def get_text_embedding(text: str) -> list[float]:
    response = requests.post(
        f"{EMBED_URL}/v1/embeddings",
        json={"inputs": [{"text": text}], "instruction": INSTRUCTION},
        timeout=60,
    )
    response.raise_for_status()
    return response.json()["data"][0]["embedding"]


@app.cls(
    gpu="A100",
    image=vllm_image,
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
    },
    timeout=10 * MINUTES,
    scaledown_window=15 * MINUTES,
    max_containers=10,
)
class EmbeddingServer:
    @modal.enter()
    def start_engine(self):
        from vllm import LLM, EngineArgs
        from fastapi import FastAPI
        from pydantic import BaseModel

        logger.info(f"Loading {MODEL_NAME}")
        engine_args = EngineArgs(
            model=MODEL_NAME,
            runner="pooling",
            dtype="bfloat16",
            trust_remote_code=True,
            attention_backend="flashinfer",
            max_model_len=4096 * 10,
        )
        self.llm = LLM(**vars(engine_args))

        self.llm.embed("Warm up the model")
        

        self.web_app = FastAPI()

        class EmbedInput(BaseModel):
            text: Optional[str] = None
            video: Optional[str] = None

        class EmbedRequest(BaseModel):
            inputs: List[EmbedInput]
            instruction: str = INSTRUCTION

        @self.web_app.get("/health")
        def health():
            return {"status": "ok"}

        @self.web_app.post("/v1/embeddings")
        def create_embeddings(request: EmbedRequest):
            vllm_inputs = [
                prepare_vllm_inputs(
                    inp.model_dump(exclude_none=True),
                    self.llm,
                    request.instruction,
                )
                for inp in request.inputs
            ]

            outputs = self.llm.embed(vllm_inputs)

            return {
                "object": "list",
                "model": MODEL_NAME,
                "data": [
                    {
                        "index": i,
                        "object": "embedding",
                        "embedding": out.outputs.embedding,
                    }
                    for i, out in enumerate(outputs)
                ],
            }

    @modal.exit()
    def stop(self):
        logger.info("Shutting down LLM engine")
        del self.llm

    @modal.asgi_app()
    def serve(self):
        return self.web_app


# ---------------------------------------------------------------------------
# Embedding batch pipeline
# ---------------------------------------------------------------------------


class ClipEmbedding:
    def __init__(self, url: str, embedding: list[float]):
        self.url = url
        self.embedding = embedding


@app.function(
    image=orchestrator_image,
    volumes={EMBEDDING_STORE_DIR: embedding_store_vol},
    timeout=60 * MINUTES,
)
def embed_and_save_batch(video_urls: list[str], batch_index: int):
    import pickle
    from io import BytesIO

    import requests
    logger.info(f"Embedding batch {batch_index} started ({len(video_urls)} clips)")

    payload = {
        "inputs": [{"video": url} for url in video_urls],
        "instruction": INSTRUCTION,
    }
    response = requests.post(
        f"{EMBED_URL}/v1/embeddings",
        json=payload,
        timeout=600,
    )
    response.raise_for_status()

    data = sorted(response.json()["data"], key=lambda x: x["index"])
    clip_embeddings = [
        ClipEmbedding(url, item["embedding"]) for url, item in zip(video_urls, data)
    ]

    buf = BytesIO()
    pickle.dump(clip_embeddings, buf)
    buf.seek(0)

    with embedding_store_vol.batch_upload() as batch:
        batch.put_file(buf, f"embeddings_{batch_index}.pkl")
    embedding_store_vol.commit()

    logger.info(f"Saved embeddings for batch {batch_index}")


@app.function(
    image=orchestrator_image,
    volumes={DATASET_DIR: dance_clips_vol},
    timeout=60 * MINUTES,
)
def orchestrate_embedding_jobs(total_clips: int):
    import pandas as pd

    with open(os.path.join(DATASET_DIR, CLIPS_FILE_NAME), "r") as f:
        clips_df = pd.read_csv(f)

    video_batch_offsets = [
        i * EMBED_BATCH_SIZE for i in range(max(total_clips // EMBED_BATCH_SIZE, 1))
    ]

    get_text_embedding("Warm up the model")
    logger.info("Embedding server warmed up")

    embedding_jobs = []
    for batch_index, offset in enumerate(video_batch_offsets):
        batch_clip_urls = (
            clips_df.iloc[offset : offset + EMBED_BATCH_SIZE]
            .iloc[:, 0]
            .astype(str)
            .tolist()
        )
        embedding_jobs.append(embed_and_save_batch.spawn(batch_clip_urls, batch_index))
        logger.info(f"Submitted batch {batch_index} (clips [{offset}:{offset + EMBED_BATCH_SIZE}))")

    modal.FunctionCall.gather(*embedding_jobs)
    logger.info(f"All {len(embedding_jobs)} embedding jobs complete")


# ---------------------------------------------------------------------------
# Query Router
# ---------------------------------------------------------------------------


def wait_for_gpu_server(url: str, timeout_seconds: int = 600, poll_interval: int = 5):
    """Poll the GPU server's /health endpoint until it responds 200."""
    import requests

    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        try:
            resp = requests.get(f"{url}/health", timeout=10)
            if resp.status_code == 200:
                logger.info(f"GPU server healthy at {url}")
                return
        except (requests.ConnectionError, requests.Timeout):
            pass
        logger.info(f"Waiting for GPU server at {url} ...")
        time.sleep(poll_interval)
    raise TimeoutError(
        f"GPU server at {url} did not become healthy within {timeout_seconds}s"
    )


@app.function(
    image=query_router_image,
    volumes={EMBEDDING_STORE_DIR: embedding_store_vol},
    scaledown_window=30 * MINUTES,
)
@modal.asgi_app()
def query_router():
    import glob
    import pickle
    import re

    import numpy as np
    import requests
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse

    class _ClipEmbedding:
        def __init__(self, url: str, embedding: list[float]):
            self.url = url
            self.embedding = embedding

    class _Unpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if name == "ClipEmbedding":
                return _ClipEmbedding
            return super().find_class(module, name)

    pkl_files = glob.glob(os.path.join(EMBEDDING_STORE_DIR, "embeddings_*.pkl"))
    pkl_files.sort(key=lambda p: int(re.search(r"embeddings_(\d+)\.pkl", p).group(1)))

    all_embeddings: list[_ClipEmbedding] = []
    for pkl_file in pkl_files:
        with open(pkl_file, "rb") as f:
            all_embeddings.extend(_Unpickler(f).load())

    if len(all_embeddings) == 0:
        raise ValueError("Embeddings store is empty")

    logger.info(f"Loaded {len(all_embeddings)} embeddings from store")
    embedding_matrix = np.array([e.embedding for e in all_embeddings])
    embedding_urls = [e.url for e in all_embeddings]

    wait_for_gpu_server(EMBED_URL)

    web_app = FastAPI()

    @web_app.get("/health")
    def health():
        return {"status": "ok"}

    @web_app.post("/search")
    async def search(request: Request):
        query = await request.json()
        query_type = query.get("type")
        logger.info(f"Search query: type={query_type!r} text={query.get('text', '')!r}")

        if query_type == "video":
            raise ValueError("Video queries are not yet supported")
        elif query_type != "text":
            raise ValueError("Invalid query type")

        query_embedding = get_text_embedding(query.get("text", ""))

        query_vec = np.array(query_embedding)
        # vectors are alreadynormalized, dotproduct == cosine sim
        similarities = embedding_matrix @ query_vec

        best_idx = int(np.argmax(similarities))
        best_url = embedding_urls[best_idx]

        return JSONResponse(
            content={
                "url": best_url,
                "score": float(similarities[best_idx]),
            }
        )

    return web_app


# ---------------------------------------------------------------------------
# Local entrypoint — downloads dataset, kicks off embedding pipeline
# ---------------------------------------------------------------------------


@app.local_entrypoint()
def main():
    import requests
    import pandas as pd

    logger.info("Starting video clip search pipeline")
    processed_clips_vol = modal.Volume.from_name("dance-clips", create_if_missing=True)

    volume_files = [file.path for file in processed_clips_vol.listdir("/")]

    if CLIPS_FILE_NAME not in volume_files:
        response = requests.get(AIST_DANCE_BASIC_2MBPS_LINK_LIST_FILE_PATH)
        response.raise_for_status()
        basic_dance_clip_df = pd.read_csv(StringIO(response.text), header=None)

        response = requests.get(AIST_DANCE_ADVANCED_2MBPS_LINK_LIST_FILE_PATH)
        response.raise_for_status()
        advanced_dance_clip_df = pd.read_csv(StringIO(response.text), header=None)

        dance_clip_df = pd.concat(
            [basic_dance_clip_df, advanced_dance_clip_df], ignore_index=True
        )

        path_col = dance_clip_df.columns[0]
        front_camera_view_clips = dance_clip_df[
            dance_clip_df[path_col].apply(
                lambda clip_path: pd.notna(clip_path) and "_c01_" in str(clip_path)
            )
        ]
        total_clips_count = len(front_camera_view_clips)

        with tempfile.NamedTemporaryFile(suffix=".csv", delete=True) as f:
            front_camera_view_clips.to_csv(f.name, index=False)
            f.flush()

            with processed_clips_vol.batch_upload() as batch:
                batch.put_file(f.name, "dance-clips.csv")

            processed_clips_vol.commit()
    else:
        local_path = os.path.join(os.path.abspath("."), CLIPS_FILE_NAME)
        subprocess.run(
            [
                "modal",
                "volume",
                "get",
                "dance-clips",
                CLIPS_FILE_NAME,
                local_path,
                "--force",
            ],
            check=True,
        )
        clips_df = pd.read_csv(local_path, header=None)
        total_clips_count = len(clips_df)

    logger.info(f"Launching embedding pipeline for {total_clips_count} clips")
    orchestrate_embedding_jobs.remote(total_clips_count)
