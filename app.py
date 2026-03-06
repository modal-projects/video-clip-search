from io import StringIO
import os
import subprocess
import tempfile

import modal
from typing import Any, Dict, List


app = modal.App("video-embed-example")

AIST_DANCE_BASIC_2MBPS_LINK_LIST_FILE_PATH = (
    "https://aistdancedb.ongaaccel.jp/data/video_refined/2M/refined_2M_sBM_url.csv"
)
AIST_DANCE_ADVANCED_2MBPS_LINK_LIST_FILE_PATH = (
    "https://aistdancedb.ongaaccel.jp/data/video_refined/2M/refined_2M_sFM_url.csv"
)
EMBED_BATCH_SIZE = 100
MINUTES = 60
CLIPS_FILE_NAME = "dance-clips.csv"
DATASET_DIR = "/root/dance-clips"
EMBEDDING_STORE_DIR = "/root/embeddings"


@app.local_entrypoint()
def main():
    import requests
    import pandas as pd

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

        # Filter for camera view 1 (_c01_ in path, e.g. gBR_sBA_c01_d03_mBR3_ch04.mp4)
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

    orchestrate_embedding_jobs.remote(total_clips_count)


dance_clips_vol = modal.Volume.from_name("dance-clips")
embedding_store_vol = modal.Volume.from_name("dance-clip-embeddings", create_if_missing=True)
orchestrator_image = modal.Image.debian_slim().uv_pip_install("pandas")


@app.function(
    image=orchestrator_image,
    volumes={DATASET_DIR: dance_clips_vol},
    timeout=60 * MINUTES,
)
def orchestrate_embedding_jobs(total_clips: int) -> list[modal.FunctionCall]:
    import pandas as pd

    qwen3_vl_embed = Qwen3VLEmbedderVLlm()
    with open(os.path.join(DATASET_DIR, CLIPS_FILE_NAME), "r") as f:
        clips_df = pd.read_csv(f)

    video_batch_offsets = [
        i * EMBED_BATCH_SIZE for i in range(max(total_clips // EMBED_BATCH_SIZE, 1))
    ]

    embedding_jobs = []

    for batch_index, offset in enumerate(video_batch_offsets):
        batch_start = offset
        batch_end = min(total_clips, offset + EMBED_BATCH_SIZE)
        batch_clip_urls = (
            clips_df.iloc[batch_start:batch_end].iloc[:, 0].astype(str).tolist()
        )

        inputs = [
            {"video": video_url, "instruction": INSTRUCTION}
            for video_url in batch_clip_urls
        ]

        embedding_jobs.append(qwen3_vl_embed.process.spawn(inputs, batch_index))

    modal.FunctionCall.gather(*embedding_jobs)

    return embedding_jobs


@app.function(
    image=orchestrator_image,
    volumes={DATASET_DIR: dance_clips_vol, EMBEDDING_STORE_DIR: embedding_store_vol},
    timeout=60 * MINUTES,
)
def save_embeddings(embeddings: list["ClipEmbedding"], batch_index: int):
    import pickle
    from io import BytesIO

    buf = BytesIO()
    pickle.dump(embeddings, buf)
    buf.seek(0)

    with embedding_store_vol.batch_upload() as batch:
        batch.put_file(buf, f"embeddings_{batch_index}.pkl")


vllm_image = (
    modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.13")
    .entrypoint([])
    .apt_install("ffmpeg")
    .uv_pip_install(
        "vllm==0.16.0",
        "huggingface-hub==0.36.0",
        "qwen-vl-utils==0.0.14",
        "torchcodec==0.9.0",  # fastest video loader for Qwen3VL
    )
    .env({"HF_XET_HIGH_PERFORMANCE": "1", "FORCE_QWENVL_VIDEO_READER": "torchcodec"})
)
hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)
GPU = "A100"

INSTRUCTION = "Represent the user's input."


def format_input_to_conversation(
    input_dict: Dict[str, Any], instruction: str = "Represent the user's input."
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

    if not content:
        content.append({"type": "text", "text": ""})

    return [
        {"role": "system", "content": [{"type": "text", "text": instruction}]},
        {"role": "user", "content": content},
    ]


class ClipEmbedding:
    def __init__(self, url: str, embedding: list[float]):
        self.url = url
        self.embedding = embedding


@app.cls(
    image=vllm_image,
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
    },
    gpu=GPU,
    timeout=60 * MINUTES,
    max_containers=10,
)
class Qwen3VLEmbedderVLlm:
    @modal.enter()
    def startup(self):
        from vllm import LLM
        from vllm import EngineArgs

        engine_args = EngineArgs(
            model="Qwen/Qwen3-VL-Embedding-8B",
            runner="pooling",
            dtype="bfloat16",
            trust_remote_code=True,
            attention_backend="flashinfer",
            max_model_len=4096 * 10,
        )
        self.llm = LLM(**vars(engine_args))

    def prepare_vllm_inputs(
        self,
        input_dict: Dict[str, Any],
        llm,
        instruction: str = "Represent the user's input.",
    ) -> Dict[str, Any]:
        from qwen_vl_utils import process_vision_info

        conversation = format_input_to_conversation(input_dict, instruction)

        prompt_text = llm.llm_engine.tokenizer.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True
        )

        _, video_inputs, video_kwargs = process_vision_info(
            conversation,
            image_patch_size=16,  # 16 for Qwen3-VL
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

    @modal.method()
    def process(self, inputs: list[dict], batch_index: int):
        vllm_inputs = [
            self.prepare_vllm_inputs(
                input,
                self.llm,
                instruction=INSTRUCTION,
            )
            for input in inputs
        ]

        outputs = self.llm.embed(vllm_inputs)
        clip_embeddings = [
            ClipEmbedding(input["video"], out.outputs.embedding)
            for input, out in zip(inputs, outputs)
        ]

        save_embeddings.remote(clip_embeddings, batch_index)

        return clip_embeddings

    @modal.exit()
    def stop(self):
        del self.llm
