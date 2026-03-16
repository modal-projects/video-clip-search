from io import BytesIO, StringIO
import logging
import os
from pathlib import Path
import tempfile
import time

import requests
import pandas as pd
import modal
import torch

logger = logging.getLogger(__name__)

app = modal.App("video-clip-search-embed")

EMBED_BATCH_SIZE = 200
CLIPS_FILE_NAME = "dance-clips.csv"
CLIPS_DIR = "/root/clips"
EMBEDDING_STORE_DIR = "/root/embeddings"
MINUTES = 60

AIST_DANCE_BASIC_2MBPS_LINK_LIST_FILE_PATH = (
    "https://aistdancedb.ongaaccel.jp/data/video_refined/2M/refined_2M_sBM_url.csv"
)
AIST_DANCE_ADVANCED_2MBPS_LINK_LIST_FILE_PATH = (
    "https://aistdancedb.ongaaccel.jp/data/video_refined/2M/refined_2M_sFM_url.csv"
)

MODEL_NAME = "Qwen/Qwen3-VL-Embedding-8B"
VLLM_MAX_MODEL_LEN = 4096 * 10


# ---------------------------------------------------------------------------
# Clip list helpers
# ---------------------------------------------------------------------------


def fetch_dance_clip_list(
    basic_url: str = AIST_DANCE_BASIC_2MBPS_LINK_LIST_FILE_PATH,
    advanced_url: str = AIST_DANCE_ADVANCED_2MBPS_LINK_LIST_FILE_PATH,
):

    print(f"Fetching basic dance clip list from {basic_url}")
    response = requests.get(basic_url)
    response.raise_for_status()
    basic_dance_clip_df = pd.read_csv(StringIO(response.text), header=None)

    print(f"Fetching advanced dance clip list from {advanced_url}")
    response = requests.get(advanced_url)
    response.raise_for_status()
    advanced_dance_clip_df = pd.read_csv(StringIO(response.text), header=None)

    return pd.concat([basic_dance_clip_df, advanced_dance_clip_df], ignore_index=True)


INCLUDED_CAMERA_VIEWS = ["c01", "c02", "c08"]


def filter_camera_view_clips(df):

    path_col = df.columns[0]
    return df[
        df[path_col].apply(
            lambda clip_path: (
                pd.notna(clip_path)
                and any(
                    camera_view in str(clip_path)
                    for camera_view in INCLUDED_CAMERA_VIEWS
                )
            )
        )
    ]


def create_camera_view_clips_df(
    basic_url: str = AIST_DANCE_BASIC_2MBPS_LINK_LIST_FILE_PATH,
    advanced_url: str = AIST_DANCE_ADVANCED_2MBPS_LINK_LIST_FILE_PATH,
):
    """Fetch dance clip list from AIST and return only clips for included camera views."""
    return filter_camera_view_clips(
        fetch_dance_clip_list(basic_url=basic_url, advanced_url=advanced_url)
    )


# ---------------------------------------------------------------------------
# Images & Volumes
# ---------------------------------------------------------------------------

downloader_image = modal.Image.debian_slim().uv_pip_install(
    "requests", "pandas", "torch", "torchvision", "torchaudio", "qwen-vl-utils==0.0.14"
)

vllm_image = (
    modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.13")
    .entrypoint([])
    .apt_install("ffmpeg")
    .uv_pip_install(
        "vllm==0.16.0",
        "huggingface-hub==0.36.0",
        "qwen-vl-utils==0.0.14",
        "torchcodec==0.9.0",
        "pandas",
        "numpy",
        "pyarrow",
        "torch",
        "torchvision",
        "torchaudio",
    )
    .env({"HF_XET_HIGH_PERFORMANCE": "1", "FORCE_QWENVL_VIDEO_READER": "torchcodec"})
    .run_commands(
        [
            "pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu129"
        ]
    )
)

orchestrator_image = modal.Image.debian_slim().uv_pip_install(
    "pandas",
    "pyarrow",
    "requests",
    "torch",
    "torchvision",
    "torchaudio",
    "qwen-vl-utils==0.0.14",
)

clips_vol = modal.Volume.from_name("clips-data", create_if_missing=True)
embedding_store_vol = modal.Volume.from_name(
    "danced-video-embeddings", create_if_missing=True
)
hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)


# ---------------------------------------------------------------------------
# Stage 1: Download clips (CPU, high parallelism)
# ---------------------------------------------------------------------------


@app.function(
    image=downloader_image,
    volumes={CLIPS_DIR: clips_vol},
    timeout=10 * MINUTES,
    max_containers=50,
)
def download_clip(clip_url: str) -> str:
    """Download a single video clip to the clips volume. Skips if already present."""
    import requests
    from pathlib import Path

    filename = Path(clip_url).name
    dest = Path(CLIPS_DIR) / filename

    if dest.exists():
        return filename

    resp = requests.get(clip_url, timeout=120)
    resp.raise_for_status()

    dest.write_bytes(resp.content)
    clips_vol.commit()

    print(f"Downloaded {filename}")
    return filename


# ---------------------------------------------------------------------------
# Stage 2: Embed videos (L40S GPU, offline vLLM)
# ---------------------------------------------------------------------------


@app.cls(
    gpu="RTX-PRO-6000",
    image=vllm_image,
    volumes={
        CLIPS_DIR: clips_vol,
        EMBEDDING_STORE_DIR: embedding_store_vol,
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
    },
    timeout=30 * MINUTES,
    scaledown_window=10 * MINUTES,
    max_containers=5,
)
class Embedder:
    @modal.enter()
    def start(self):
        from vllm import LLM, EngineArgs

        print("Loading vLLM embedding model...")
        engine_args = EngineArgs(
            model=MODEL_NAME,
            runner="pooling",
            dtype="bfloat16",
            trust_remote_code=True,
            max_model_len=VLLM_MAX_MODEL_LEN,
        )
        self.llm = LLM(**vars(engine_args))

        # Warm up
        self.llm.embed([{"prompt": "warm up", "multi_modal_data": None}])
        print("vLLM embedding model ready")

    def _prepare_video_inputs(self, video_path: str) -> dict:
        """Build one vLLM embed input from a local video path."""

        messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "Represent the user's input video of a dance move.",
                    }
                ],
            },
            {
                "role": "user",
                "content": [{"type": "video", "video": f"file://{video_path}"}],
            },
        ]

        processor = self.llm.llm_engine.tokenizer
        prompt_text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        vision_kwargs = {
            "return_video_kwargs": True,
            "return_video_metadata": True,
        }
        image_patch_size = getattr(
            getattr(processor, "image_processor", None), "patch_size", None
        )
        if image_patch_size is not None:
            vision_kwargs["image_patch_size"] = image_patch_size

        image_inputs, video_inputs, video_kwargs = process_vision_info(
            messages, **vision_kwargs
        )
        if not video_inputs:
            raise ValueError(
                f"process_vision_info returned no video inputs for {video_path}"
            )

        mm_data = {}
        if image_inputs is not None:
            mm_data["image"] = image_inputs
        if video_inputs is not None:
            mm_data["video"] = video_inputs

        return {
            "prompt": prompt_text,
            "multi_modal_data": mm_data,
            "mm_processor_kwargs": video_kwargs,
        }

    @modal.method()
    def embed_batch(
        self, batch_index: int, video_filenames: list[str], video_urls: list[str]
    ):
        """Embed a batch of videos and save as a parquet file."""

        video_paths = [str(Path(CLIPS_DIR) / filename) for filename in video_filenames]
        vllm_inputs = [
            self._prepare_video_inputs(video_path) for video_path in video_paths
        ]

        start = time.time()
        outputs = self.llm.embed(vllm_inputs)
        duration_s = time.time() - start

        rows = []
        for url, output in zip(video_urls, outputs):
            rows.append({"url": url, "embedding": output.outputs.embedding})

        print(f"Batch {batch_index}: embedded {len(rows)} videos in {int(duration_s)}s")

        df = pd.DataFrame(rows)
        buf = BytesIO()
        df.to_parquet(buf, index=False)
        buf.seek(0)

        with embedding_store_vol.batch_upload() as batch:
            batch.put_file(buf, f"embeddings_{batch_index}.parquet")
        embedding_store_vol.commit()

        print(f"Saved embeddings for batch {batch_index}")

    @modal.exit()
    def stop(self):
        del self.llm


# ---------------------------------------------------------------------------
# Orchestrator (runs remotely, coordinates download → embed pipeline)
# ---------------------------------------------------------------------------


@app.function(
    image=orchestrator_image,
    volumes={CLIPS_DIR: clips_vol, EMBEDDING_STORE_DIR: embedding_store_vol},
    timeout=60 * MINUTES,
)
def orchestrate():

    clips_df = pd.read_csv(os.path.join(CLIPS_DIR, CLIPS_FILE_NAME))
    clip_urls = clips_df.iloc[:, 0].astype(str).tolist()
    print(f"Total clips to process: {len(clip_urls)}")

    # --- Stage 1: Download ---
    print(f"Downloading {len(clip_urls)} clips...")
    list(download_clip.map(clip_urls))
    print("All clips downloaded")

    # --- Stage 2: Embed ---
    existing = set(os.listdir(EMBEDDING_STORE_DIR))

    batches = []
    for i in range(0, len(clip_urls), EMBED_BATCH_SIZE):
        batch_idx = i // EMBED_BATCH_SIZE
        parquet_name = f"embeddings_{batch_idx}.parquet"
        if parquet_name in existing:
            print(f"{parquet_name} exists, skipping")
            continue

        batch_urls = clip_urls[i : i + EMBED_BATCH_SIZE]
        batch_filenames = [Path(u).name for u in batch_urls]
        batches.append((batch_idx, batch_filenames, batch_urls))

    if not batches:
        print("All batches already embedded")
        return

    print(f"Submitting {len(batches)} embedding batches")
    embedder = Embedder()
    list(embedder.embed_batch.starmap(batches))
    print(f"All {len(batches)} embedding batches complete")


# ---------------------------------------------------------------------------
# Local entrypoint — prepares clip list, kicks off orchestrator
# ---------------------------------------------------------------------------


@app.local_entrypoint()
def main():
    dataset_vol = modal.Volume.from_name("clips-data", create_if_missing=True)
    volume_files = [file.path for file in dataset_vol.listdir("/")]

    if CLIPS_FILE_NAME not in volume_files:
        print("Creating and uploading camera view clip list...")
        camera_view_clips_df = create_camera_view_clips_df()
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=True) as f:
            camera_view_clips_df.to_csv(f.name, index=False)
            f.flush()
            with dataset_vol.batch_upload() as batch:
                batch.put_file(f.name, CLIPS_FILE_NAME)
            dataset_vol.commit()
        print(f"Uploaded {len(camera_view_clips_df)} clips to volume")
    else:
        print("Clip list already exists in volume")

    print("Launching orchestrator...")
    orchestrate.remote()


"""
Video processing utilities with GPU-accelerated torchcodec decoding.

Mirrors qwen_vl_utils.process_vision_info but creates
torchcodec.VideoDecoder(device='cuda') for on-GPU video decoding.
"""

import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple, Union


from qwen_vl_utils.vision_process import (
    FRAME_FACTOR,
    MODEL_SEQ_LEN,
    SPATIAL_MERGE_SIZE,
    VIDEO_MAX_TOKEN_NUM,
    VIDEO_MIN_TOKEN_NUM,
    calculate_video_frame_range,
    ceil_by_factor,
    extract_vision_info,
    fetch_image,
    smart_nframes,
    smart_resize,
)

logger = logging.getLogger(__name__)

TORCHCODEC_NUM_THREADS = int(os.environ.get("TORCHCODEC_NUM_THREADS", 8))


# ---------------------------------------------------------------------------
# GPU torchcodec reader
# ---------------------------------------------------------------------------


def _read_video_torchcodec_gpu(
    ele: Dict[str, Any],
) -> Tuple[torch.Tensor, dict, float]:
    """Read video using torchcodec with GPU-accelerated decoding (device='cuda')."""
    import torch
    from torchvision import transforms
    from torchvision.transforms import InterpolationMode
    from torchcodec.decoders import VideoDecoder

    video_path = ele["video"]
    st = time.time()
    decoder = VideoDecoder(
        video_path, num_ffmpeg_threads=TORCHCODEC_NUM_THREADS, device="cuda"
    )
    video_fps = decoder.metadata.average_fps
    total_frames = decoder.metadata.num_frames
    start_frame, end_frame, total_frames = calculate_video_frame_range(
        ele, total_frames, video_fps
    )
    nframes = smart_nframes(ele, total_frames=total_frames, video_fps=video_fps)
    idx = torch.linspace(start_frame, end_frame, nframes).round().long().tolist()
    sample_fps = nframes / max(total_frames, 1e-6) * video_fps
    video = decoder.get_frames_at(indices=idx).data
    logger.info(
        f"torchcodec GPU: {video_path=}, {total_frames=}, {video_fps=}, "
        f"time={time.time() - st:.3f}s"
    )

    video_metadata = dict(
        fps=video_fps,
        frames_indices=idx,
        total_num_frames=total_frames,
        video_backend="torchcodec",
    )
    return video, video_metadata, sample_fps


# ---------------------------------------------------------------------------
# fetch_video — same as qwen_vl_utils but wired to the GPU reader
# ---------------------------------------------------------------------------


def fetch_video(
    ele: Dict[str, Any],
    image_patch_size: int = 14,
    return_video_sample_fps: bool = False,
    return_video_metadata: bool = False,
) -> Union[torch.Tensor, Tuple]:
    """Fetch and preprocess a video, using GPU torchcodec decoding."""
    import torch
    from torchvision import transforms
    from torchvision.transforms import InterpolationMode
    from concurrent.futures import ThreadPoolExecutor
    import numpy as np

    image_factor = image_patch_size * SPATIAL_MERGE_SIZE
    VIDEO_FRAME_MIN_PIXELS = VIDEO_MIN_TOKEN_NUM * image_factor * image_factor
    VIDEO_FRAME_MAX_PIXELS = VIDEO_MAX_TOKEN_NUM * image_factor * image_factor

    if isinstance(ele["video"], str):
        video, video_metadata, sample_fps = _read_video_torchcodec_gpu(ele)
    else:
        # List-of-frames fallback (CPU path, same as upstream)
        assert isinstance(ele["video"], (list, tuple))
        process_info = ele.copy()
        process_info.pop("type", None)
        process_info.pop("video", None)

        max_workers = min(8, len(ele["video"]))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(
                    fetch_image, {"image": frame, **process_info}, image_patch_size
                )
                for frame in ele["video"]
            ]
            image_list = [f.result() for f in futures]

        nframes = ceil_by_factor(len(image_list), FRAME_FACTOR)
        if len(image_list) < nframes:
            image_list.extend([image_list[-1]] * (nframes - len(image_list)))

        sample_fps = ele.get("sample_fps", 2.0)
        video = torch.stack(
            [torch.from_numpy(np.array(img).transpose(2, 0, 1)) for img in image_list]
        )
        raw_fps = process_info.pop("raw_fps", sample_fps)
        video_metadata = dict(
            fps=raw_fps,
            frames_indices=list(range(len(video))),
            total_num_frames=(nframes / sample_fps) * raw_fps,
        )

    nframes, _, height, width = video.shape
    min_pixels = ele.get("min_pixels", VIDEO_FRAME_MIN_PIXELS)
    total_pixels = ele.get(
        "total_pixels", MODEL_SEQ_LEN * image_factor * image_factor * 0.9
    )
    max_pixels = max(
        min(VIDEO_FRAME_MAX_PIXELS, total_pixels / nframes * FRAME_FACTOR),
        int(min_pixels * 1.05),
    )
    max_pixels_supposed = ele.get("max_pixels", max_pixels)
    if max_pixels_supposed > max_pixels:
        logger.warning(
            f"The given max_pixels[{max_pixels_supposed}] exceeds limit[{max_pixels}]."
        )
    max_pixels = min(max_pixels_supposed, max_pixels)

    if "resized_height" in ele and "resized_width" in ele:
        resized_height, resized_width = smart_resize(
            ele["resized_height"], ele["resized_width"], factor=image_factor
        )
    else:
        resized_height, resized_width = smart_resize(
            height,
            width,
            factor=image_factor,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )

    video = transforms.functional.resize(
        video,
        [resized_height, resized_width],
        interpolation=InterpolationMode.BICUBIC,
        antialias=True,
    ).float()

    final_video = (video, video_metadata) if return_video_metadata else video
    if return_video_sample_fps:
        return final_video, sample_fps
    return final_video


# ---------------------------------------------------------------------------
# process_vision_info — same as qwen_vl_utils but uses our GPU fetch_video
# ---------------------------------------------------------------------------


def process_vision_info(
    conversations: Union[List[Dict[str, Any]], List[List[Dict[str, Any]]]],
    return_video_kwargs: bool = False,
    return_video_metadata: bool = False,
    image_patch_size: int = 14,
) -> Tuple[
    Optional[List],
    Optional[List[Union[torch.Tensor, List]]],
    Optional[Dict[str, Any]],
]:
    """Process vision info from conversations, using GPU video decoding."""
    vision_infos = extract_vision_info(conversations)

    image_inputs = []
    video_inputs = []
    video_sample_fps_list = []

    for vision_info in vision_infos:
        if "image" in vision_info or "image_url" in vision_info:
            image_inputs.append(
                fetch_image(vision_info, image_patch_size=image_patch_size)
            )
        elif "video" in vision_info:
            video_input, video_sample_fps = fetch_video(
                vision_info,
                return_video_sample_fps=True,
                image_patch_size=image_patch_size,
                return_video_metadata=return_video_metadata,
            )
            video_sample_fps_list.append(video_sample_fps)
            video_inputs.append(video_input)
        else:
            raise ValueError("image, image_url or video should in content.")

    if len(image_inputs) == 0:
        image_inputs = None
    if len(video_inputs) == 0:
        video_inputs = None

    video_kwargs = {"do_sample_frames": False}
    if not return_video_metadata:
        video_kwargs.update({"fps": video_sample_fps_list})

    if return_video_kwargs:
        return image_inputs, video_inputs, video_kwargs
    return image_inputs, video_inputs
