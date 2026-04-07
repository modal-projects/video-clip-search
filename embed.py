import logging
import os
import sys
import tempfile
import time
import requests

import modal


from typing import Any, Dict, List, Optional, Tuple, Union
from io import BytesIO, StringIO


logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    stream=sys.stdout,
)

app = modal.App("video-clip-embed")

BATCH_SIZE = 300
CLIPS_DIR = "/root/clips"
CLIPS_FILE_NAME = "video-clips.csv"
EMBEDDING_STORE_DIR = "/root/embeddings"
GPU = "RTX-PRO-6000"
MINUTES = 60
NUM_NVDEC_UNITS = 4  # For RTX-PRO-6000
# cpu threads per gpu video decoding task
TORCHCODEC_NUM_THREADS = 2
MAX_NUM_VISUAL_TOKENS = 5120

# ---------------------------------------------------------------------------
# Images & Volumes
# ---------------------------------------------------------------------------

downloader_image = modal.Image.debian_slim().uv_pip_install(
    "requests==2.32.3",
    "pandas==3.0.1",
    "torch==2.9.0",
    "torchvision==0.24.0",
    "qwen-vl-utils==0.0.14",
)

with downloader_image.imports():
    import pandas as pd
    import requests

vllm_image = (
    modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.13")
    .entrypoint([])
    .apt_install("ffmpeg")
    .uv_pip_install(
        "vllm==0.17.0",
        "huggingface-hub==0.36.0",
        "qwen-vl-utils==0.0.14",
        "pandas==3.0.1",
        "numpy==2.0.0",
        "pyarrow==23.0.1",
    )
    .run_commands(
        "pip install torchcodec==0.10.0 --index-url=https://download.pytorch.org/whl/cu129"
    )
    .uv_pip_install("torch==2.10.0", "torchvision==0.25.0", "torchaudio==2.10.0")
    .env(
        {
            "HF_XET_HIGH_PERFORMANCE": "1",
            "FORCE_QWENVL_VIDEO_READER": "torchcodec",
            "TORCHCODEC_NUM_THREADS": str(TORCHCODEC_NUM_THREADS),
        }
    )
)
with vllm_image.imports():
    from vllm import LLM, EngineArgs
    import torch
    from qwen_vl_utils.vision_process import (
        FRAME_FACTOR,
        MODEL_SEQ_LEN,
        SPATIAL_MERGE_SIZE,
        VIDEO_MAX_TOKEN_NUM,
        VIDEO_MIN_TOKEN_NUM,
        calculate_video_frame_range,
        extract_vision_info,
        smart_nframes,
        smart_resize,
    )
    from torchcodec.decoders import VideoDecoder
    from torchvision import transforms
    from torchvision.transforms import InterpolationMode

orchestrator_image = modal.Image.debian_slim().uv_pip_install(
    "pandas==3.0.1",
    "pyarrow==23.0.1",
    "requests==2.32.3",
    "torch==2.9.0",
    "torchvision==0.24.0",
    "qwen-vl-utils==0.0.14",
)

with orchestrator_image.imports():
    import pandas as pd


# Volume to store video clips
clips_vol = modal.Volume.from_name("video-clips", create_if_missing=True)
# Volume to store video embeddings
embedding_store_vol = modal.Volume.from_name(
    "colqwen3-video-embeddings", create_if_missing=True
)
# Volume for Hugging Face cache of model weights
hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
# Volume for vLLM cache
vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)


# ---------------------------------------------------------------------------
# Local entrypoint: usage `modal run --detach embed.py`
# ---------------------------------------------------------------------------


@app.local_entrypoint()
def main():

    # Prepare your dataset: in this case download the dataframe with video URLs and metadata
    prep_dataset.remote()
    logger.info("Dataset prepared")

    # Kick off parallel embedding pipeline
    logger.info("Launching embedding pipeline...")
    orchestrate.remote()


# ---------------------------------------------------------------------------
# Step 1: Orchestrator (runs remotely, coordinates download → embed pipeline)
# ---------------------------------------------------------------------------
#
@app.function(
    image=orchestrator_image,
    volumes={CLIPS_DIR: clips_vol, EMBEDDING_STORE_DIR: embedding_store_vol},
    timeout=60 * MINUTES,
)
def orchestrate():

    clips_df = pd.read_csv(os.path.join(CLIPS_DIR, CLIPS_FILE_NAME))
    clip_urls = clips_df.iloc[:, 0].astype(str).tolist()
    logger.info(f"Total clips to process: {len(clip_urls)}")

    # Creates async parallel jobs for downloading clips
    batch_results = [
        download_clip_batch.spawn(clip_urls[i : i + BATCH_SIZE], i)
        for i in range(0, len(clip_urls), BATCH_SIZE)
    ]

    logger.info(f"Queued {len(batch_results)} batches for embedding")


# ---------------------------------------------------------------------------
# Step 2: Download clips in parallel
# ---------------------------------------------------------------------------
@app.function(
    image=downloader_image,
    volumes={CLIPS_DIR: clips_vol},
    timeout=30 * MINUTES,
)
def download_clip_batch(clip_urls: list[str], batch_index: int):
    """Downloads a batch of clips to a volume, kicks off embedding batch job"""
    from pathlib import Path

    filenames = [Path(clip_url).name for clip_url in clip_urls]
    logger.info(f"Downloading {len(clip_urls)} clips in batch {batch_index}")
    list(download_clip.map(clip_urls))

    clips_vol.commit()

    # Begin async embedding batch job
    embedder = Embedder()
    embedder.embed_batch.spawn(batch_index, filenames, clip_urls)


@app.function(
    image=downloader_image,
    volumes={CLIPS_DIR: clips_vol},
    timeout=30 * MINUTES,
)
@modal.concurrent(target_inputs=8, max_inputs=16)
def download_clip(clip_url: str) -> str:
    """Downloads a clip to a volume"""
    from pathlib import Path

    filename = Path(clip_url).name
    dest = Path(CLIPS_DIR) / filename

    if not dest.exists():
        resp = requests.get(clip_url, timeout=MINUTES)
        resp.raise_for_status()
        dest.write_bytes(resp.content)

    return filename


# ---------------------------------------------------------------------------
# Step 3: Embed videos with vLLM offline inference, save to parquets in Modal volume
# ---------------------------------------------------------------------------

# args for processing video inputs with Qwen series models
VISION_KWARGS = {
    "return_video_kwargs": True,
    "return_video_metadata": True,
}


@app.cls(
    gpu=GPU,
    cpu=NUM_NVDEC_UNITS * TORCHCODEC_NUM_THREADS,
    image=vllm_image,
    volumes={
        CLIPS_DIR: clips_vol,
        EMBEDDING_STORE_DIR: embedding_store_vol,
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
    },
    timeout=30 * MINUTES,
    scaledown_window=10 * MINUTES,
)
class Embedder:
    @modal.enter()
    def start(self):
        VLLM_MAX_MODEL_LEN = 4096 * 20

        logger.info("Loading vLLM embedding model...")
        engine_args = EngineArgs(
            model="TomoroAI/tomoro-colqwen3-embed-4b",
            runner="pooling",
            dtype="bfloat16",
            trust_remote_code=True,
            async_scheduling=True,
            attention_backend="flashinfer",
            max_model_len=VLLM_MAX_MODEL_LEN,
        )
        self.llm = LLM(**vars(engine_args))
        self.processor = self.llm.llm_engine.tokenizer

        # Set max number of visual tokens per video input, per model card guidance
        setattr(self.processor, "max_num_visual_tokens", MAX_NUM_VISUAL_TOKENS)
        image_processor = getattr(self.processor, "image_processor", None)
        if image_processor is not None:
            setattr(image_processor, "max_num_visual_tokens", MAX_NUM_VISUAL_TOKENS)

        # Warm up model
        warump_prompt = ["warm up text input"]
        for _ in range(3):
            self.llm.encode(warump_prompt, pooling_task="token_embed")

        logger.info("vLLM embedding model ready")

    def _prepare_video_inputs(self, video_path: str, vision_kwargs: dict) -> dict:
        """Build one vLLM embed input from a local video path."""

        messages = [
            {
                # Qwen3 recommends providing a system prompt to describe the task. We apply the default.
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "Represent the user's input.",
                    }
                ],
            },
            {
                "role": "user",
                "content": [{"type": "video", "video": f"file://{video_path}"}],
            },
        ]

        processor = self.processor
        prompt_text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # apply Qwen series image patch size
        image_patch_size = getattr(
            getattr(processor, "image_processor", None), "patch_size", None
        )
        if image_patch_size is not None:
            vision_kwargs["image_patch_size"] = image_patch_size

        # decode video frames into tensors
        _, video_inputs, video_kwargs = self._process_video_inputs(
            messages, **vision_kwargs
        )
        if not video_inputs:
            raise ValueError(
                f"process_video_inputs returned no video inputs for {video_path}"
            )

        mm_data = {}
        if video_inputs is not None:
            mm_data["video"] = video_inputs

        # format for multimodal input expected by vLLM
        return {
            "prompt": prompt_text,
            "multi_modal_data": mm_data,
            "mm_processor_kwargs": video_kwargs,
        }

    def _read_video_torchcodec(
        self,
        ele: Dict[str, Any],
        num_ffmpeg_threads: int = 4,
    ) -> Tuple["torch.Tensor", dict, float]:
        """Read video using torchcodec, preferring CUDA decode when supported."""
        video_path = ele["video"]
        st = time.time()
        decode_device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            decoder = VideoDecoder(
                video_path, num_ffmpeg_threads=num_ffmpeg_threads, device="cuda"
            )
        except RuntimeError as err:
            # Torch can report CUDA available while torchcodec/ffmpeg lacks CUDA decode support.
            if "Unsupported device: cuda" not in str(err):
                raise
            decode_device = "cpu"
            logger.warning(
                "torchcodec cuda decode unavailable; falling back to cpu decode "
                f"for {video_path}: {err}"
            )
            decoder = VideoDecoder(video_path, num_ffmpeg_threads=num_ffmpeg_threads)

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
            f"torchcodec decode {decode_device}: {video_path=}, {total_frames=}, {video_fps=}, "
            f"time={time.time() - st:.3f}s"
        )

        video_metadata = dict(
            fps=video_fps,
            frames_indices=idx,
            total_num_frames=total_frames,
            video_backend=f"torchcodec-{decode_device}",
        )
        return video, video_metadata, sample_fps

    def _fetch_and_decode_video(
        self,
        ele: Dict[str, Any],
        image_patch_size: int = 14,
        return_video_sample_fps: bool = False,
        return_video_metadata: bool = False,
    ) -> Union["torch.Tensor", Tuple]:
        """Fetch and preprocess a video, using GPU torchcodec decoding."""
        # QwenVL series specific video frame tokenization parameters
        image_factor = image_patch_size * SPATIAL_MERGE_SIZE
        video_frame_min_pixels = VIDEO_MIN_TOKEN_NUM * image_factor * image_factor
        video_frame_max_pixels = VIDEO_MAX_TOKEN_NUM * image_factor * image_factor

        if not isinstance(ele.get("video"), str):
            raise ValueError("video input must be a string path or URL")

        # number of cpu threads per gpu video decoding task
        num_ffmpeg_threads = int(os.environ.get("TORCHCODEC_NUM_THREADS", 4))
        video, video_metadata, sample_fps = self._read_video_torchcodec(
            ele, num_ffmpeg_threads=num_ffmpeg_threads
        )
        # return video to cpu
        video = video.to("cpu")

        nframes, _, height, width = video.shape
        min_pixels = ele.get("min_pixels", video_frame_min_pixels)
        total_pixels = ele.get(
            "total_pixels", MODEL_SEQ_LEN * image_factor * image_factor * 0.9
        )
        max_pixels = max(
            min(video_frame_max_pixels, total_pixels / nframes * FRAME_FACTOR),
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

    def _process_video_inputs(
        self,
        conversations: Union[List[Dict[str, Any]], List[List[Dict[str, Any]]]],
        return_video_kwargs: bool = False,
        return_video_metadata: bool = False,
        image_patch_size: int = 14,
    ) -> Tuple[
        Optional[List],
        Optional[List["torch.Tensor"]],
        Optional[Dict[str, Any]],
    ]:
        """Process video-only vision info from conversations using GPU decoding."""
        # Get video inputs from conversations
        vision_infos = extract_vision_info(conversations)

        video_inputs = []
        video_sample_fps_list = []

        for vision_info in vision_infos:
            if "video" not in vision_info:
                raise ValueError("Expected video input type")

            video_input, video_sample_fps = self._fetch_and_decode_video(
                vision_info,
                return_video_sample_fps=True,
                image_patch_size=image_patch_size,
                return_video_metadata=return_video_metadata,
            )
            video_sample_fps_list.append(video_sample_fps)
            video_inputs.append(video_input)

        if len(video_inputs) == 0:
            video_inputs = None

        video_kwargs = {"do_sample_frames": False}
        if not return_video_metadata:
            video_kwargs.update({"fps": video_sample_fps_list})

        if return_video_kwargs:
            return None, video_inputs, video_kwargs
        return None, video_inputs

    @modal.method()
    def embed_batch(
        self, batch_index: int, video_filenames: list[str], video_urls: list[str]
    ):
        """Embed a batch of videos and save as a parquet file."""

        from concurrent.futures import ThreadPoolExecutor
        from pathlib import Path

        file_name = f"embeddings_{batch_index}.parquet"
        if os.path.exists(os.path.join(EMBEDDING_STORE_DIR, file_name)):
            logger.info(
                f"Embeddings for batch {batch_index} already exist, skipping..."
            )
            return

        process_start = time.time()
        video_paths = [str(Path(CLIPS_DIR) / filename) for filename in video_filenames]
        with ThreadPoolExecutor(max_workers=NUM_NVDEC_UNITS) as pool:
            vllm_inputs = list(
                pool.map(
                    lambda video_path: self._prepare_video_inputs(
                        video_path, VISION_KWARGS.copy()
                    ),
                    video_paths,
                )
            )

        process_duration_s = time.time() - process_start
        logger.info(
            f"Processed {len(vllm_inputs)} videos in {int(process_duration_s)}s"
        )

        embed_start = time.time()
        outputs = self.llm.encode(vllm_inputs, pooling_task="token_embed")
        duration_s = time.time() - embed_start

        logger.info(
            f"Batch {batch_index}: embedded {len(vllm_inputs)} videos in {int(duration_s)}s"
        )

        # Create parquet rows: one row per token embedding per video (multi-vector)
        rows = []
        for url, output in zip(video_urls, outputs):
            token_embeddings = (
                output.outputs.data
            )  # 2D: (num_visual_tokens, embedding_dim)
            for token_idx, token_vec in enumerate(token_embeddings):
                rows.append(
                    {
                        "url": url,
                        "token_index": token_idx,
                        "embedding": token_vec.tolist(),
                    }
                )

        df = pd.DataFrame(rows)
        buf = BytesIO()
        df.to_parquet(buf, index=False)
        buf.seek(0)

        # Save embeddings to Modal volume
        with embedding_store_vol.batch_upload() as batch:
            batch.put_file(buf, f"embeddings_{batch_index}.parquet")
        embedding_store_vol.commit()
        # Optionally, write embeddings to a vector database here
        logger.info(f"Saved embeddings for batch {batch_index}")

    @modal.exit()
    def stop(self):
        del self.llm


#
# Preprocess dataset, downloading metadata and storing in Modal volume.
# Refactor / remove this to your dataset.
#
@app.function(
    image=downloader_image,
    volumes={CLIPS_DIR: clips_vol},
    timeout=60 * MINUTES,
)
def prep_dataset():

    # AIST Dance Video Database, Basic and Advanced clips
    AIST_DANCE_BASIC_2MBPS_LINK_LIST_FILE_PATH = (
        "https://aistdancedb.ongaaccel.jp/data/video_refined/2M/refined_2M_sBM_url.csv"
    )
    AIST_DANCE_ADVANCED_2MBPS_LINK_LIST_FILE_PATH = (
        "https://aistdancedb.ongaaccel.jp/data/video_refined/2M/refined_2M_sFM_url.csv"
    )
    # Include the front, left, and right side camera views
    INCLUDED_CAMERA_VIEWS = ["c01", "c02", "c08"]

    # Create a single dataframe with all dance clip URLs
    def fetch_dance_clip_list(basic_url: str, advanced_url: str):
        logger.info(f"Fetching basic dance clip list from {basic_url}")
        response = requests.get(basic_url)
        response.raise_for_status()
        basic_dance_clip_df = pd.read_csv(StringIO(response.text), header=None)

        logger.info(f"Fetching advanced dance clip list from {advanced_url}")
        response = requests.get(advanced_url)
        response.raise_for_status()
        advanced_dance_clip_df = pd.read_csv(StringIO(response.text), header=None)

        return pd.concat(
            [basic_dance_clip_df, advanced_dance_clip_df], ignore_index=True
        )

    # Filter clips to only include the front, left, and right side camera views
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

    clips_file_path = os.path.join(CLIPS_DIR, CLIPS_FILE_NAME)

    if not os.path.exists(clips_file_path):
        logger.info("Creating and uploading camera view clip list...")
        # Get data frame of all clips and filter
        all_clips_df = fetch_dance_clip_list(
            AIST_DANCE_BASIC_2MBPS_LINK_LIST_FILE_PATH,
            AIST_DANCE_ADVANCED_2MBPS_LINK_LIST_FILE_PATH,
        )
        camera_view_clips_df = filter_camera_view_clips(all_clips_df)

        with tempfile.NamedTemporaryFile(suffix=".csv", delete=True) as f:
            camera_view_clips_df.to_csv(f.name, index=False)
            f.flush()
            # Upload metadata file to volume
            with clips_vol.batch_upload() as batch:
                batch.put_file(f.name, CLIPS_FILE_NAME)
            clips_vol.commit()

        logger.info(f"Uploaded {len(camera_view_clips_df)} clips to volume")
    else:
        logger.info(f"Clip list {CLIPS_FILE_NAME} already exists in volume")
