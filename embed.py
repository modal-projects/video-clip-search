from io import StringIO
import logging
import os
import tempfile
import time
import pandas as pd
import requests

import modal


logger = logging.getLogger(__name__)

app = modal.App("video-clip-search-embed")

EMBED_BATCH_SIZE = 50
CLIPS_FILE_NAME = "dance-clips.csv"
DATASET_DIR = "/root/dance-clips"
EMBEDDING_STORE_DIR = "/root/embeddings"
MINUTES = 60

AIST_DANCE_BASIC_2MBPS_LINK_LIST_FILE_PATH = (
    "https://aistdancedb.ongaaccel.jp/data/video_refined/2M/refined_2M_sBM_url.csv"
)
AIST_DANCE_ADVANCED_2MBPS_LINK_LIST_FILE_PATH = (
    "https://aistdancedb.ongaaccel.jp/data/video_refined/2M/refined_2M_sFM_url.csv"
)

EMBED_URL = (
    "https://modal-labs-adamkelch-dev--video-clip-search-servers-embe-e12e88.modal.run"
)
MODEL_NAME = "Qwen/Qwen3-VL-Embedding-8B"




# fetch the dance clip list(s) from the AIST Dance Video Database
def fetch_dance_clip_list(
    basic_url: str = AIST_DANCE_BASIC_2MBPS_LINK_LIST_FILE_PATH,
    advanced_url: str = AIST_DANCE_ADVANCED_2MBPS_LINK_LIST_FILE_PATH,
):

    # Two sets of clips, "Basic" and "Advanced"

    print(f"Fetching basic dance clip list from {basic_url}")

    response = requests.get(basic_url)
    response.raise_for_status()
    basic_dance_clip_df = pd.read_csv(StringIO(response.text), header=None)

    print(f"Fetching advanced dance clip list from {advanced_url}")

    response = requests.get(advanced_url)
    response.raise_for_status()
    advanced_dance_clip_df = pd.read_csv(StringIO(response.text), header=None)

    return pd.concat([basic_dance_clip_df, advanced_dance_clip_df], ignore_index=True)


# front, left, right camera views
INCLUDED_CAMERA_VIEWS = ["c01", "c02", "c08"]


# filter clips to only include the camera views in INCLUDED_CAMERA_VIEWS
def filter_camera_view_clips(df):
    import pandas as pd

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
) -> pd.DataFrame:
    """Fetch dance clip list from AIST and return only clips for included camera views."""
    return filter_camera_view_clips(
        fetch_dance_clip_list(basic_url=basic_url, advanced_url=advanced_url)
    )


# --- Image & Volumes ---

orchestrator_image = modal.Image.debian_slim().uv_pip_install(
    "pandas", "pyarrow", "requests"
)

dance_clips_vol = modal.Volume.from_name("clips-data", create_if_missing=True)
embedding_store_vol = modal.Volume.from_name(
    "video-clip-embeddings", create_if_missing=True
)


# ---------------------------------------------------------------------------
# Embedding batch pipeline
# ---------------------------------------------------------------------------


@app.function(
    image=orchestrator_image,
    timeout=60 * MINUTES,
    volumes={"/root/embeddings": embedding_store_vol},
)
def embed_and_save_batch(video_urls: list[str], batch_index: int):
    from io import BytesIO

    import pandas as pd
    import requests
    import os
    from pathlib import Path

    print(f"Embedding batch {batch_index} started for {len(video_urls)} clips")

    embedding_store_path = Path(EMBEDDING_STORE_DIR)

    batch_path = embedding_store_path / f"embeddings_{batch_index}.parquet"

    if batch_path.exists():
        print(f"Batch {batch_index} already exists, skipping")
        return

    response = requests.post(
        f"{EMBED_URL}/v1/embeddings",
        json={
            "model": MODEL_NAME,
            "input": [
                {"type": "video_url", "video_url": {"url": url}}
                for url in video_urls
            ],
        },
        timeout=600,
    )
    response.raise_for_status()

    data = sorted(response.json()["data"], key=lambda x: x["index"])
    df = pd.DataFrame(
        {"url": video_urls, "embedding": [item["embedding"] for item in data]}
    )

    buf = BytesIO()
    df.to_parquet(buf, index=False)
    buf.seek(0)

    with embedding_store_vol.batch_upload() as batch:
        batch.put_file(buf, f"embeddings_{batch_index}.parquet")
    embedding_store_vol.commit()

    print(f"Saved embeddings for batch {batch_index}")


@app.function(
    image=orchestrator_image,
    volumes={DATASET_DIR: dance_clips_vol},
    timeout=60 * MINUTES,
)
def orchestrate_embedding_jobs():
    import pandas as pd

    with open(os.path.join(DATASET_DIR, CLIPS_FILE_NAME), "r") as f:
        clips_df = pd.read_csv(f)

    total_clips = len(clips_df)
    num_batches = max(total_clips // EMBED_BATCH_SIZE, 1)

    # Each batch is (List[clip URL], int)
    batches = []
    for batch_index in range(num_batches):
        offset = batch_index * EMBED_BATCH_SIZE
        batch_clip_urls = (
            clips_df.iloc[offset : offset + EMBED_BATCH_SIZE]
            .iloc[:, 0]
            .astype(str)
            .tolist()
        )
        batches.append((batch_clip_urls, batch_index))
        print(
            f"Submitted batch {batch_index} (clips [{offset}:{offset + EMBED_BATCH_SIZE}])"
        )

    list(embed_and_save_batch.starmap(batches))
    print(f"All {len(batches)} embedding jobs complete")


# ---------------------------------------------------------------------------
# Local entrypoint — prepares dataset, warms up server, runs embedding jobs
# ---------------------------------------------------------------------------


def _warm_up_embedding_server(timeout_seconds: int = 300, poll_interval: int = 5):
    """POST a test request to the embedding server, blocking until it's ready."""
    import requests

    payload = {
        "model": MODEL_NAME,
        "input": [{"type": "text", "text": "warm up"}],
    }
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        try:
            resp = requests.post(f"{EMBED_URL}/v1/embeddings", json=payload, timeout=30)
            if resp.status_code == 200:
                print("Embedding server is ready")
                return
            print(f"Embedding server returned {resp.status_code}, retrying...")
        except (requests.ConnectionError, requests.Timeout):
            print("Embedding server not yet up, retrying...")
        time.sleep(poll_interval)
    raise TimeoutError(
        f"Embedding server did not become ready within {timeout_seconds}s"
    )


@app.local_entrypoint()
def main():
    processed_clips_vol = modal.Volume.from_name("clips-data", create_if_missing=True)

    volume_files = [file.path for file in processed_clips_vol.listdir("/")]

    if CLIPS_FILE_NAME not in volume_files:
        print("Creating and uploading camera view clip list...")
        camera_view_clips_df = create_camera_view_clips_df()

        with tempfile.NamedTemporaryFile(suffix=".csv", delete=True) as f:
            camera_view_clips_df.to_csv(f.name, index=False)
            f.flush()

            with processed_clips_vol.batch_upload() as batch:
                batch.put_file(f.name, CLIPS_FILE_NAME)

            processed_clips_vol.commit()

    print(f"Volume files: {processed_clips_vol.listdir('/')}")
    print("Warming up embedding server...")
    _warm_up_embedding_server()

    print("Launching embedding pipeline...")
    orchestrate_embedding_jobs.remote()
