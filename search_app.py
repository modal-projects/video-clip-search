import modal

app = modal.App("video-search-app")

MODEL_NAME = "Qwen/Qwen3-VL-Embedding-8B"
MINUTES = 60

vllm_image = (
    modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.13")
    .entrypoint([])
    .uv_pip_install(
        "vllm==0.16.0",
        "huggingface-hub==0.36.0",
        "qwen-vl-utils==0.0.14",
        "torchcodec==0.10.0",  # fastest video loader for Qwen3VL
    )
    .env({"HF_XET_HIGH_PERFORMANCE": "1", "FORCE_QWENVL_VIDEO_READER": "torchcodec"})
)

hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)

VLLM_SERVER_PORT = 8000


@app.function(
    gpu="L40S",
    image=vllm_image,
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
    },
    timeout=10 * MINUTES,
    scaledown_window=15 * MINUTES,
)
@modal.concurrent(max_inputs=30)
@modal.web_server(port=VLLM_SERVER_PORT, startup_timeout=9 * MINUTES)
def serve_query_embbedings():
    import subprocess

    subprocess.Popen(
        " ".join([
            "vllm", "serve", MODEL_NAME,
            "--runner", "pooling",
            "--host", "0.0.0.0",
            "--port", str(VLLM_SERVER_PORT),
            "--max-model-len", str(4096 * 4),
            "--attention-backend", "flashinfer",
        ]),
        shell=True,
    )
