import modal
import numpy as np

app = modal.App("video-query-router")


query_router_image = modal.Image.debian_slim().uv_pip_install(
    "fastapi", "requests", "numpy"
)
embedding_store_vol = modal.Volume.from_name("dance-clip-embeddings")
MODEL_NAME = "Qwen/Qwen3-VL-Embedding-8B"
QUERY_GPU_SERVER_URL = "https://modal-labs-adamkelch-dev--video-search-app-serve-query-e-bc4760.modal.run"
EMBEDDING_STORE_DIR = "/root/embeddings"
MINUTES = 60

@app.function(
    image=query_router_image,
    volumes={EMBEDDING_STORE_DIR: embedding_store_vol},
    scaledown_window=30 * MINUTES,
)
@modal.asgi_app()
def query_router_app():
    import os
    import re
    import glob
    import pickle

    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse

    class ClipEmbedding:
        def __init__(self, url: str, embedding: list[float]):
            self.url = url
            self.embedding = embedding

    class _Unpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if name == "ClipEmbedding":
                return ClipEmbedding
            return super().find_class(module, name)

    pkl_files = glob.glob(os.path.join(EMBEDDING_STORE_DIR, "embeddings_*.pkl"))
    pkl_files.sort(key=lambda p: int(re.search(r"embeddings_(\d+)\.pkl", p).group(1)))

    all_embeddings = []
    for pkl_file in pkl_files:
        with open(pkl_file, "rb") as f:
            all_embeddings.extend(_Unpickler(f).load())

    if len(all_embeddings) == 0:
        raise ValueError("Embeddings store is empty")

    embedding_matrix = np.array([e.embedding for e in all_embeddings])
    embedding_urls = [e.url for e in all_embeddings]
    norms = np.linalg.norm(embedding_matrix, axis=1, keepdims=True)
    embedding_matrix = embedding_matrix / norms

    def get_text_embedding(text: str) -> list[float]:
        import requests

        response = requests.post(
            f"{QUERY_GPU_SERVER_URL}/v1/embeddings",
            json={
                "model": MODEL_NAME,
                "input": text,
            },
        )
        response.raise_for_status()
        return response.json()["data"][0]["embedding"]

    # prewarm the model container
    get_text_embedding("Warm up the model")

    web_app = FastAPI()

    @web_app.post("/search")
    async def search(request: Request):
        query = await request.json()
        query_type = query.get("type")

        if query_type == "video":
            raise ValueError("Video queries are not yet supported")
        elif query_type != "text":
            raise ValueError("Invalid query type")

        query_embedding = get_text_embedding(query.get("text", ""))

        query_vec = np.array(query_embedding)
        query_vec = query_vec / np.linalg.norm(query_vec)

        similarities = embedding_matrix @ query_vec

        best_idx = int(np.argmax(similarities))
        best_url = embedding_urls[best_idx]

        return JSONResponse(content={
            "url": best_url,
            "score": float(similarities[best_idx]),
        })

    return web_app
