import base64
import numpy as np
import requests

VLLM_URL = "https://modal-labs-adamkelch-dev--video-clip-search-servers--134493-dev.modal.run"
IMAGE_PATH = "/Users/adamkelch/Modal/bird.png"

with open(IMAGE_PATH, "rb") as f:
    image_base64 = base64.b64encode(f.read()).decode("utf-8")

image_response = requests.post(
    f"{VLLM_URL}/v1/embeddings",
    json={
        "model": "Qwen/Qwen3-VL-Embedding-8B",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_base64}"
                        },
                    }
                ],
            }
        ],
        "encoding_format": "float",
    },
)
image_response.raise_for_status()
image_embedding = np.array(image_response.json()["data"][0]["embedding"])

text_response = requests.post(
    f"{VLLM_URL}/v1/embeddings",
    json={
        "model": "Qwen/Qwen3-VL-Embedding-8B",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "A robin sits on a tree branch in the woods",
                    }
                ],
            }
        ],
        "encoding_format": "float",
    },
)
text_response.raise_for_status()
text_embedding = np.array(text_response.json()["data"][0]["embedding"])

cosine_sim = np.dot(image_embedding, text_embedding) / (
    np.linalg.norm(image_embedding) * np.linalg.norm(text_embedding)
)
print(f"Cosine similarity: {cosine_sim:.6f}")
