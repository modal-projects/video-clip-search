import os
from typing import Any

import requests


class QueryInferenceClient:
    """Convenience client for the vLLM pooling inference server."""

    def __init__(self, base_url: str, model: str):
        self.base_url = base_url.rstrip("/")
        self.model = model

    def _post_pooling(
        self, payload: dict[str, Any], timeout_s: int = 120
    ) -> dict[str, Any]:
        response = requests.post(
            f"{self.base_url}/pooling",
            json=payload,
            timeout=timeout_s,
        )
        response.raise_for_status()
        return response.json()

    @staticmethod
    def _extract_embedding(pooling_response: dict[str, Any]) -> list[list[float]]:
        embedding = pooling_response["data"][0]["data"]
        return embedding

    def embed_text(self, text: str, timeout_s: int = 120) -> dict[str, Any]:
        payload = {
            "model": self.model,
            "input": [text],
        }
        result = self._post_pooling(payload, timeout_s=timeout_s)
        embedding = self._extract_embedding(result)
        return {
            "embedding": embedding,
            "total_tokens": len(embedding),
        }

    def embed_image(self, image_url: str, timeout_s: int = 120) -> dict[str, Any]:
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": "Represent this image."}],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                },
            ],
        }
        result = self._post_pooling(payload, timeout_s=timeout_s)
        embedding = self._extract_embedding(result)
        return {
            "embedding": embedding,
            "total_tokens": len(embedding),
        }

    def embed_video(self, video_url: str, timeout_s: int = 300) -> dict[str, Any]:
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": "Represent this video."}],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "video_url", "video_url": {"url": video_url}},
                    ],
                },
            ],
        }
        result = self._post_pooling(payload, timeout_s=timeout_s)
        embedding = self._extract_embedding(result)
        return {
            "embedding": embedding,
            "total_tokens": len(embedding),
        }


if __name__ == "__main__":
    inference_base_url = os.environ.get("INFERENCE_BASE_URL")
    if not inference_base_url:
        raise ValueError("INFERENCE_BASE_URL is not set")
    client = QueryInferenceClient(
        base_url=inference_base_url,
        model="TomoroAI/tomoro-colqwen3-embed-4b",
    )

    text_result = client.embed_text(
        "A video of a dancer spinning around like a ballerina"
    )
    print(
        f"Multi vector text embedding: {text_result['embedding']}, total tokens: {text_result['total_tokens']}"
    )

    # Example video embedding inference
    # dance_video_example_url = "https://storage.repository.aist.go.jp/1095/v1.0.0/video/2M/gBR_sFM_c01_d05_mBR5_ch14.mp4"
    # video_result = client.embed_video(dance_video_example_url)
    # print(f"Multi vector video embedding: {video_result['embedding']}")

    # Example image embedding inference
    # image_url = "https://images.com/picture.jpeg"
    # image_result = client.embed_image(image_url)
    # print(f"Multi vector image embedding: {image_result['embedding']}")
