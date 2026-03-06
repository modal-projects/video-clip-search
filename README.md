# Video Clip Search

A Modal example demonstrating video embedding and semantic search — find video clips using natural language queries.

## Concept

This project showcases two flavors of video retrieval:

- **Video Retrieval (V-RET):** Cross-modal search across a corpus of videos. Given a text query like "zebra runs across a plain", find the best matching clip from a large collection.
- **Moment Retrieval (M-RET):** Intra-video search. Given a text query (and optionally a reference video clip), find the best matching segments within a single longer video.

Both tasks are benchmarked on [MMEB (Massive Multimodal Embedding Benchmark)](https://huggingface.co/spaces/TIGER-Lab/MMEB-Leaderboard).

## Model

The primary model is **Qwen3-VL-Embedding-8B** (released January 2026), currently ranked #1 on MMEB for M-RET and #2 for V-RET. It represents a significant leap over earlier video embedding models — roughly 20 points ahead of the prior generation (e.g. VLM2Vec, released July 2025) on both tasks.

**Qwen3-VL-Embedding-2B** is also a strong option: only 3–5 points lower on M/V-RET benchmarks, but substantially smaller and faster — a good trade-off for demo-scale workloads.

## Dataset

This example uses the **[AIST Dance Video Database](https://aistdancedb.ongaaccel.jp/)** — a publicly available academic dataset of street dance videos paired with copyright-cleared music.

Key details:
- **10 dance genres** including basic, advanced, cypher, battle, showcase, and group styles
- **40 professional dancers** performing solo and group routines
- **Multi-camera recordings** with up to 9 simultaneous camera angles per performance
- Both refined (edited) and raw video versions available
- ~515 GB total; subsets can be downloaded via filtered search
- Free to use for research with attribution (cite Tsuchida et al., ISMIR 2019)

For simplicity, this example uses only the basic and advanced clips from the front camera view (c01). This reduced corpus contains ~1,400 clips and is a natural fit for V-RET — searching across clips with text queries like "two dancers doing footwork" or even using a video of yourself dancing as a query to find similar clips.

## Why Modal

- GPU-accelerated embedding inference at scale
- Serverless architecture — embed a large video corpus without managing infrastructure
- Pairs naturally with Modal Volumes for storing embeddings and video data
- Fast iteration between CPU (search/retrieval) and GPU (embedding) workloads
