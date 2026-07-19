#!/usr/bin/env python3
"""Regenerate embedding vectors for the retrieval eval set.

Usage:

    uv run python scripts/generate_eval_vectors.py

Reads ``retrieval_qa/tests/retrieval_qa/retrieval/eval_set.yaml``,
calls the OpenAI embedding API (``text-embedding-3-small``) for **all** entries
every run, writes only the sidecar ``eval_vectors.json`` (keyed by
SHA-256 hash).  Does **not** modify the YAML file.
"""

import hashlib
import json
from pathlib import Path

import yaml

from core.clients import EmbeddingsClient
from core.config.settings import Settings

EVAL_SET_PATH = (
    Path(__file__).resolve().parent.parent
    / "retrieval_qa/tests/retrieval_qa/retrieval/eval_set.yaml"
)
EVAL_VECTORS_PATH = EVAL_SET_PATH.with_name("eval_vectors.json")
BATCH_SIZE = 20


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def main() -> None:
    """Read eval YAML, regenerate all embeddings, write sidecar JSON only."""
    client = EmbeddingsClient()
    model_name = Settings().embedding_model
    dimensions = 1536

    with open(EVAL_SET_PATH) as f:
        data = yaml.safe_load(f)

    texts_to_embed: list[str] = []

    for doc in data["documents"]:
        for chunk in doc["chunks"]:
            texts_to_embed.append(chunk["content"])

    for query in data["queries"]:
        texts_to_embed.append(query["query"])

    vectors: dict[str, list[float]] = {}

    for i in range(0, len(texts_to_embed), BATCH_SIZE):
        batch = texts_to_embed[i : i + BATCH_SIZE]
        print(f"Embedding batch {i // BATCH_SIZE + 1} ({len(batch)} texts)...")
        batch_vectors = client.embed(batch)
        for j, vector in enumerate(batch_vectors):
            content_hash = _sha256(texts_to_embed[i + j])
            vectors[content_hash] = vector

    sidecar = {
        "model": model_name,
        "dimensions": dimensions,
        "vectors": vectors,
    }
    with open(EVAL_VECTORS_PATH, "w") as f:
        json.dump(sidecar, f)

    print(f"Done. Generated {len(texts_to_embed)} embedding(s).")


if __name__ == "__main__":
    main()
