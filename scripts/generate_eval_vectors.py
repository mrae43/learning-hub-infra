#!/usr/bin/env python3
"""Regenerate embedding vectors for the retrieval eval set.

Usage:

    uv run python scripts/generate_eval_vectors.py

Reads ``retrieval_qa/tests/retrieval_qa/retrieval/eval_set.yaml``,
calls the OpenAI embedding API (``text-embedding-3-small``) for **all** entries
every run, writes both ``eval_vectors.json`` (sidecar) and updated inline
vectors back to the YAML.

This is an expand step -- both files carry the same vector data so nothing
consuming the YAML breaks yet.
"""

import hashlib
import json
from pathlib import Path
from typing import Any

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
    """Read eval YAML, regenerate all embeddings, write YAML + sidecar JSON."""
    client = EmbeddingsClient()
    model_name = Settings().embedding_model
    dimensions = 1536

    with open(EVAL_SET_PATH) as f:
        data = yaml.safe_load(f)

    texts_to_embed: list[str] = []
    source_info: list[tuple[dict[str, Any], str, str, str]] = []

    for doc in data["documents"]:
        for chunk in doc["chunks"]:
            texts_to_embed.append(chunk["content"])
            source_info.append((chunk, "content", "embedding", "content_sha256"))

    for query in data["queries"]:
        texts_to_embed.append(query["query"])
        source_info.append((query, "query", "query_embedding", "content_sha256"))

    vectors: dict[str, list[float]] = {}

    for i in range(0, len(texts_to_embed), BATCH_SIZE):
        batch = texts_to_embed[i : i + BATCH_SIZE]
        print(f"Embedding batch {i // BATCH_SIZE + 1} ({len(batch)} texts)...")
        batch_vectors = client.embed(batch)
        for j, vector in enumerate(batch_vectors):
            entry, text_key, embedding_key, hash_key = source_info[i + j]
            text = entry[text_key]
            content_hash = _sha256(text)
            entry[embedding_key] = vector
            entry[hash_key] = content_hash
            vectors[content_hash] = vector

    with open(EVAL_SET_PATH, "w") as f:
        yaml.dump(data, f, default_flow_style=None, sort_keys=False, width=200)

    sidecar = {
        "model": model_name,
        "dimensions": dimensions,
        "vectors": vectors,
    }
    with open(EVAL_VECTORS_PATH, "w") as f:
        json.dump(sidecar, f)

    print(f"Done. Updated {len(texts_to_embed)} embedding(s).")


if __name__ == "__main__":
    main()
