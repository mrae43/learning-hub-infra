#!/usr/bin/env python3
"""Regenerate embedding vectors for the retrieval eval set.

Usage:

    uv run python scripts/generate_eval_vectors.py

Reads ``retrieval_qa/tests/retrieval_qa/retrieval/eval_set.yaml``,
calls the OpenAI embedding API (``text-embedding-3-small``) for **all** entries
every run, writes the sidecar ``eval_vectors.json`` (keyed by SHA-256 hash)
**and** updates ``content_sha256`` in the YAML so both files form a consistent
snapshot.

Edit content in the YAML, run this script, and both files are in sync —
no manual hash updates needed.
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


def _rewrite_content_hashes(data: dict[str, Any]) -> None:
    """Recompute ``content_sha256`` for every chunk and query entry in-place."""
    for doc in data.get("documents", []):
        for chunk in doc.get("chunks", []):
            chunk["content_sha256"] = _sha256(chunk["content"])
    for query in data.get("queries", []):
        query["content_sha256"] = _sha256(query["query"])


def main() -> None:
    """Read eval YAML, recompute content hashes, regenerate embeddings, write both."""
    client = EmbeddingsClient()
    model_name = Settings().embedding_model

    with open(EVAL_SET_PATH) as f:
        data = yaml.safe_load(f)

    _rewrite_content_hashes(data)

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

    if not vectors:
        print("No texts to embed.")
        return

    dimensions = len(vectors[next(iter(vectors))])

    sidecar = {
        "model": model_name,
        "dimensions": dimensions,
        "vectors": vectors,
    }
    with open(EVAL_VECTORS_PATH, "w") as f:
        json.dump(sidecar, f)

    with open(EVAL_SET_PATH, "w") as f:
        yaml.dump(data, f, default_flow_style=None, sort_keys=False, allow_unicode=True)

    print(f"Done. Generated {len(texts_to_embed)} embedding(s), dimensions={dimensions}.")


if __name__ == "__main__":
    main()
