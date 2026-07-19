#!/usr/bin/env python3
"""Regenerate embedding vectors for the retrieval eval set.

Usage:

    uv run python scripts/generate_eval_vectors.py

Reads ``retrieval_qa/tests/retrieval_qa/retrieval/eval_set.yaml``,
calls the OpenAI embedding API (``text-embedding-3-small``) for entries whose
SHA-256 hash has changed, and writes updated vectors and hashes back to the
YAML.

Entries whose stored SHA-256 matches the runtime hash of the content are
skipped (incremental mode).
"""

import hashlib
from pathlib import Path
from typing import Any

import yaml

from core.clients import EmbeddingsClient

EVAL_SET_PATH = (
    Path(__file__).resolve().parent.parent
    / "retrieval_qa/tests/retrieval_qa/retrieval/eval_set.yaml"
)
BATCH_SIZE = 20


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _needs_update(entry: dict[str, Any], text_key: str, hash_key: str) -> bool:
    stored_hash = entry.get(hash_key)
    if stored_hash is None:
        return True
    actual_hash = _sha256(str(entry[text_key]))
    return bool(actual_hash != stored_hash)


def main() -> None:
    """Read eval YAML, regenerate embeddings for changed content, write back."""
    client = EmbeddingsClient()

    with open(EVAL_SET_PATH) as f:
        data = yaml.safe_load(f)

    texts_to_embed: list[str] = []
    source_info: list[tuple[dict[str, Any], str, str, str]] = []

    for doc in data["documents"]:
        for chunk in doc["chunks"]:
            if _needs_update(chunk, "content", "content_sha256"):
                texts_to_embed.append(chunk["content"])
                source_info.append((chunk, "content", "embedding", "content_sha256"))

    for query in data["queries"]:
        if _needs_update(query, "query", "content_sha256"):
            texts_to_embed.append(query["query"])
            source_info.append((query, "query", "query_embedding", "content_sha256"))

    if not texts_to_embed:
        print("All vectors are up to date.")
        return

    for i in range(0, len(texts_to_embed), BATCH_SIZE):
        batch = texts_to_embed[i : i + BATCH_SIZE]
        print(f"Embedding batch {i // BATCH_SIZE + 1} ({len(batch)} texts)...")
        vectors = client.embed(batch)
        for j, vector in enumerate(vectors):
            entry, text_key, embedding_key, hash_key = source_info[i + j]
            text = entry[text_key]
            entry[embedding_key] = vector
            entry[hash_key] = _sha256(text)

    with open(EVAL_SET_PATH, "w") as f:
        yaml.dump(data, f, default_flow_style=None, sort_keys=False, width=200)

    print(f"Done. Updated {len(texts_to_embed)} embedding(s).")


if __name__ == "__main__":
    main()
