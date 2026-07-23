#!/usr/bin/env python3
"""Regenerate embedding vectors for the retrieval eval set.

Usage:

    uv run python scripts/generate_eval_vectors.py

Reads ``retrieval_qa/tests/retrieval_qa/retrieval/eval_set.yaml``,
calls the configured embedding API (``Settings().embedding_model``) for **all**
entries every run, writes the sidecar ``eval_vectors.json`` (keyed by SHA-256
hash) **and** updates ``content_sha256`` in the YAML so both files form a
consistent snapshot.

Edit content in the YAML, run this script, and both files are in sync —
no manual hash updates needed.
"""

import json
from collections import OrderedDict
from pathlib import Path
from typing import Any

import yaml

from core.clients import EmbeddingsClient
from core.config.settings import Settings
from retrieval_qa._utils import _sha256

EVAL_SET_PATH = (
    Path(__file__).resolve().parent.parent
    / "retrieval_qa/tests/retrieval_qa/retrieval/eval_set.yaml"
)
EVAL_VECTORS_PATH = EVAL_SET_PATH.with_name("eval_vectors.json")
BATCH_SIZE = 20


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

    # Warn if YAML declares a different model than settings.
    existing_model = data.get("embedding_model")
    if existing_model is not None and existing_model != model_name:
        print(
            f"Warning: eval_set.yaml declares embedding_model={existing_model!r}, "
            f"but Settings().embedding_model={model_name!r}. "
            f"Overwriting YAML with {model_name!r}."
        )

    # Ensure embedding_model is the first key and matches settings.
    ordered: OrderedDict[str, Any] = OrderedDict()
    ordered["embedding_model"] = model_name
    for k, v in data.items():
        if k != "embedding_model":
            ordered[k] = v
    data = ordered

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
