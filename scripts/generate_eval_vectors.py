#!/usr/bin/env python3
"""Pre-compute embedding sidecars for chunk-size tuning evaluation.

Usage:

    uv run python scripts/generate_eval_vectors.py [--force-regenerate]

Reads ``eval_corpus/eval_set_tuning.yaml`` for source document paths.  For each
unique source document:

    1. Runs structure-aware parent chunking once (parent chunks are invariant
       across child split sizes).
    2. For each of the three chunk-size configs (256/10%, 512/15%, 1024/20%):
       recursively splits each parent chunk, embeds the resulting children,
       and writes a variant sidecar JSON file.

Sidecar files are written to ``eval_corpus/eval_vectors_{size}_{overlap}.json``.
Existing sidecars are skipped unless ``--force-regenerate`` is passed.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from hashlib import sha256
from pathlib import Path
from typing import Any

import yaml

from core.clients.embeddings_client import Embedder, EmbeddingsClient, InMemoryEmbedder
from core.config.settings import Settings
from core.types.document import DocumentType
from ingestion.splitting import recursive_fixed_size_split
from retrieval_qa.chunking import get_chunker

# ── Constants ───────────────────────────────────────────────────────────────

_REPO_ROOT = Path(__file__).resolve().parent.parent
_TUNING_YAML = _REPO_ROOT / "eval_corpus" / "eval_set_tuning.yaml"
_EVAL_CORPUS_DIR = _REPO_ROOT / "eval_corpus"

CHUNK_CONFIGS: dict[str, dict[str, int | float]] = {
    "256_10": {"chunk_size": 256, "overlap_ratio": 0.10},
    "512_15": {"chunk_size": 512, "overlap_ratio": 0.15},
    "1024_20": {"chunk_size": 1024, "overlap_ratio": 0.20},
}

_BATCH_SIZE = 20


# ── Helpers ─────────────────────────────────────────────────────────────────


def _sha256(text: str) -> str:
    return sha256(text.encode()).hexdigest()


def _resolve_document_type(path_str: str) -> DocumentType:
    """Map a source-document path to its ``DocumentType``.

    Raises:
        ValueError: The path does not match a known subdirectory.
    """
    rel = Path(path_str)
    try:
        prefix = rel.parts[1]  # eval_corpus/<prefix>/...
    except IndexError:
        raise ValueError(f"Unknown document type for path: {path_str}") from None

    mapping = {
        "books": DocumentType.BOOK,
        "papers": DocumentType.PAPER,
        "synthetic": DocumentType.DOCUMENTATION,
    }
    try:
        return mapping[prefix]
    except KeyError:
        raise ValueError(
            f"Unknown document type for path: {path_str} (unrecognised prefix {prefix!r})"
        ) from None


def _collect_source_documents(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract unique source documents from the tuning YAML.

    Returns a list of ``{"source_path": str, "document_type": DocumentType}``.
    """
    queries = data.get("queries", [])
    seen: set[str] = set()
    docs: list[dict[str, Any]] = []
    for q in queries:
        source = q.get("source_document", "")
        if source and source not in seen:
            seen.add(source)
            docs.append(
                {
                    "source_path": source,
                    "document_type": _resolve_document_type(source),
                }
            )
    return docs


def _chunk_file(
    file_path: Path,
    doc_type: DocumentType,
) -> Sequence[Any]:
    """Run the structure-aware chunker for ``doc_type`` on ``file_path``.

    Returns a sequence of ``Chunk`` objects (parent chunks).
    """
    chunker = get_chunker(doc_type)
    return chunker.chunk(file_path)


def _build_parent_entries(
    parents: Sequence[Any],
    chunk_size: int,
    overlap_ratio: float,
) -> list[dict[str, Any]]:
    """Build parent entries with children for a single chunk-size config.

    Each parent chunk is split into child chunks via
    ``recursive_fixed_size_split``.  The result is a list of dicts ready to
    be serialised into the sidecar tree.
    """
    parent_entries: list[dict[str, Any]] = []
    for parent in parents:
        children = recursive_fixed_size_split(
            parent.content,
            parent.token_count,
            chunk_size=chunk_size,
            overlap_ratio=overlap_ratio,
        )
        child_list: list[dict[str, Any]] = []
        for child in children:
            child_list.append(
                {
                    "content_sha256": _sha256(child.content),
                    "content": child.content,
                    "token_count": child.token_count,
                    "position": child.position,
                }
            )
        parent_entries.append(
            {
                "content_sha256": _sha256(parent.content),
                "content": parent.content,
                "token_count": parent.token_count,
                "metadata": parent.metadata.model_dump(),
                "children": child_list,
            }
        )
    return parent_entries


def _embed_child_texts(
    texts: list[str],
    embedder: Embedder,
) -> list[list[float]]:
    """Embed a list of child texts in batches."""
    all_vectors: list[list[float]] = []
    for i in range(0, len(texts), _BATCH_SIZE):
        batch = texts[i : i + _BATCH_SIZE]
        all_vectors.extend(embedder.embed(batch))
    return all_vectors


def _build_document_entry(
    source_path: str,
    document_type: str,
    parent_entries: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build a single document entry for the sidecar."""
    return {
        "source_path": source_path,
        "document_type": document_type,
        "parents": parent_entries,
    }


def _build_sidecar(
    documents: list[dict[str, Any]],
    config: dict[str, int | float],
    model_name: str,
    child_vectors: dict[str, list[float]],
) -> dict[str, Any]:
    """Assemble the full sidecar dict.

    Args:
        documents: Document entries (each with parent-child tree).
        config: The chunk-size config dict.
        model_name: Embedding model name.
        child_vectors: Mapping of child content_sha256 → embedding vector.

    Returns:
        The sidecar dict ready for JSON serialisation.
    """
    dimension = 0
    if child_vectors:
        first = next(iter(child_vectors.values()))
        dimension = len(first)

    return {
        "model": model_name,
        "config": config,
        "dimensions": dimension,
        "documents": documents,
        "vectors": child_vectors,
    }


def _sidecar_path(config_key: str) -> Path:
    """Return the sidecar path for a given config key (e.g. ``256_10``)."""
    return _EVAL_CORPUS_DIR / f"eval_vectors_{config_key}.json"


def _should_skip_sidecar(path: Path, force: bool) -> bool:
    """Return ``True`` if the sidecar already exists and ``force`` is ``False``."""
    return path.exists() and not force


def _chunk_documents(
    documents: list[dict[str, Any]],
) -> dict[str, Sequence[Any]]:
    """Run structure-aware chunking on all source documents.

    Parent chunks are invariant across child split sizes, so this is called
    once and the results are reused for all configs.

    Returns a dict mapping ``source_path`` → sequence of ``Chunk`` objects.
    """
    parent_cache: dict[str, Sequence[Any]] = {}
    for doc_info in documents:
        source_path = doc_info["source_path"]
        doc_type = doc_info["document_type"]
        abs_path = _REPO_ROOT / source_path
        print(f"  chunking: {source_path}")
        parent_cache[source_path] = _chunk_file(abs_path, doc_type)
    return parent_cache


def _process_for_config(
    source_path: str,
    document_type_value: str,
    parents: Sequence[Any],
    chunk_size: int,
    overlap_ratio: float,
    embedder: Embedder,
) -> tuple[dict[str, Any], dict[str, list[float]]]:
    """Split parents, embed children, return a document entry and child vectors.

    Returns a tuple of ``(document_entry_dict, child_vectors_dict)``.
    """
    parent_entries = _build_parent_entries(parents, chunk_size, overlap_ratio)
    all_child_texts = [child["content"] for pe in parent_entries for child in pe["children"]]

    embeddings = _embed_child_texts(all_child_texts, embedder) if all_child_texts else []

    flat_children = [child for pe in parent_entries for child in pe["children"]]
    child_vectors: dict[str, list[float]] = {}
    for child, vec in zip(flat_children, embeddings, strict=True):
        child_vectors[child["content_sha256"]] = vec

    return _build_document_entry(
        source_path=source_path,
        document_type=document_type_value,
        parent_entries=parent_entries,
    ), child_vectors


# ── Main ────────────────────────────────────────────────────────────────────


def build_all_sidecars(
    *,
    force: bool = False,
    use_in_memory: bool = False,
) -> int:
    """Orchestrate the full sidecar generation.

    Algorithm per document:
      1. Run structure-aware parent chunking once (parent chunks are invariant
         across child split sizes).
      2. For each config (256/10%, 512/15%, 1024/20%): recursively split each
         parent chunk, embed the children, write the sidecar.

    Args:
        force: If ``True``, regenerate existing sidecars.
        use_in_memory: If ``True``, use ``InMemoryEmbedder`` instead of the
            real API (useful for testing / offline development).

    Returns:
        Exit code (0 = success).
    """
    if not _TUNING_YAML.exists():
        print(f"Error: {_TUNING_YAML} not found", file=sys.stderr)
        return 1

    model_name = Settings().embedding_model
    embedder: Embedder = InMemoryEmbedder() if use_in_memory else EmbeddingsClient()

    with open(_TUNING_YAML) as f:
        data = yaml.safe_load(f)

    documents = _collect_source_documents(data)
    if not documents:
        print("No documents found in eval_set_tuning.yaml; nothing to do.")
        return 0

    # Phase 1 — chunk all documents once (parent chunks are invariant).
    print("Phase 1: chunking all source documents...")
    parent_cache = _chunk_documents(documents)

    # Phase 2 — for each config, split, embed, and write sidecar.
    for config_key, config in CHUNK_CONFIGS.items():
        sidecar_path = _sidecar_path(config_key)

        if _should_skip_sidecar(sidecar_path, force=force):
            print(
                f"[skip]  {sidecar_path.name} already exists (use --force-regenerate to overwrite)"
            )
            continue

        chunk_size = int(config["chunk_size"])
        overlap_ratio = float(config["overlap_ratio"])

        print(f"Phase 2: {sidecar_path.name}  (size={chunk_size}, overlap={overlap_ratio})")

        all_document_entries: list[dict[str, Any]] = []
        all_child_vectors: dict[str, list[float]] = {}

        for doc_info in documents:
            source_path = doc_info["source_path"]
            doc_type = doc_info["document_type"]
            parents = parent_cache[source_path]
            print(f"  -> {source_path}")
            doc_entry, doc_vectors = _process_for_config(
                source_path=source_path,
                document_type_value=doc_type.value,
                parents=parents,
                chunk_size=chunk_size,
                overlap_ratio=overlap_ratio,
                embedder=embedder,
            )
            all_document_entries.append(doc_entry)
            all_child_vectors.update(doc_vectors)

        sidecar = _build_sidecar(
            documents=all_document_entries,
            config={"chunk_size": chunk_size, "overlap_ratio": overlap_ratio},
            model_name=model_name,
            child_vectors=all_child_vectors,
        )

        sidecar_path.write_text(json.dumps(sidecar, indent=2))
        print(f"  wrote {sidecar_path.name} ({len(all_child_vectors)} children)")

    return 0


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Pre-compute embedding sidecars for chunk-size tuning evaluation.",
    )
    parser.add_argument(
        "--force-regenerate",
        action="store_true",
        help="Overwrite existing sidecar files instead of skipping them.",
    )
    parser.add_argument(
        "--in-memory",
        action="store_true",
        help="Use InMemoryEmbedder instead of the real API (for testing).",
    )
    args = parser.parse_args()

    return build_all_sidecars(force=args.force_regenerate, use_in_memory=args.in_memory)


if __name__ == "__main__":
    sys.exit(main())
