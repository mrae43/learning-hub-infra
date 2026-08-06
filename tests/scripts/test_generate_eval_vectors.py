"""Tests for scripts/generate_eval_vectors.py."""

from __future__ import annotations

from pathlib import Path

import pytest

from core.types.document import DocumentType

# Functions under test
from scripts.generate_eval_vectors import (
    CHUNK_CONFIGS,
    _build_document_entry,
    _build_sidecar,
    _collect_source_documents,
    _resolve_document_type,
    _should_skip_sidecar,
    _sidecar_path,
)

# ── Seam 1: _resolve_document_type ────────────────────────────────────────


@pytest.mark.parametrize(
    ("path_str", "expected"),
    [
        ("eval_corpus/books/ddia-excerpts.md", DocumentType.DOCUMENTATION),
        ("eval_corpus/books/deep-learning-concepts.md", DocumentType.DOCUMENTATION),
        ("eval_corpus/papers/flash-attention.md", DocumentType.DOCUMENTATION),
        ("eval_corpus/papers/vllm-paged-attention.md", DocumentType.DOCUMENTATION),
        ("eval_corpus/synthetic/rag-system-reference.md", DocumentType.DOCUMENTATION),
        ("eval_corpus/synthetic/solid-principles-reference.md", DocumentType.DOCUMENTATION),
        ("eval_corpus/books/ddia-excerpts.pdf", DocumentType.BOOK),
        ("eval_corpus/papers/flash-attention.pdf", DocumentType.PAPER),
    ],
)
def test_resolve_document_type_from_path(path_str: str, expected: DocumentType) -> None:
    assert _resolve_document_type(path_str) == expected


def test_resolve_document_type_unknown_path_raises() -> None:
    with pytest.raises(ValueError, match="Unknown document type for path"):
        _resolve_document_type("eval_corpus/unknown/foo.pdf")


# ── Seam 2: _collect_source_documents ─────────────────────────────────────


def test_collect_source_documents_deduplicates() -> None:
    data = {
        "queries": [
            {"source_document": "eval_corpus/books/ddia-excerpts.md"},
            {"source_document": "eval_corpus/books/ddia-excerpts.md"},
            {"source_document": "eval_corpus/papers/flash-attention.md"},
        ]
    }
    docs = _collect_source_documents(data)
    assert len(docs) == 2
    paths = {d["source_path"] for d in docs}
    types = {d["document_type"] for d in docs}
    assert paths == {"eval_corpus/books/ddia-excerpts.md", "eval_corpus/papers/flash-attention.md"}
    assert types == {DocumentType.DOCUMENTATION}


def test_collect_source_documents_empty_queries() -> None:
    data: dict[str, object] = {"queries": []}
    assert _collect_source_documents(data) == []


def test_collect_source_documents_missing_queries_key() -> None:
    data: dict[str, object] = {}
    assert _collect_source_documents(data) == []


# ── Seam 3: _build_document_entry and _build_sidecar ──────────────────────

_PARENT_A = {
    "content_sha256": "aaa",
    "content": "Parent A content",
    "token_count": 10,
    "metadata": {"page": "Page 1", "section": None},
    "children": [
        {"content_sha256": "c1", "token_count": 5, "position": 0},
        {"content_sha256": "c2", "token_count": 5, "position": 1},
    ],
}

_PARENT_B = {
    "content_sha256": "bbb",
    "content": "Parent B content",
    "token_count": 8,
    "metadata": {"page": "Page 2", "section": "Overview"},
    "children": [
        {"content_sha256": "c3", "token_count": 8, "position": 0},
    ],
}

_DOC_ENTRY = {
    "source_path": "eval_corpus/synthetic/ref.md",
    "document_type": "documentation",
    "parents": [_PARENT_A, _PARENT_B],
}

_CHILD_VECTORS: dict[str, list[float]] = {
    "c1": [0.1, 0.2],
    "c2": [0.3, 0.4],
    "c3": [0.5, 0.6],
}


def test_build_sidecar_structure() -> None:
    config = {"chunk_size": 512, "overlap_ratio": 0.15}
    model_name = "test-model"

    sidecar = _build_sidecar(
        documents=[_DOC_ENTRY],
        config=config,
        model_name=model_name,
        child_vectors=_CHILD_VECTORS,
    )

    assert sidecar["model"] == model_name
    assert sidecar["dimensions"] == 2
    assert sidecar["config"] == config
    assert len(sidecar["documents"]) == 1

    doc = sidecar["documents"][0]
    assert doc["source_path"] == "eval_corpus/synthetic/ref.md"
    assert doc["document_type"] == "documentation"
    assert len(doc["parents"]) == 2

    parent_a = doc["parents"][0]
    assert parent_a["content_sha256"] == "aaa"
    assert parent_a["token_count"] == 10
    assert len(parent_a["children"]) == 2

    child_0 = parent_a["children"][0]
    assert child_0["content_sha256"] == "c1"
    assert child_0["position"] == 0

    assert sidecar["vectors"]["c1"] == [0.1, 0.2]
    assert sidecar["vectors"]["c3"] == [0.5, 0.6]


def test_build_sidecar_empty_documents() -> None:
    sidecar = _build_sidecar(
        documents=[],
        config={"chunk_size": 256, "overlap_ratio": 0.1},
        model_name="m",
        child_vectors={},
    )
    assert sidecar["documents"] == []
    assert sidecar["dimensions"] == 0
    assert sidecar["vectors"] == {}


def test_build_document_entry() -> None:
    entry = _build_document_entry(
        source_path="eval_corpus/papers/paper.md",
        document_type="paper",
        parent_entries=[_PARENT_A, _PARENT_B],
    )
    assert entry["source_path"] == "eval_corpus/papers/paper.md"
    assert entry["document_type"] == "paper"
    assert len(entry["parents"]) == 2


def test_build_document_entry_preserves_input_order() -> None:
    """_build_document_entry passes through parent entries as-is."""
    parent_with_children = {
        "content_sha256": "p1",
        "content": "Parent",
        "token_count": 5,
        "metadata": {},
        "children": [
            {"content_sha256": "a_first", "token_count": 2, "position": 0},
            {"content_sha256": "z_last", "token_count": 3, "position": 1},
        ],
    }
    entry = _build_document_entry(
        source_path="doc.md",
        document_type="book",
        parent_entries=[parent_with_children],
    )
    assert entry["parents"] == [parent_with_children]


# ── Seam 4: _should_skip_sidecar ──────────────────────────────────────────


def test_should_skip_sidecar_when_exists_and_not_forced(tmp_path: Path) -> None:
    sidecar = tmp_path / "exists.json"
    sidecar.write_text("{}")
    assert _should_skip_sidecar(sidecar, force=False) is True


def test_should_not_skip_sidecar_when_forced(tmp_path: Path) -> None:
    sidecar = tmp_path / "exists.json"
    sidecar.write_text("{}")
    assert _should_skip_sidecar(sidecar, force=True) is False


def test_should_not_skip_sidecar_when_not_exists(tmp_path: Path) -> None:
    sidecar = tmp_path / "missing.json"
    assert _should_skip_sidecar(sidecar, force=False) is False


# ── Seam: _sidecar_path ───────────────────────────────────────────────────


def test_sidecar_path_naming() -> None:
    path = _sidecar_path("256_10")
    assert path.name == "eval_vectors_256_10.json"
    assert "eval_corpus" in str(path)


def test_all_configs_have_sidecar_paths() -> None:
    for key in CHUNK_CONFIGS:
        path = _sidecar_path(key)
        assert path.name.startswith("eval_vectors_")
        assert path.name.endswith(".json")


# ── Seam 5: End-to-end with InMemoryEmbedder ──────────────────────────────


def test_chunk_and_build_with_in_memory_embedder(tmp_path: Path) -> None:
    """End-to-end: chunk a test doc, split into children, embed, build sidecar."""
    from core.clients.embeddings_client import InMemoryEmbedder
    from scripts.generate_eval_vectors import (
        _build_parent_entries,
        _chunk_file,
        _embed_child_texts,
    )

    # Create a tiny test markdown file
    doc_path = tmp_path / "test_doc.md"
    doc_path.write_text("# Test Page\n\nSome content here for testing.")

    # Chunk it
    from core.types.document import DocumentType

    parents = _chunk_file(doc_path, DocumentType.DOCUMENTATION)
    assert len(parents) > 0

    chunk_size = 256
    overlap_ratio = 0.1

    # Split and embed
    parent_entries = _build_parent_entries(parents, chunk_size, overlap_ratio)
    all_child_texts = [child["content"] for pe in parent_entries for child in pe["children"]]
    embedder = InMemoryEmbedder(dimension=4, scale=0.01)
    embeddings = _embed_child_texts(all_child_texts, embedder)

    child_vectors: dict[str, list[float]] = {}
    flat_children = [child for pe in parent_entries for child in pe["children"]]
    for child, vec in zip(flat_children, embeddings, strict=True):
        child_vectors[child["content_sha256"]] = vec

    config = {"chunk_size": chunk_size, "overlap_ratio": overlap_ratio}
    sidecar = _build_sidecar(
        documents=[
            _build_document_entry(
                source_path=str(doc_path),
                document_type="documentation",
                parent_entries=parent_entries,
            )
        ],
        config=config,
        model_name="test-model",
        child_vectors=child_vectors,
    )

    assert sidecar["model"] == "test-model"
    assert sidecar["config"]["chunk_size"] == 256
    assert len(sidecar["documents"]) == 1
    assert len(sidecar["vectors"]) == len(flat_children)
    for vec in sidecar["vectors"].values():
        assert len(vec) == 4  # dimension
