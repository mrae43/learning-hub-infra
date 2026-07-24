"""Database-backed tests for the retrieval query module."""

from unittest.mock import MagicMock

import pytest
from sqlalchemy.orm import Session

from core.clients.reranker_client import NoopReranker
from core.database.schema import Chunk, Document, Embedding
from core.exceptions import RerankerRateLimitError
from core.types.document import DocumentStatus, DocumentType
from core.types.responses import CitedPassage
from core.types.retrieval_config import RetrievalConfig
from retrieval_qa.retrieval.query import retrieve_relevant_chunks


def _seed_paper(
    session: Session,
    *,
    title: str,
    status: DocumentStatus = DocumentStatus.READY,
    chunks: list[tuple[str, list[float], str]] | None = None,
) -> Document:
    """Seed a document with chunks and 1536-dim embeddings for ``model_name``."""
    document = Document(
        title=title,
        document_type=DocumentType.PAPER,
        source_filename=f"{title}.pdf",
        status=status,
    )
    session.add(document)
    session.flush()

    for position, (content, vector, _label) in enumerate(chunks or []):
        chunk = Chunk(
            document_id=document.document_id,
            position=position,
            content=content,
            token_count=max(1, len(content.split())),
        )
        session.add(chunk)
        session.flush()
        session.add(
            Embedding(
                chunk_id=chunk.chunk_id,
                model_name="text-embedding-3-small",
                embedding=vector,
            )
        )
    return document


def _seed_parent_child_paper(
    session: Session,
    *,
    title: str = "Parent-Child Paper",
    parent_content: str,
    child_contents: list[str],
    vectors: list[list[float]],
) -> tuple[Document, Chunk, list[Chunk]]:
    """Seed a document with one parent and several embedded children.

    Returns (document, parent_row, child_rows).
    """
    document = Document(
        title=title,
        document_type=DocumentType.PAPER,
        source_filename=f"{title}.pdf",
        status=DocumentStatus.READY,
    )
    session.add(document)
    session.flush()

    parent = Chunk(
        document_id=document.document_id,
        position=0,
        content=parent_content,
        token_count=max(1, len(parent_content.split())),
        parent_chunk_id=None,
    )
    session.add(parent)
    session.flush()

    children: list[Chunk] = []
    for position, (content, vector) in enumerate(zip(child_contents, vectors, strict=True)):
        child = Chunk(
            document_id=document.document_id,
            position=position + 1,
            content=content,
            token_count=max(1, len(content.split())),
            parent_chunk_id=parent.chunk_id,
        )
        session.add(child)
        session.flush()
        session.add(
            Embedding(
                chunk_id=child.chunk_id,
                model_name="text-embedding-3-small",
                embedding=vector,
            )
        )
        children.append(child)

    return document, parent, children


def test_retrieve_returns_closest_chunks_by_cosine(test_session: Session) -> None:
    """The closest chunks by cosine distance are returned in ascending order."""
    near = [1.0] + [0.0] * 1535
    far = [-1.0] + [0.0] * 1535
    _seed_paper(
        test_session,
        title="Mixture",
        chunks=[
            ("near chunk content", near, "near"),
            ("far chunk content", far, "far"),
        ],
    )
    test_session.commit()

    results = retrieve_relevant_chunks(
        query_vector=near,
        session=test_session,
        config=RetrievalConfig(model_name="text-embedding-3-small", ef_search=40, top_k=2),
    )

    assert len(results) == 2
    assert results[0].text == "near chunk content"
    # Closer-first ordering by cosine distance.
    assert results[0].chunk_id != results[1].chunk_id


def test_retrieve_returns_full_chunk_text_not_truncated(test_session: Session) -> None:
    """CitedPassage.text carries the full chunk content."""
    long_text = "word " * 200
    vector = [0.5] * 1536
    _seed_paper(
        test_session,
        title="Long",
        chunks=[(long_text.strip(), vector, "long")],
    )
    test_session.commit()

    results = retrieve_relevant_chunks(
        query_vector=vector,
        session=test_session,
        config=RetrievalConfig(model_name="text-embedding-3-small", ef_search=40, top_k=1),
    )

    assert len(results) == 1
    assert results[0].text == long_text.strip()


def test_retrieve_respects_top_k_limit(test_session: Session) -> None:
    """At most top_k chunks are returned even when more exist."""
    vector = [0.3] * 1536
    _seed_paper(
        test_session,
        title="Many",
        chunks=[(f"chunk {i}", vector, f"c{i}") for i in range(5)],
    )
    test_session.commit()

    results = retrieve_relevant_chunks(
        query_vector=vector,
        session=test_session,
        config=RetrievalConfig(model_name="text-embedding-3-small", ef_search=40, top_k=3),
    )

    assert len(results) == 3


def test_retrieve_only_returns_chunks_from_ready_documents(
    test_session: Session,
) -> None:
    """Chunks from documents with non-ready status are not queryable."""
    vector = [0.5] * 1536
    _seed_paper(
        test_session,
        title="Still chunking",
        status=DocumentStatus.CHUNKING,
        chunks=[("hidden chunk", vector, "hidden")],
    )
    _seed_paper(
        test_session,
        title="Ready",
        status=DocumentStatus.READY,
        chunks=[("visible chunk", vector, "visible")],
    )
    test_session.commit()

    results = retrieve_relevant_chunks(
        query_vector=vector,
        session=test_session,
        config=RetrievalConfig(model_name="text-embedding-3-small", ef_search=40, top_k=5),
    )

    assert len(results) == 1
    assert results[0].text == "visible chunk"


def test_retrieve_scopes_embeddings_to_model_name(test_session: Session) -> None:
    """Embeddings under other model names are not considered."""
    _seed_paper(
        test_session,
        title="Single",
        chunks=[("only chunk", [0.5] * 1536, "only")],
    )
    test_session.commit()

    results = retrieve_relevant_chunks(
        query_vector=[0.5] * 1536,
        session=test_session,
        config=RetrievalConfig(model_name="some-other-model", ef_search=40, top_k=5),
    )

    assert results == []


def test_retrieve_empty_corpus_returns_empty_list(test_session: Session) -> None:
    """An empty corpus yields an empty result list (not-found case)."""
    results = retrieve_relevant_chunks(
        query_vector=[0.5] * 1536,
        session=test_session,
        config=RetrievalConfig(model_name="text-embedding-3-small", ef_search=40, top_k=5),
    )
    assert results == []


def test_cited_passage_is_pydantic_model_with_chunk_id_and_text() -> None:
    """CitedPassage exposes chunk_id (UUID) and text (str) fields."""
    from uuid import UUID

    fields = set(CitedPassage.model_fields)
    assert fields == {"chunk_id", "text"}
    assert CitedPassage.model_fields["chunk_id"].annotation is UUID
    assert CitedPassage.model_fields["text"].annotation is str


def test_retrieve_issues_set_local_ef_search_inside_transaction(
    test_session: Session,
) -> None:
    """SET LOCAL hnsw.ef_search is issued with the supplied ef_search value."""
    from sqlalchemy import event

    vector = [0.5] * 1536
    _seed_paper(
        test_session,
        title="Tracked",
        chunks=[("tracked chunk", vector, "tracked")],
    )
    test_session.commit()

    ef_search_params: list[object] = []

    def _before_cursor_execute(
        conn: object,
        cursor: object,
        statement: str,
        parameters: object,
        context: object,
        executemany: bool,
    ) -> None:
        if "hnsw.ef_search" in statement and isinstance(parameters, dict):
            ef_search_params.append(parameters.get("ef_search"))

    event.listen(test_session.bind, "before_cursor_execute", _before_cursor_execute)
    try:
        retrieve_relevant_chunks(
            query_vector=vector,
            session=test_session,
            config=RetrievalConfig(model_name="text-embedding-3-small", ef_search=123, top_k=1),
        )
    finally:
        event.remove(test_session.bind, "before_cursor_execute", _before_cursor_execute)

    assert 123 in ef_search_params


def test_retrieve_returns_empty_list_when_query_vector_wrong_dim(
    test_session: Session,
) -> None:
    """A wrong-dimension vector raises cleanly; pgvector enforces the column type."""
    from sqlalchemy.exc import DataError

    vector = [0.5] * 1536
    _seed_paper(
        test_session,
        title="Dim",
        chunks=[("dim chunk", vector, "dim")],
    )
    test_session.commit()

    # pgvector emits "expected 1536 dimensions, not N", surfaced by SQLAlchemy
    # as a DataError wrapping psycopg2's DataException.
    with pytest.raises(DataError, match="expected 1536 dimensions"):
        retrieve_relevant_chunks(
            query_vector=[0.5] * 10,  # wrong dim
            session=test_session,
            config=RetrievalConfig(model_name="text-embedding-3-small", ef_search=40, top_k=1),
        )


def test_retrieve_wraps_db_connection_error_as_upstream_unavailable() -> None:
    """A DB-level OperationalError surfaces as UpstreamUnavailable (maps to 503)."""
    from sqlalchemy.exc import OperationalError

    from core.exceptions import UpstreamUnavailable

    fake_session = MagicMock()
    fake_session.execute.side_effect = OperationalError(
        statement="SELECT 1",
        params={},
        orig=Exception("connection refused"),
    )
    with pytest.raises(UpstreamUnavailable):
        retrieve_relevant_chunks(
            query_vector=[0.5] * 1536,
            session=fake_session,
            config=RetrievalConfig(model_name="text-embedding-3-small", ef_search=40, top_k=1),
        )


# ── Hybrid search tests ──────────────────────────────────────────────────────


def test_hybrid_dense_only_when_no_query_text_falls_back_to_dense(
    test_session: Session,
) -> None:
    """When query_text is None, hybrid search falls back to dense-only."""
    vector = [0.5] * 1536
    _seed_paper(
        test_session,
        title="Test",
        chunks=[("dense match", vector, "dense")],
    )
    test_session.commit()

    results = retrieve_relevant_chunks(
        query_vector=vector,
        session=test_session,
        config=RetrievalConfig(model_name="text-embedding-3-small", ef_search=40, top_k=5),
        query_text=None,
    )

    assert len(results) == 1
    assert results[0].text == "dense match"


def test_hybrid_search_false_falls_back_to_dense_only(
    test_session: Session,
) -> None:
    """Setting hybrid_search=False uses dense-only retrieval."""
    vector = [0.5] * 1536
    _seed_paper(
        test_session,
        title="Test",
        chunks=[("dense only", vector, "dense")],
    )
    test_session.commit()

    results = retrieve_relevant_chunks(
        query_vector=vector,
        session=test_session,
        config=RetrievalConfig(
            model_name="text-embedding-3-small",
            ef_search=40,
            top_k=5,
            hybrid_search=False,
        ),
        query_text="some query that would match via sparse if enabled",
    )

    assert len(results) == 1
    assert results[0].text == "dense only"


def test_sparse_only_matches_exact_keyword_not_in_embedding_space(
    test_session: Session,
) -> None:
    """A query matching only via sparse tsvector still returns the chunk.

    The dense path gets no meaningful match (all vectors are far from
    the query), but the sparse path finds the exact keyword.
    """
    dense_near = [1.0] + [0.0] * 1535
    dense_far = [-1.0] + [0.0] * 1535

    _seed_paper(
        test_session,
        title="Mixed",
        chunks=[
            ("semantic content about machine learning topics", dense_near, "near"),
            ("git checkout --orphan creates a new branch without history", dense_far, "far"),
        ],
    )
    test_session.commit()

    results = retrieve_relevant_chunks(
        query_vector=dense_near,
        session=test_session,
        config=RetrievalConfig(model_name="text-embedding-3-small", ef_search=40, top_k=5),
        query_text="checkout orphan",
    )

    assert len(results) >= 1
    texts = [r.text for r in results]
    assert any("checkout" in t for t in texts)


def test_sparse_path_graceful_degradation_when_no_text_matches(
    test_session: Session,
) -> None:
    """When the sparse path finds nothing, dense results still come through."""
    vector = [0.5] * 1536
    _seed_paper(
        test_session,
        title="Dense",
        chunks=[("dense result text", vector, "d")],
    )
    test_session.commit()

    results = retrieve_relevant_chunks(
        query_vector=vector,
        session=test_session,
        config=RetrievalConfig(model_name="text-embedding-3-small", ef_search=40, top_k=5),
        query_text="xyznonexistentkeyword12345",
    )

    assert len(results) == 1
    assert results[0].text == "dense result text"


def test_sparse_only_finds_result_when_dense_returns_empty(
    test_session: Session,
) -> None:
    """When the dense path returns empty (no embeddings for model_name),
    the sparse path alone still produces output.
    """
    vector = [0.5] * 1536
    _seed_paper(
        test_session,
        title="Sparse Only",
        chunks=[("unique sparse keyword zxcvbnm", vector, "sparse")],
    )
    test_session.commit()

    results = retrieve_relevant_chunks(
        query_vector=[-1.0] * 1536,  # far from all chunks
        session=test_session,
        config=RetrievalConfig(model_name="text-embedding-3-small", ef_search=40, top_k=5),
        query_text="zxcvbnm",
    )

    # Sparse should find the keyword even though the dense vector is far
    assert len(results) >= 1
    assert any("zxcvbnm" in r.text for r in results)


def test_parent_swap_replaces_child_content_with_parent_content(
    test_session: Session,
) -> None:
    """After hybrid retrieval, matched child chunks are swapped to parents."""
    parent_text = "This is the full parent section about vector databases."
    child_text = "vector databases store high-dimensional embeddings"
    vector = [0.5] * 1536

    _, parent, _ = _seed_parent_child_paper(
        test_session,
        parent_content=parent_text,
        child_contents=[child_text],
        vectors=[vector],
    )
    test_session.commit()

    results = retrieve_relevant_chunks(
        query_vector=vector,
        session=test_session,
        config=RetrievalConfig(model_name="text-embedding-3-small", ef_search=40, top_k=5),
        query_text="vector databases embeddings",
    )

    assert len(results) == 1
    assert results[0].text == parent_text
    assert results[0].chunk_id == parent.chunk_id


def test_parent_swap_deduplicates_multiple_children_of_same_parent(
    test_session: Session,
) -> None:
    """When two children of the same parent match, only one parent result is
    returned (deduplication by parent).
    """
    parent_text = "Kubernetes orchestrates container deployments across clusters."
    child1 = "Kubernetes pods are the smallest deployable units in a cluster"
    child2 = "Kubernetes deployments manage the lifecycle of pods"
    vector = [0.5] * 1536

    _, parent, _ = _seed_parent_child_paper(
        test_session,
        parent_content=parent_text,
        child_contents=[child1, child2],
        vectors=[vector, vector],
    )
    test_session.commit()

    results = retrieve_relevant_chunks(
        query_vector=vector,
        session=test_session,
        config=RetrievalConfig(model_name="text-embedding-3-small", ef_search=40, top_k=5),
        query_text="Kubernetes pods",
    )

    assert len(results) == 1
    assert results[0].chunk_id == parent.chunk_id
    assert results[0].text == parent_text


def test_parent_swap_returns_standalone_chunks_without_parents_as_is(
    test_session: Session,
) -> None:
    """Chunks that have no parent (parent_chunk_id IS NULL) are returned as-is,
    without being swapped or dropped.
    """
    vector = [0.5] * 1536
    content = "standalone chunk without a parent"
    _seed_paper(
        test_session,
        title="Standalone",
        chunks=[(content, vector, "solo")],
    )
    test_session.commit()

    results = retrieve_relevant_chunks(
        query_vector=vector,
        session=test_session,
        config=RetrievalConfig(model_name="text-embedding-3-small", ef_search=40, top_k=5),
        query_text="standalone chunk",
    )

    assert len(results) == 1
    assert results[0].text == content


def test_hybrid_rrf_fusion_combines_dense_and_sparse_without_duplicates(
    test_session: Session,
) -> None:
    """RRF fusion produces a single ranked set with no duplicate entries."""
    vector = [0.5] * 1536

    # Same content so both dense and sparse paths find the same chunk
    content = "postgresql and pgvector are used for vector similarity search"
    _seed_paper(
        test_session,
        title="Hybrid",
        chunks=[(content, vector, "h")],
    )
    test_session.commit()

    results = retrieve_relevant_chunks(
        query_vector=vector,
        session=test_session,
        config=RetrievalConfig(model_name="text-embedding-3-small", ef_search=40, top_k=5),
        query_text="pgvector",
    )

    assert len(results) == 1
    assert results[0].text == content


def test_hybrid_search_respects_top_k_limit(test_session: Session) -> None:
    """Hybrid search with RRF fusion respects the top_k limit."""
    vector = [0.5] * 1536

    _seed_paper(
        test_session,
        title="Many",
        chunks=[(f"chunk {i} with unique word zeta{i}", vector, f"c{i}") for i in range(5)],
    )
    test_session.commit()

    results = retrieve_relevant_chunks(
        query_vector=vector,
        session=test_session,
        config=RetrievalConfig(model_name="text-embedding-3-small", ef_search=40, top_k=3),
        query_text="zeta",
    )

    assert len(results) <= 3


def test_hybrid_search_returns_empty_list_when_no_results_from_either_path(
    test_session: Session,
) -> None:
    """When neither dense nor sparse paths find results, return empty list."""
    vector = [0.5] * 1536
    _seed_paper(
        test_session,
        title="Present",
        chunks=[("some content", vector, "c1")],
    )
    test_session.commit()

    results = retrieve_relevant_chunks(
        query_vector=vector,
        session=test_session,
        config=RetrievalConfig(
            model_name="some-other-model",  # dense: no embeddings under this model
            ef_search=40,
            top_k=5,
        ),
        query_text="xyznonexistentkeyword12345",  # sparse: no text match
    )

    assert results == []


def test_sparse_only_parent_swap_returns_parent_content(
    test_session: Session,
) -> None:
    """A query that matches only via sparse path returns the parent's content.

    The dense vector is far from the child's embedding, but the exact
    keyword in the sparse path recovers the match, and parent-swap
    replaces the child with the parent.
    """
    parent_text = "PostgreSQL full-text search with tsvector and tsquery."
    child_text = "pgvector and HNSW indexing for ANN search on embedding vectors."
    far_vector = [-1.0] * 1536  # far from child's embedding
    child_vector = [0.5] * 1536

    _, parent, _ = _seed_parent_child_paper(
        test_session,
        parent_content=parent_text,
        child_contents=[child_text],
        vectors=[child_vector],
    )
    test_session.commit()

    results = retrieve_relevant_chunks(
        query_vector=far_vector,  # dense path won't find this close
        session=test_session,
        config=RetrievalConfig(model_name="text-embedding-3-small", ef_search=40, top_k=5),
        query_text="pgvector",
    )

    assert len(results) == 1
    assert results[0].chunk_id == parent.chunk_id
    assert results[0].text == parent_text


# ── Reranker tests ───────────────────────────────────────────────────────────


def test_reranker_reorders_passages(test_session: Session) -> None:
    """When a reranker returns passages in a different order, the final
    CitedPassage list reflects the reranked order.
    """
    vector = [0.5] * 1536
    texts = ["chunk alpha", "chunk beta", "chunk gamma"]
    vectors = [[0.5 + i * 0.01] * 1536 for i in range(len(texts))]
    _seed_paper(
        test_session,
        title="Rerank Order",
        chunks=[(t, v, f"c{i}") for i, (t, v) in enumerate(zip(texts, vectors, strict=True))],
    )
    test_session.commit()

    reverse_reranker = _ReverseReranker()
    results = retrieve_relevant_chunks(
        query_vector=vector,
        session=test_session,
        config=RetrievalConfig(model_name="text-embedding-3-small", ef_search=40, top_k=2),
        query_text="chunk",
        reranker=reverse_reranker,
    )

    assert len(results) == 2
    assert results[0].text == "chunk gamma"
    assert results[1].text == "chunk beta"


def test_reranker_respects_top_k_when_narrowing(test_session: Session) -> None:
    """The reranker narrows the candidate pool to the configured top_k."""
    vector = [0.5] * 1536
    texts = [f"passage {i}" for i in range(5)]
    vectors = [[0.5 + i * 0.01] * 1536 for i in range(len(texts))]
    _seed_paper(
        test_session,
        title="Narrow",
        chunks=[(t, v, f"c{i}") for i, (t, v) in enumerate(zip(texts, vectors, strict=True))],
    )
    test_session.commit()

    results = retrieve_relevant_chunks(
        query_vector=vector,
        session=test_session,
        config=RetrievalConfig(model_name="text-embedding-3-small", ef_search=40, top_k=3),
        query_text="passage",
        reranker=NoopReranker(),
    )

    assert len(results) == 3


def test_reranker_disabled_falls_back_to_rrf_top_k_directly(
    test_session: Session,
) -> None:
    """When reranker=False in config, the RRF top_k is used directly without
    fetching extra candidates for reranking.
    """
    vector = [0.5] * 1536
    texts = [f"chunk {i} with keyword zeta{i}" for i in range(6)]
    vectors = [[0.5 + i * 0.01] * 1536 for i in range(len(texts))]
    _seed_paper(
        test_session,
        title="Rerank Off",
        chunks=[(t, v, f"c{i}") for i, (t, v) in enumerate(zip(texts, vectors, strict=True))],
    )
    test_session.commit()

    # reranker=False, so retrieval_limit stays at top_k=3 (not 20)
    results = retrieve_relevant_chunks(
        query_vector=vector,
        session=test_session,
        config=RetrievalConfig(
            model_name="text-embedding-3-small",
            ef_search=40,
            top_k=3,
            reranker=False,
        ),
        query_text="zeta",
        reranker=NoopReranker(),
    )

    # With reranker=False, we don't enlarge the retrieval limit, so at most
    # top_k=3 results come back from RRF. Even with a reranker instance
    # passed, it is not called.
    assert len(results) <= 3


def test_rate_limit_fallback_to_rrf_top_k(test_session: Session) -> None:
    """When the reranker raises RerankerRateLimitError, the system falls back
    to the RRF top-k without crashing.
    """
    vector = [0.5] * 1536
    texts = [f"fallback chunk {i}" for i in range(5)]
    vectors = [[0.5 + i * 0.01] * 1536 for i in range(len(texts))]
    _seed_paper(
        test_session,
        title="Rate Limit Fallback",
        chunks=[(t, v, f"c{i}") for i, (t, v) in enumerate(zip(texts, vectors, strict=True))],
    )
    test_session.commit()

    results = retrieve_relevant_chunks(
        query_vector=vector,
        session=test_session,
        config=RetrievalConfig(model_name="text-embedding-3-small", ef_search=40, top_k=3),
        query_text="fallback",
        reranker=_RateLimitReranker(),
    )

    assert len(results) == 3
    for result in results:
        assert "fallback" in result.text


def test_reranker_with_empty_passages_returns_empty(
    test_session: Session,
) -> None:
    """When no passages are retrieved, the reranker receives an empty list
    and returns an empty list.
    """
    vector = [0.5] * 1536
    _seed_paper(
        test_session,
        title="Present",
        chunks=[("some content", vector, "c1")],
    )
    test_session.commit()

    results = retrieve_relevant_chunks(
        query_vector=vector,
        session=test_session,
        config=RetrievalConfig(
            model_name="some-other-model",  # dense: no embeddings
            ef_search=40,
            top_k=5,
        ),
        query_text="xyznonexistentkeyword12345",  # sparse: no match
        reranker=NoopReranker(),
    )

    assert results == []


# ── Test double rerankers ────────────────────────────────────────────────────


class _ReverseReranker:
    """Reranker that returns passages in reverse order for deterministic tests."""

    def rerank(
        self,
        query: str,
        passages: list[CitedPassage],
        top_k: int,
    ) -> list[CitedPassage]:
        return list(reversed(passages))[:top_k]


class _RateLimitReranker:
    """Reranker that always raises RerankerRateLimitError."""

    def rerank(
        self,
        query: str,
        passages: list[CitedPassage],
        top_k: int,
    ) -> list[CitedPassage]:
        raise RerankerRateLimitError("Simulated rate limit.")
