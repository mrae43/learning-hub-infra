"""Database-backed tests for the dense semantic-neighbor search primitive."""

from unittest.mock import MagicMock

import pytest
from sqlalchemy import func
from sqlalchemy.orm import Session

from core.database.schema import Chunk, Document, Embedding
from core.exceptions import UpstreamUnavailable
from core.retrieval.neighbors import search_dense_neighbors
from core.types.document import DocumentStatus, DocumentType
from core.types.retrieval_config import RetrievalConfig


def _seed_paper(
    session: Session,
    *,
    title: str,
    status: DocumentStatus = DocumentStatus.READY,
    chunks: list[tuple[str, list[float]]] | None = None,
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

    for position, (content, vector) in enumerate(chunks or []):
        chunk = Chunk(
            document_id=document.document_id,
            position=position,
            content=content,
            token_count=max(1, len(content.split())),
            content_search=func.to_tsvector("english", content),
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


def test_dense_neighbors_orders_by_closeness_and_scores_by_cosine_similarity(
    test_session: Session,
) -> None:
    """Neighbors come back closest-first, scored by cosine similarity.

    The score is ``1 - cosine distance``: 1.0 for an identical direction,
    0.0 for orthogonal, -1.0 for opposite — so harness-side similarity
    thresholds (ADR-0019) can compare against a meaningful value.
    """
    query = [1.0] + [0.0] * 1535
    identical = [1.0] + [0.0] * 1535
    orthogonal = [0.0, 1.0] + [0.0] * 1534
    opposite = [-1.0] + [0.0] * 1535
    _seed_paper(
        test_session,
        title="Neighbors",
        chunks=[
            ("opposite chunk", opposite),
            ("orthogonal chunk", orthogonal),
            ("identical chunk", identical),
        ],
    )
    test_session.commit()

    neighbors = search_dense_neighbors(
        session=test_session,
        query_vector=query,
        config=RetrievalConfig(model_name="text-embedding-3-small", ef_search=40, top_k=3),
    )

    assert [n.text for n in neighbors] == [
        "identical chunk",
        "orthogonal chunk",
        "opposite chunk",
    ]
    assert neighbors[0].score == pytest.approx(1.0)
    assert neighbors[1].score == pytest.approx(0.0)
    assert neighbors[2].score == pytest.approx(-1.0)


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
            content_search=func.to_tsvector("english", content),
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


def test_dense_neighbors_respects_top_k_limit(test_session: Session) -> None:
    """At most ``config.top_k`` neighbors are returned even when more exist."""
    vector = [0.3] * 1536
    _seed_paper(
        test_session,
        title="Many",
        chunks=[(f"chunk {i}", vector) for i in range(5)],
    )
    test_session.commit()

    neighbors = search_dense_neighbors(
        session=test_session,
        query_vector=vector,
        config=RetrievalConfig(model_name="text-embedding-3-small", ef_search=40, top_k=3),
    )

    assert len(neighbors) == 3


def test_dense_neighbors_only_returns_chunks_from_ready_documents(
    test_session: Session,
) -> None:
    """Chunks from documents with non-ready status are not queryable."""
    vector = [0.5] * 1536
    _seed_paper(
        test_session,
        title="Still chunking",
        status=DocumentStatus.CHUNKING,
        chunks=[("hidden chunk", vector)],
    )
    _seed_paper(
        test_session,
        title="Ready",
        status=DocumentStatus.READY,
        chunks=[("visible chunk", vector)],
    )
    test_session.commit()

    neighbors = search_dense_neighbors(
        session=test_session,
        query_vector=vector,
        config=RetrievalConfig(model_name="text-embedding-3-small", ef_search=40, top_k=5),
    )

    assert [n.text for n in neighbors] == ["visible chunk"]


def test_dense_neighbors_scopes_embeddings_to_model_name(test_session: Session) -> None:
    """Embeddings under other model names are not considered."""
    _seed_paper(
        test_session,
        title="Single",
        chunks=[("only chunk", [0.5] * 1536)],
    )
    test_session.commit()

    neighbors = search_dense_neighbors(
        session=test_session,
        query_vector=[0.5] * 1536,
        config=RetrievalConfig(model_name="some-other-model", ef_search=40, top_k=5),
    )

    assert neighbors == []


def test_dense_neighbors_empty_corpus_returns_empty_list(test_session: Session) -> None:
    """An empty corpus yields an empty neighbor list, not an exception."""
    neighbors = search_dense_neighbors(
        session=test_session,
        query_vector=[0.5] * 1536,
        config=RetrievalConfig(model_name="text-embedding-3-small", ef_search=40, top_k=5),
    )
    assert neighbors == []


def test_dense_neighbors_carry_parent_chunk_id_through(test_session: Session) -> None:
    """Child neighbors keep their ``parent_chunk_id`` for downstream parent fetch."""
    vector = [0.5] * 1536
    _, parent, children = _seed_parent_child_paper(
        test_session,
        parent_content="enclosing section about vector databases",
        child_contents=["child one about embeddings", "child two about indexes"],
        vectors=[vector, vector],
    )
    test_session.commit()

    neighbors = search_dense_neighbors(
        session=test_session,
        query_vector=vector,
        config=RetrievalConfig(model_name="text-embedding-3-small", ef_search=40, top_k=5),
    )

    assert len(neighbors) == 2
    assert {n.chunk_id for n in neighbors} == {c.chunk_id for c in children}
    assert all(n.parent_chunk_id == parent.chunk_id for n in neighbors)


def test_dense_neighbors_issues_set_local_ef_search_inside_transaction(
    test_session: Session,
) -> None:
    """SET LOCAL hnsw.ef_search is issued with the supplied ef_search value."""
    from sqlalchemy import event

    vector = [0.5] * 1536
    _seed_paper(
        test_session,
        title="Tracked",
        chunks=[("tracked chunk", vector)],
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
        search_dense_neighbors(
            session=test_session,
            query_vector=vector,
            config=RetrievalConfig(model_name="text-embedding-3-small", ef_search=123, top_k=1),
        )
    finally:
        event.remove(test_session.bind, "before_cursor_execute", _before_cursor_execute)

    assert 123 in ef_search_params


def test_dense_neighbors_wraps_db_connection_error_as_upstream_unavailable() -> None:
    """A DB-level OperationalError surfaces as UpstreamUnavailable (maps to 503)."""
    from sqlalchemy.exc import OperationalError

    fake_session = MagicMock()
    fake_session.execute.side_effect = OperationalError(
        statement="SELECT 1",
        params={},
        orig=Exception("connection refused"),
    )
    with pytest.raises(UpstreamUnavailable):
        search_dense_neighbors(
            session=fake_session,
            query_vector=[0.5] * 1536,
            config=RetrievalConfig(model_name="text-embedding-3-small", ef_search=40, top_k=1),
        )


def test_dense_neighbors_wrong_dim_vector_raises_data_error(test_session: Session) -> None:
    """A wrong-dimension vector raises cleanly; pgvector enforces the column type."""
    from sqlalchemy.exc import DataError

    vector = [0.5] * 1536
    _seed_paper(
        test_session,
        title="Dim",
        chunks=[("dim chunk", vector)],
    )
    test_session.commit()

    with pytest.raises(DataError, match="expected 1536 dimensions"):
        search_dense_neighbors(
            session=test_session,
            query_vector=[0.5] * 10,  # wrong dim
            config=RetrievalConfig(model_name="text-embedding-3-small", ef_search=40, top_k=1),
        )
