"""Database-backed tests for the retrieval query module."""

from unittest.mock import MagicMock

import pytest
from sqlalchemy.orm import Session

from core.database.schema import Chunk, Document, Embedding
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
