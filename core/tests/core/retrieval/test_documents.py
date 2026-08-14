"""Database-backed tests for the document-relative text lookup primitive."""

from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from sqlalchemy import func
from sqlalchemy.orm import Session

from core.database.schema import Chunk, Document, Embedding
from core.exceptions import UpstreamUnavailable
from core.retrieval.documents import fetch_document_chunks
from core.types.document import DocumentStatus, DocumentType
from core.types.responses import CitedPassage


def _seed_parent_child_paper(
    session: Session,
    *,
    title: str = "Parent-Child Paper",
    status: DocumentStatus = DocumentStatus.READY,
    parent_contents: list[str],
) -> tuple[Document, list[Chunk], list[Chunk]]:
    """Seed a document with several parents and one child per parent.

    Returns (document, parent_rows, child_rows).
    """
    document = Document(
        title=title,
        document_type=DocumentType.PAPER,
        source_filename=f"{title}.pdf",
        status=status,
    )
    session.add(document)
    session.flush()

    parents: list[Chunk] = []
    children: list[Chunk] = []
    for position, content in enumerate(parent_contents):
        parent = Chunk(
            document_id=document.document_id,
            position=position,
            content=content,
            token_count=max(1, len(content.split())),
            parent_chunk_id=None,
        )
        session.add(parent)
        session.flush()
        parents.append(parent)

        child = Chunk(
            document_id=document.document_id,
            position=position + len(parent_contents),
            content=f"child of: {content}",
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
                embedding=[0.5] * 1536,
            )
        )
        children.append(child)

    return document, parents, children


def test_fetch_document_chunks_returns_parent_chunks_in_document_order(
    test_session: Session,
) -> None:
    """Parent chunks come back in position order, children are excluded."""
    contents = ["first section", "second section", "third section"]
    _, parents, _ = _seed_parent_child_paper(test_session, parent_contents=contents)
    test_session.commit()

    chunks = fetch_document_chunks(
        session=test_session,
        document_id=parents[0].document_id,
        limit=10,
    )

    assert chunks is not None
    assert chunks == [CitedPassage(chunk_id=p.chunk_id, text=p.content) for p in parents]
    assert [c.text for c in chunks] == contents


def test_fetch_document_chunks_respects_limit(test_session: Session) -> None:
    """At most ``limit`` parent chunks are returned, still in document order."""
    contents = [f"section {i}" for i in range(5)]
    _, parents, _ = _seed_parent_child_paper(test_session, parent_contents=contents)
    test_session.commit()

    chunks = fetch_document_chunks(
        session=test_session,
        document_id=parents[0].document_id,
        limit=2,
    )

    assert chunks is not None
    assert [c.text for c in chunks] == ["section 0", "section 1"]


def test_fetch_document_chunks_returns_none_for_unknown_document(
    test_session: Session,
) -> None:
    """A document_id that matches no document is an unresolvable anchor."""
    assert fetch_document_chunks(session=test_session, document_id=uuid4(), limit=5) is None


def test_fetch_document_chunks_returns_none_when_document_not_ready(
    test_session: Session,
) -> None:
    """Chunks of a non-ready document are invisible (ADR-0014 visibility gate)."""
    _, parents, _ = _seed_parent_child_paper(
        test_session,
        title="Still embedding",
        status=DocumentStatus.EMBEDDING,
        parent_contents=["hidden section"],
    )
    test_session.commit()

    assert (
        fetch_document_chunks(
            session=test_session,
            document_id=parents[0].document_id,
            limit=5,
        )
        is None
    )


def test_fetch_document_chunks_returns_empty_for_ready_document_without_chunks(
    test_session: Session,
) -> None:
    """A ready document with no chunks resolves to an empty context list."""
    document = Document(
        title="Empty",
        document_type=DocumentType.PAPER,
        source_filename="Empty.pdf",
        status=DocumentStatus.READY,
    )
    test_session.add(document)
    test_session.commit()

    assert (
        fetch_document_chunks(
            session=test_session,
            document_id=document.document_id,
            limit=5,
        )
        == []
    )


def test_fetch_document_chunks_wraps_db_connection_error_as_upstream_unavailable() -> None:
    """A DB-level OperationalError surfaces as UpstreamUnavailable (maps to 503)."""
    from sqlalchemy.exc import OperationalError

    fake_session = MagicMock()
    fake_session.execute.side_effect = OperationalError(
        statement="SELECT 1",
        params={},
        orig=Exception("connection refused"),
    )
    with pytest.raises(UpstreamUnavailable):
        fetch_document_chunks(session=fake_session, document_id=uuid4(), limit=5)
