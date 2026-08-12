"""Database-backed tests for the parent-chunk fetch primitive."""

from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from sqlalchemy import func
from sqlalchemy.orm import Session

from core.database.schema import Chunk, Document, Embedding
from core.exceptions import UpstreamUnavailable
from core.retrieval.parents import fetch_parent_chunk
from core.types.document import DocumentStatus, DocumentType
from core.types.responses import CitedPassage


def _seed_parent_child_paper(
    session: Session,
    *,
    title: str = "Parent-Child Paper",
    status: DocumentStatus = DocumentStatus.READY,
    parent_content: str,
    child_contents: list[str],
) -> tuple[Document, Chunk, list[Chunk]]:
    """Seed a document with one parent and several children.

    Children get embeddings and a sparse index; the parent stays unembedded
    (ADR-0016). Returns (document, parent_row, child_rows).
    """
    document = Document(
        title=title,
        document_type=DocumentType.PAPER,
        source_filename=f"{title}.pdf",
        status=status,
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
    for position, content in enumerate(child_contents):
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
                embedding=[0.5] * 1536,
            )
        )
        children.append(child)

    return document, parent, children


def test_fetch_parent_chunk_returns_parent_of_child_anchor(test_session: Session) -> None:
    """Anchoring on a child chunk resolves to its enclosing parent passage.

    Depth Dive anchors text/code passages via ``chunk_id`` (ADR-0014); the
    fetch must widen a child anchor to the parent's content so the dive gets
    the enclosing section, not the fixed-size child window.
    """
    parent_text = "The enclosing section explains vector databases in full."
    _, parent, children = _seed_parent_child_paper(
        test_session,
        parent_content=parent_text,
        child_contents=["child window one", "child window two"],
    )
    test_session.commit()

    passage = fetch_parent_chunk(session=test_session, chunk_id=children[0].chunk_id)

    assert passage == CitedPassage(chunk_id=parent.chunk_id, text=parent_text)


def test_fetch_parent_chunk_returns_parent_anchor_itself(test_session: Session) -> None:
    """Anchoring on a parent chunk (Harness A cite ids are post-parent-swap)."""
    parent_text = "The section itself, cited directly."
    _, parent, _ = _seed_parent_child_paper(
        test_session,
        parent_content=parent_text,
        child_contents=["child window"],
    )
    test_session.commit()

    passage = fetch_parent_chunk(session=test_session, chunk_id=parent.chunk_id)

    assert passage == CitedPassage(chunk_id=parent.chunk_id, text=parent_text)


def test_fetch_parent_chunk_returns_standalone_chunk_itself(test_session: Session) -> None:
    """A chunk with no parent (parent_chunk_id IS NULL) resolves to itself."""
    content = "standalone chunk without a parent"
    document = Document(
        title="Standalone",
        document_type=DocumentType.PAPER,
        source_filename="Standalone.pdf",
        status=DocumentStatus.READY,
    )
    test_session.add(document)
    test_session.flush()
    chunk = Chunk(
        document_id=document.document_id,
        position=0,
        content=content,
        token_count=max(1, len(content.split())),
    )
    test_session.add(chunk)
    test_session.commit()

    passage = fetch_parent_chunk(session=test_session, chunk_id=chunk.chunk_id)

    assert passage == CitedPassage(chunk_id=chunk.chunk_id, text=content)


def test_fetch_parent_chunk_returns_none_for_unknown_chunk_id(test_session: Session) -> None:
    """An anchor that matches no chunk is unresolvable, not an error."""
    passage = fetch_parent_chunk(session=test_session, chunk_id=uuid4())
    assert passage is None


def test_fetch_parent_chunk_returns_none_when_document_not_ready(
    test_session: Session,
) -> None:
    """Chunks of non-ready documents are invisible to the fetch (ADR-0014 gate)."""
    _, parent, children = _seed_parent_child_paper(
        test_session,
        title="Still embedding",
        status=DocumentStatus.EMBEDDING,
        parent_content="parent of a not-yet-ready document",
        child_contents=["child window"],
    )
    test_session.commit()

    assert fetch_parent_chunk(session=test_session, chunk_id=children[0].chunk_id) is None
    assert fetch_parent_chunk(session=test_session, chunk_id=parent.chunk_id) is None


def test_fetch_parent_chunk_wraps_db_connection_error_as_upstream_unavailable() -> None:
    """A DB-level OperationalError surfaces as UpstreamUnavailable (maps to 503)."""
    from sqlalchemy.exc import OperationalError

    fake_session = MagicMock()
    fake_session.execute.side_effect = OperationalError(
        statement="SELECT 1",
        params={},
        orig=Exception("connection refused"),
    )
    with pytest.raises(UpstreamUnavailable):
        fetch_parent_chunk(session=fake_session, chunk_id=uuid4())
