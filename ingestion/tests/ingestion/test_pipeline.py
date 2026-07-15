"""Integration tests for the ingestion pipeline."""

from unittest.mock import MagicMock

import pytest
from sqlalchemy.orm import Session

from core.database.schema import Chunk, Document, Embedding
from core.exceptions import IngestionError
from core.types.document import DocumentStatus, DocumentType
from ingestion.pipeline import run_ingestion


@pytest.fixture
def fake_embeddings_client() -> MagicMock:
    """A mocked embeddings client that returns fixed 1536-dim vectors."""
    client = MagicMock()

    def _embed(texts: list[str]) -> list[list[float]]:
        return [[0.01 * (i + 1)] * 1536 for i in range(len(texts))]

    client.embed.side_effect = _embed
    return client


def test_pipeline_happy_path_reaches_ready(
    test_session: Session,
    fake_embeddings_client: MagicMock,
    sample_paper_pdf: bytes,
) -> None:
    """A paper ingestion advances validating -> chunking -> embedding -> ready."""
    document = Document(
        title="Sample Paper",
        document_type=DocumentType.PAPER,
        source_filename="sample.pdf",
    )
    test_session.add(document)
    test_session.flush()

    run_ingestion(
        document_id=document.document_id,
        title="Sample Paper",
        document_type="paper",
        source_filename="sample.pdf",
        file_bytes=sample_paper_pdf,
        session=test_session,
        embeddings_client=fake_embeddings_client,
        model_name="text-embedding-3-small",
    )

    test_session.commit()
    refreshed = test_session.get(Document, document.document_id)
    assert refreshed is not None
    assert refreshed.status == DocumentStatus.READY
    assert refreshed.error_message is None

    chunks = (
        test_session.query(Chunk)
        .filter(Chunk.document_id == document.document_id)
        .order_by(Chunk.position)
        .all()
    )
    assert len(chunks) >= 1
    positions = [chunk.position for chunk in chunks]
    assert positions == sorted(positions)
    assert positions == list(range(len(chunks)))

    embeddings = test_session.query(Embedding).all()
    assert len(embeddings) == len(chunks)
    for embedding in embeddings:
        assert len(embedding.embedding) == 1536


def test_pipeline_failure_marks_failed_with_error(
    test_session: Session,
    sample_paper_pdf: bytes,
) -> None:
    """A mocked embeddings client that raises surfaces as a failure."""
    document = Document(
        title="Failing Paper",
        document_type=DocumentType.PAPER,
        source_filename="fail.pdf",
    )
    test_session.add(document)
    test_session.flush()

    failing_client = MagicMock()
    failing_client.embed.side_effect = RuntimeError("upstream down")

    with pytest.raises(IngestionError):
        run_ingestion(
            document_id=document.document_id,
            title="Failing Paper",
            document_type="paper",
            source_filename="fail.pdf",
            file_bytes=sample_paper_pdf,
            session=test_session,
            embeddings_client=failing_client,
            model_name="text-embedding-3-small",
        )

    test_session.rollback()
    # The caller (task layer) would persist failed status; verify the rollback
    # removed partial chunks so a retry starts clean.
    chunks = test_session.query(Chunk).filter(Chunk.document_id == document.document_id).all()
    assert chunks == []


def test_pipeline_rejects_unsupported_document_type(
    test_session: Session,
    fake_embeddings_client: MagicMock,
) -> None:
    """A document type without a chunker raises IngestionError."""
    document = Document(
        title="Unknown",
        document_type=DocumentType.BOOK,
        source_filename="book.epub",
    )
    test_session.add(document)
    test_session.flush()

    with pytest.raises(IngestionError):
        run_ingestion(
            document_id=document.document_id,
            title="Unknown",
            document_type="book",
            source_filename="book.epub",
            file_bytes=b"contents",
            session=test_session,
            embeddings_client=fake_embeddings_client,
            model_name="text-embedding-3-small",
        )


def test_reingestion_creates_new_document_id(
    test_session: Session,
    fake_embeddings_client: MagicMock,
    sample_paper_pdf: bytes,
) -> None:
    """Re-uploading creates a new row; old chunks and embeddings coexist."""
    first = Document(
        title="Paper",
        document_type=DocumentType.PAPER,
        source_filename="sample.pdf",
    )
    test_session.add(first)
    test_session.flush()
    run_ingestion(
        document_id=first.document_id,
        title="Paper",
        document_type="paper",
        source_filename="sample.pdf",
        file_bytes=sample_paper_pdf,
        session=test_session,
        embeddings_client=fake_embeddings_client,
        model_name="text-embedding-3-small",
    )
    test_session.commit()

    first_chunk_count = (
        test_session.query(Chunk).filter(Chunk.document_id == first.document_id).count()
    )

    second = Document(
        title="Paper",
        document_type=DocumentType.PAPER,
        source_filename="sample.pdf",
    )
    test_session.add(second)
    test_session.flush()
    run_ingestion(
        document_id=second.document_id,
        title="Paper",
        document_type="paper",
        source_filename="sample.pdf",
        file_bytes=sample_paper_pdf,
        session=test_session,
        embeddings_client=fake_embeddings_client,
        model_name="text-embedding-3-small",
    )
    test_session.commit()

    assert second.document_id != first.document_id
    assert (
        test_session.query(Chunk).filter(Chunk.document_id == first.document_id).count()
        == first_chunk_count
    )
    assert (
        test_session.query(Chunk).filter(Chunk.document_id == second.document_id).count()
        == first_chunk_count
    )
