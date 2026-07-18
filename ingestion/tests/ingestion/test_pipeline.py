"""Integration tests for the ingestion pipeline."""

from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy.orm import Session

from core.database.schema import Chunk, Document, Embedding
from core.exceptions import IngestionError
from core.types.document import DocumentStatus, DocumentType
from ingestion.pipeline import _validate_type_metadata, run_ingestion


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
        document_type=DocumentType.PAPER,
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
            document_type=DocumentType.PAPER,
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


def test_pipeline_book_happy_path_reaches_ready(
    test_session: Session,
    fake_embeddings_client: MagicMock,
    sample_book_pdf: bytes,
) -> None:
    """A book ingestion advances validating -> chunking -> embedding -> ready."""
    document = Document(
        title="Sample Book",
        document_type=DocumentType.BOOK,
        source_filename="sample.pdf",
    )
    test_session.add(document)
    test_session.flush()

    run_ingestion(
        document_id=document.document_id,
        title="Sample Book",
        document_type=DocumentType.BOOK,
        source_filename="sample.pdf",
        file_bytes=sample_book_pdf,
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
    for chunk in chunks:
        assert "chapter" in chunk.type_metadata
        assert isinstance(chunk.type_metadata["chapter"], int)

    embeddings = test_session.query(Embedding).all()
    assert len(embeddings) == len(chunks)


def test_pipeline_documentation_happy_path_reaches_ready(
    test_session: Session,
    fake_embeddings_client: MagicMock,
    sample_documentation_md: bytes,
) -> None:
    """A documentation ingestion advances validating -> chunking -> embedding -> ready."""
    document = Document(
        title="Sample Docs",
        document_type=DocumentType.DOCUMENTATION,
        source_filename="docs.md",
    )
    test_session.add(document)
    test_session.flush()

    run_ingestion(
        document_id=document.document_id,
        title="Sample Docs",
        document_type=DocumentType.DOCUMENTATION,
        source_filename="docs.md",
        file_bytes=sample_documentation_md,
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
    for chunk in chunks:
        assert "page" in chunk.type_metadata
        assert isinstance(chunk.type_metadata["page"], str)


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
        document_type=DocumentType.PAPER,
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
        document_type=DocumentType.PAPER,
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


# ── _validate_type_metadata unit tests ─────────────────────────────────────


class TestValidateTypeMetadata:
    """Unit tests for ``_validate_type_metadata()``."""

    def test_passes_valid_book_metadata(self) -> None:
        """Valid BookChunkMetadata passes validation."""
        _validate_type_metadata(DocumentType.BOOK, {"chapter": 1, "heading": "Intro"})

    def test_rejects_string_chapter_in_book(self) -> None:
        """A string chapter value raises IngestionError."""
        with pytest.raises(IngestionError):
            _validate_type_metadata(DocumentType.BOOK, {"chapter": "three", "heading": "Methods"})

    def test_rejects_extra_fields(self) -> None:
        """An unknown key in type_metadata raises IngestionError."""
        with pytest.raises(IngestionError):
            _validate_type_metadata(
                DocumentType.BOOK,
                {"chapter": 1, "heading": "Intro", "sneaky": "bad"},
            )

    def test_passes_valid_paper_metadata(self) -> None:
        """Valid PaperChunkMetadata passes validation."""
        _validate_type_metadata(
            DocumentType.PAPER,
            {"section": "Methods", "subsection": None, "page": 3},
        )

    def test_rejects_invalid_paper_metadata(self) -> None:
        """Paper metadata with string page raises IngestionError."""
        with pytest.raises(IngestionError):
            _validate_type_metadata(
                DocumentType.PAPER,
                {"section": "Methods", "subsection": None, "page": "three"},
            )

    def test_passes_valid_documentation_metadata(self) -> None:
        """Valid DocumentationChunkMetadata passes validation."""
        _validate_type_metadata(
            DocumentType.DOCUMENTATION,
            {"page": "42", "section": "Setup"},
        )

    def test_rejects_invalid_documentation_metadata(self) -> None:
        """Documentation metadata with missing section raises IngestionError."""
        with pytest.raises(IngestionError):
            _validate_type_metadata(
                DocumentType.DOCUMENTATION,
                {"page": "42"},
            )


# ── Pipeline-level validation integration tests ───────────────────────────


class TestPipelineValidation:
    """Integration tests verifying validation runs before DB writes."""

    def test_pipeline_rejects_string_chapter_in_book(
        self,
        test_session: Session,
        fake_embeddings_client: MagicMock,
    ) -> None:
        """Invalid type_metadata (string chapter) bubbles up as IngestionError.

        This simulates a chunker bug (or a code path that bypasses Pydantic
        construction) and verifies the pipeline catches it before persisting.
        """
        document = Document(
            title="Bad Book",
            document_type=DocumentType.BOOK,
            source_filename="bad.pdf",
        )
        test_session.add(document)
        test_session.flush()

        with patch("ingestion.pipeline._chunk_document") as mock_chunk:
            mock_chunk.return_value = [
                ("content", {"chapter": "three", "heading": "Methods"}, 10),
            ]
            with pytest.raises(IngestionError):
                run_ingestion(
                    document_id=document.document_id,
                    title="Bad Book",
                    document_type=DocumentType.BOOK,
                    source_filename="bad.pdf",
                    file_bytes=b"irrelevant",
                    session=test_session,
                    embeddings_client=fake_embeddings_client,
                    model_name="text-embedding-3-small",
                )

        # No chunks should have been persisted
        chunks = test_session.query(Chunk).filter(Chunk.document_id == document.document_id).all()
        assert chunks == []

    def test_pipeline_rejects_string_page_in_paper(
        self,
        test_session: Session,
        fake_embeddings_client: MagicMock,
    ) -> None:
        """Paper metadata with string page raises IngestionError."""
        document = Document(
            title="Bad Paper",
            document_type=DocumentType.PAPER,
            source_filename="bad.pdf",
        )
        test_session.add(document)
        test_session.flush()

        with patch("ingestion.pipeline._chunk_document") as mock_chunk:
            mock_chunk.return_value = [
                ("content", {"section": "Methods", "subsection": None, "page": "iv"}, 10),
            ]
            with pytest.raises(IngestionError):
                run_ingestion(
                    document_id=document.document_id,
                    title="Bad Paper",
                    document_type=DocumentType.PAPER,
                    source_filename="bad.pdf",
                    file_bytes=b"irrelevant",
                    session=test_session,
                    embeddings_client=fake_embeddings_client,
                    model_name="text-embedding-3-small",
                )

        chunks = test_session.query(Chunk).filter(Chunk.document_id == document.document_id).all()
        assert chunks == []

    def test_happy_path_book_still_succeeds(
        self,
        test_session: Session,
        fake_embeddings_client: MagicMock,
        sample_book_pdf: bytes,
    ) -> None:
        """Existing book happy path is not broken by validation step."""
        document = Document(
            title="Sample Book",
            document_type=DocumentType.BOOK,
            source_filename="sample.pdf",
        )
        test_session.add(document)
        test_session.flush()

        run_ingestion(
            document_id=document.document_id,
            title="Sample Book",
            document_type=DocumentType.BOOK,
            source_filename="sample.pdf",
            file_bytes=sample_book_pdf,
            session=test_session,
            embeddings_client=fake_embeddings_client,
            model_name="text-embedding-3-small",
        )

        test_session.commit()
        chunks = (
            test_session.query(Chunk)
            .filter(Chunk.document_id == document.document_id)
            .order_by(Chunk.position)
            .all()
        )
        assert len(chunks) >= 1
        for chunk in chunks:
            assert "chapter" in chunk.type_metadata
            assert isinstance(chunk.type_metadata["chapter"], int)
