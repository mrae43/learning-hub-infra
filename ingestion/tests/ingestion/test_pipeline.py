"""Integration tests for the ingestion pipeline."""

from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy.orm import Session

from core.database.schema import Chunk, Document, Embedding
from core.exceptions import IngestionError
from core.types.document import DocumentStatus, DocumentType
from ingestion.models import PendingIngestion
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
        pending=PendingIngestion(
            document_id=document.document_id,
            title="Sample Paper",
            document_type=DocumentType.PAPER,
            source_filename="sample.pdf",
            file_bytes=sample_paper_pdf,
        ),
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

    # Parents are NOT embedded; only children are.
    parents = [c for c in chunks if c.parent_chunk_id is None]
    children = [c for c in chunks if c.parent_chunk_id is not None]
    assert len(parents) >= 1
    assert len(children) >= 1
    # Parents enumerate at document level; children enumerate within their parent.
    for p in parents:
        assert p.position < len(parents)
    for child in children:
        assert child.parent_chunk_id is not None
        assert "child_of" in child.type_metadata

    embeddings = test_session.query(Embedding).all()
    assert len(embeddings) == len(children)
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
            pending=PendingIngestion(
                document_id=document.document_id,
                title="Failing Paper",
                document_type=DocumentType.PAPER,
                source_filename="fail.pdf",
                file_bytes=sample_paper_pdf,
            ),
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
        pending=PendingIngestion(
            document_id=document.document_id,
            title="Sample Book",
            document_type=DocumentType.BOOK,
            source_filename="sample.pdf",
            file_bytes=sample_book_pdf,
        ),
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
    # Parents have original metadata; children inherit with "child_of" key.
    for chunk in chunks:
        assert "chapter" in chunk.type_metadata
        assert isinstance(chunk.type_metadata["chapter"], int)

    children = [c for c in chunks if c.parent_chunk_id is not None]
    assert len(children) >= 1
    for child in children:
        assert "child_of" in child.type_metadata

    embeddings = test_session.query(Embedding).all()
    assert len(embeddings) == len(children)


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
        pending=PendingIngestion(
            document_id=document.document_id,
            title="Sample Docs",
            document_type=DocumentType.DOCUMENTATION,
            source_filename="docs.md",
            file_bytes=sample_documentation_md,
        ),
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

    children = [c for c in chunks if c.parent_chunk_id is not None]
    assert len(children) >= 1
    for child in children:
        assert "child_of" in child.type_metadata


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
        pending=PendingIngestion(
            document_id=first.document_id,
            title="Paper",
            document_type=DocumentType.PAPER,
            source_filename="sample.pdf",
            file_bytes=sample_paper_pdf,
        ),
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
        pending=PendingIngestion(
            document_id=second.document_id,
            title="Paper",
            document_type=DocumentType.PAPER,
            source_filename="sample.pdf",
            file_bytes=sample_paper_pdf,
        ),
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
                    pending=PendingIngestion(
                        document_id=document.document_id,
                        title="Bad Book",
                        document_type=DocumentType.BOOK,
                        source_filename="bad.pdf",
                        file_bytes=b"irrelevant",
                    ),
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
                    pending=PendingIngestion(
                        document_id=document.document_id,
                        title="Bad Paper",
                        document_type=DocumentType.PAPER,
                        source_filename="bad.pdf",
                        file_bytes=b"irrelevant",
                    ),
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
            pending=PendingIngestion(
                document_id=document.document_id,
                title="Sample Book",
                document_type=DocumentType.BOOK,
                source_filename="sample.pdf",
                file_bytes=sample_book_pdf,
            ),
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


# ── Parent-child ingestion tests ─────────────────────────────────────


class TestParentChildIngestion:
    """Tests for parent-child chunking in the ingestion pipeline."""

    def test_parents_not_embedded_children_are(
        self,
        test_session: Session,
        fake_embeddings_client: MagicMock,
        sample_paper_pdf: bytes,
    ) -> None:
        """Parent rows are stored without embeddings; children are embedded."""
        document = Document(
            title="PC Paper",
            document_type=DocumentType.PAPER,
            source_filename="pc.pdf",
        )
        test_session.add(document)
        test_session.flush()

        run_ingestion(
            pending=PendingIngestion(
                document_id=document.document_id,
                title="PC Paper",
                document_type=DocumentType.PAPER,
                source_filename="pc.pdf",
                file_bytes=sample_paper_pdf,
            ),
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
        parents = [c for c in chunks if c.parent_chunk_id is None]
        children = [c for c in chunks if c.parent_chunk_id is not None]
        assert len(parents) >= 1
        assert len(children) >= 1

        # Parents should not have embeddings
        if parents:
            parent_ids = [p.chunk_id for p in parents]
            parent_emb_count = (
                test_session.query(Embedding).filter(Embedding.chunk_id.in_(parent_ids)).count()
            )
            assert parent_emb_count == 0

        # All children should have embeddings
        if children:
            child_ids = [c.chunk_id for c in children]
            child_emb_count = (
                test_session.query(Embedding).filter(Embedding.chunk_id.in_(child_ids)).count()
            )
            assert child_emb_count == len(children)

    def test_large_parent_creates_multiple_children(
        self,
        test_session: Session,
        fake_embeddings_client: MagicMock,
    ) -> None:
        """A parent >512 tokens produces multiple child chunks."""
        big_content = "large_parent_child_test_word " * 600
        document = Document(
            title="Big Parent",
            document_type=DocumentType.PAPER,
            source_filename="big.pdf",
        )
        test_session.add(document)
        test_session.flush()

        with patch("ingestion.pipeline._chunk_document") as mock_chunk:
            mock_chunk.return_value = [
                (big_content, {"section": "Test", "subsection": None, "page": 1}, 600),
            ]
            run_ingestion(
                pending=PendingIngestion(
                    document_id=document.document_id,
                    title="Big Parent",
                    document_type=DocumentType.PAPER,
                    source_filename="big.pdf",
                    file_bytes=b"fake",
                ),
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
        parents = [c for c in chunks if c.parent_chunk_id is None]
        children = [c for c in chunks if c.parent_chunk_id is not None]

        assert len(parents) == 1
        assert len(children) >= 2
        for child in children:
            assert child.parent_chunk_id == parents[0].chunk_id
            assert child.type_metadata.get("child_of") == str(parents[0].chunk_id)

    def test_small_parent_produces_one_child(
        self,
        test_session: Session,
        fake_embeddings_client: MagicMock,
    ) -> None:
        """A parent ≤512 tokens produces exactly one child (identity split)."""
        small_content = "small chunk content"
        document = Document(
            title="Small Parent",
            document_type=DocumentType.PAPER,
            source_filename="small.pdf",
        )
        test_session.add(document)
        test_session.flush()

        with patch("ingestion.pipeline._chunk_document") as mock_chunk:
            mock_chunk.return_value = [
                (small_content, {"section": "Intro", "subsection": None, "page": 1}, 3),
            ]
            run_ingestion(
                pending=PendingIngestion(
                    document_id=document.document_id,
                    title="Small Parent",
                    document_type=DocumentType.PAPER,
                    source_filename="small.pdf",
                    file_bytes=b"fake",
                ),
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
        parents = [c for c in chunks if c.parent_chunk_id is None]
        children = [c for c in chunks if c.parent_chunk_id is not None]

        assert len(parents) == 1
        assert len(children) == 1
        assert children[0].content == small_content

    def test_child_inherits_type_metadata_with_child_of(
        self,
        test_session: Session,
        fake_embeddings_client: MagicMock,
    ) -> None:
        """Children inherit type_metadata from parent and add 'child_of' key."""
        content = "inherit_metadata_test_word " * 400
        document = Document(
            title="Inherit Test",
            document_type=DocumentType.BOOK,
            source_filename="inherit.pdf",
        )
        test_session.add(document)
        test_session.flush()

        with patch("ingestion.pipeline._chunk_document") as mock_chunk:
            mock_chunk.return_value = [
                (content, {"chapter": 5, "heading": "Deep Dive"}, 400),
            ]
            run_ingestion(
                pending=PendingIngestion(
                    document_id=document.document_id,
                    title="Inherit Test",
                    document_type=DocumentType.BOOK,
                    source_filename="inherit.pdf",
                    file_bytes=b"fake",
                ),
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
        children = [c for c in chunks if c.parent_chunk_id is not None]
        assert len(children) >= 1
        for child in children:
            assert child.type_metadata.get("chapter") == 5
            assert child.type_metadata.get("heading") == "Deep Dive"
            assert "child_of" in child.type_metadata
            child_of_val = child.type_metadata["child_of"]
            assert isinstance(child_of_val, str)
            assert len(child_of_val) > 0

    def test_positions_enumerate_within_parent(
        self,
        test_session: Session,
        fake_embeddings_client: MagicMock,
    ) -> None:
        """Children enumerate position within their parent; parents enumerate at document level."""
        content_a = "position_test_word_a " * 400
        content_b = "position_test_word_b " * 400
        document = Document(
            title="Position Test",
            document_type=DocumentType.PAPER,
            source_filename="pos.pdf",
        )
        test_session.add(document)
        test_session.flush()

        with patch("ingestion.pipeline._chunk_document") as mock_chunk:
            mock_chunk.return_value = [
                (content_a, {"section": "A", "subsection": None, "page": 1}, 400),
                (content_b, {"section": "B", "subsection": None, "page": 2}, 400),
            ]
            run_ingestion(
                pending=PendingIngestion(
                    document_id=document.document_id,
                    title="Position Test",
                    document_type=DocumentType.PAPER,
                    source_filename="pos.pdf",
                    file_bytes=b"fake",
                ),
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
        parents = [c for c in chunks if c.parent_chunk_id is None]
        # Parents enumerate at document level
        for i, p in enumerate(parents):
            assert p.position == i, f"Parent at index {i} has position {p.position}"

        # Children enumerate within their parent
        for parent in parents:
            parent_children = [c for c in chunks if c.parent_chunk_id == parent.chunk_id]
            for i, child in enumerate(parent_children):
                assert child.position == i, (
                    f"Child at index {i} of parent {parent.chunk_id} has position {child.position}"
                )
