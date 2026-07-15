"""Tests for the paper chunker."""

import pytest

from retrieval_qa.chunking.paper_chunker import chunk_paper


def test_paper_chunker_emits_sections(sample_paper_pdf: bytes) -> None:
    """chunk_paper extracts text and splits it into ordered chunks."""
    chunks = chunk_paper(sample_paper_pdf)

    assert len(chunks) >= 1
    for chunk in chunks:
        assert chunk.content.strip()
        assert chunk.token_count > 0
        assert chunk.metadata.section
        assert chunk.metadata.page >= 1
        assert chunk.metadata.subsection is None or isinstance(chunk.metadata.subsection, str)


def test_paper_chunker_includes_section_metadata(sample_paper_pdf: bytes) -> None:
    """Extracted chunks carry paper-specific section/subsection metadata."""
    chunks = chunk_paper(sample_paper_pdf)

    sections = [chunk.metadata.section for chunk in chunks]
    # The first section header detected should appear in the extracted chunks.
    assert any("Introduction" in section or "Methods" in section for section in sections)


def test_paper_chunker_rejects_invalid_pdf() -> None:
    """A non-PDF byte stream raises IngestionError."""
    from core.exceptions import IngestionError

    with pytest.raises(IngestionError):
        chunk_paper(b"not a pdf")
