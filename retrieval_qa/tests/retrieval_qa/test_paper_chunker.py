"""Tests for the paper chunker."""

from collections.abc import Callable
from pathlib import Path

import pytest

from retrieval_qa.chunking.paper_chunker import chunk_paper


def test_paper_chunker_emits_sections(
    sample_paper_pdf: bytes,
    temp_file: Callable[..., Path],
) -> None:
    """chunk_paper extracts text and splits it into ordered chunks."""
    pdf_path = temp_file(sample_paper_pdf, "sample.pdf")
    chunks = chunk_paper(pdf_path)

    assert len(chunks) >= 1
    for chunk in chunks:
        assert chunk.content.strip()
        assert chunk.token_count > 0
        assert chunk.metadata.section
        assert chunk.metadata.page >= 1
        assert chunk.metadata.subsection is None or isinstance(chunk.metadata.subsection, str)


def test_paper_chunker_includes_section_metadata(
    sample_paper_pdf: bytes,
    temp_file: Callable[..., Path],
) -> None:
    """Extracted chunks carry paper-specific section/subsection metadata."""
    pdf_path = temp_file(sample_paper_pdf, "sample.pdf")
    chunks = chunk_paper(pdf_path)

    sections = [chunk.metadata.section for chunk in chunks]
    # The first section header detected should appear in the extracted chunks.
    assert any("Introduction" in section or "Methods" in section for section in sections)


def test_paper_chunker_rejects_invalid_pdf(temp_file: Callable[..., Path]) -> None:
    """A non-PDF file raises IngestionError."""
    from core.exceptions import IngestionError

    pdf_path = temp_file(b"not a pdf", "invalid.pdf")
    with pytest.raises(IngestionError):
        chunk_paper(pdf_path)
