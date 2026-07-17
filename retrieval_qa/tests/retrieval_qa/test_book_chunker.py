"""Tests for the book chunker."""

import pytest

from core.exceptions import IngestionError
from core.types.chunk import BookChunkMetadata
from retrieval_qa.chunking.book_chunker import chunk_book


def test_book_chunker_emits_chapters(sample_book_pdf: bytes) -> None:
    """chunk_book extracts text and splits it into ordered chunks."""
    chunks = chunk_book(sample_book_pdf)

    assert len(chunks) >= 1
    for chunk in chunks:
        assert chunk.content.strip()
        assert chunk.token_count > 0
        assert isinstance(chunk.metadata.chapter, int)
        assert chunk.metadata.chapter >= 1
        assert chunk.metadata.heading is None or isinstance(chunk.metadata.heading, str)


def test_book_chunker_detects_chapter_boundaries(sample_book_pdf: bytes) -> None:
    """Detected chapter boundaries are reflected in chunk metadata."""
    chunks = chunk_book(sample_book_pdf)

    chapters = [chunk.metadata.chapter for chunk in chunks]
    assert 1 in chapters
    assert 2 in chapters


def test_book_chunker_epub(sample_book_epub: bytes) -> None:
    """chunk_book can parse an EPUB with chapter structure."""
    chunks = chunk_book(sample_book_epub)

    assert len(chunks) >= 1
    chapters = {chunk.metadata.chapter for chunk in chunks}
    assert 1 in chapters
    assert all(isinstance(chunk.metadata.chapter, int) for chunk in chunks)


def test_book_chunker_rejects_invalid_bytes() -> None:
    """A non-PDF/non-EPUB byte stream raises IngestionError."""
    with pytest.raises(IngestionError):
        chunk_book(b"not a book")


def test_book_chunk_metadata_validates_at_boundary(sample_book_pdf: bytes) -> None:
    """Chunks emitted by the book chunker validate against BookChunkMetadata."""
    chunks = chunk_book(sample_book_pdf)
    for chunk in chunks:
        # model_validate should succeed and extra='forbid' means unknown keys
        # would have already raised during construction.
        assert BookChunkMetadata.model_validate(chunk.metadata.model_dump())
