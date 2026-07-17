"""Tests for chunk metadata Pydantic models."""

import pytest
from pydantic import ValidationError

from core.types.chunk import (
    BookChunkMetadata,
    DocumentationChunkMetadata,
    PaperChunkMetadata,
)


def test_paper_chunk_metadata_valid() -> None:
    """PaperChunkMetadata accepts valid section metadata."""
    meta = PaperChunkMetadata(section="Introduction", subsection=None, page=1)
    assert meta.section == "Introduction"
    assert meta.subsection is None
    assert meta.page == 1


def test_paper_chunk_metadata_rejects_extra_fields() -> None:
    """PaperChunkMetadata with extra='forbid' rejects unknown fields."""
    with pytest.raises(ValidationError):
        PaperChunkMetadata.model_validate(
            {"section": "Intro", "subsection": None, "page": 1, "extra_field": "nope"}
        )


def test_book_chunk_metadata_valid() -> None:
    """BookChunkMetadata accepts valid chapter metadata."""
    meta = BookChunkMetadata(chapter=3, heading="Methods")
    assert meta.chapter == 3
    assert meta.heading == "Methods"


def test_book_chunk_metadata_rejects_extra_fields() -> None:
    """BookChunkMetadata with extra='forbid' rejects unknown fields."""
    with pytest.raises(ValidationError):
        BookChunkMetadata.model_validate({"chapter": 1, "heading": None, "sneaky": True})


def test_book_chunk_metadata_rejects_string_chapter() -> None:
    """BookChunkMetadata rejects a string chapter value at the write boundary."""
    with pytest.raises(ValidationError):
        BookChunkMetadata.model_validate({"chapter": "three", "heading": None})


def test_documentation_chunk_metadata_valid() -> None:
    """DocumentationChunkMetadata accepts valid page/section metadata."""
    meta = DocumentationChunkMetadata(page="api-reference", section="auth")
    assert meta.page == "api-reference"
    assert meta.section == "auth"


def test_documentation_chunk_metadata_rejects_extra_fields() -> None:
    """DocumentationChunkMetadata with extra='forbid' rejects unknown fields."""
    with pytest.raises(ValidationError):
        DocumentationChunkMetadata.model_validate({"page": "p1", "section": None, "bogus": 42})
