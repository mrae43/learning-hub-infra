"""Tests for the document-type chunker registry."""

from collections.abc import Sequence

import pytest
from pydantic import BaseModel

from core.exceptions import IngestionError
from core.types.chunk import (
    BookChunkMetadata,
    Chunk,
    DocumentationChunkMetadata,
    PaperChunkMetadata,
)
from core.types.document import DocumentType
from retrieval_qa.chunking import (
    BookChunker,
    DocumentationChunker,
    DocumentChunker,
    PaperChunker,
    chunker_registry,
    get_chunker,
    get_chunker_class,
    register_chunker,
)
from retrieval_qa.chunking.book_chunker import BookChunk
from retrieval_qa.chunking.documentation_chunker import DocumentationChunk
from retrieval_qa.chunking.paper_chunker import PaperChunk


def test_registry_contains_all_document_types() -> None:
    """Every known DocumentType has a registered chunker."""
    assert set(chunker_registry.keys()) == set(DocumentType)


def test_get_chunker_class_returns_registered_class() -> None:
    """get_chunker_class returns the registered chunker class for each type."""
    assert get_chunker_class(DocumentType.PAPER) is PaperChunker
    assert get_chunker_class(DocumentType.BOOK) is BookChunker
    assert get_chunker_class(DocumentType.DOCUMENTATION) is DocumentationChunker


def test_get_chunker_returns_instance_for_each_type() -> None:
    """get_chunker returns a usable chunker instance for each document type."""
    assert isinstance(get_chunker(DocumentType.PAPER), PaperChunker)
    assert isinstance(get_chunker(DocumentType.BOOK), BookChunker)
    assert isinstance(get_chunker(DocumentType.DOCUMENTATION), DocumentationChunker)


def test_chunker_classes_implement_abstract_base() -> None:
    """Each concrete chunker is a subclass of DocumentChunker."""
    assert issubclass(PaperChunker, DocumentChunker)
    assert issubclass(BookChunker, DocumentChunker)
    assert issubclass(DocumentationChunker, DocumentChunker)


def test_chunker_classes_declare_metadata_model() -> None:
    """Each concrete chunker declares a Pydantic metadata model."""
    assert issubclass(PaperChunker.metadata_model, BaseModel)
    assert issubclass(BookChunker.metadata_model, BaseModel)
    assert issubclass(DocumentationChunker.metadata_model, BaseModel)


def test_chunker_returns_chunk_base_model_objects(sample_paper_pdf: bytes) -> None:
    """Chunker instances return objects that inherit from the Chunk base model."""
    chunker = get_chunker(DocumentType.PAPER)
    result = chunker.chunk(sample_paper_pdf)
    assert isinstance(result, Sequence)
    assert len(result) >= 1
    for chunk in result:
        assert isinstance(chunk, Chunk)


def test_get_chunker_class_raises_for_unknown_document_type() -> None:
    """An unregistered document type raises IngestionError."""
    # Remove the paper chunker temporarily to simulate an unknown type.
    original = chunker_registry.pop(DocumentType.PAPER)
    try:
        with pytest.raises(IngestionError):
            get_chunker_class(DocumentType.PAPER)
    finally:
        chunker_registry[DocumentType.PAPER] = original


def test_register_chunker_allows_extension_without_editing_existing() -> None:
    """A new chunker can be registered without touching existing chunkers."""

    class FakeMetadata(BaseModel):
        value: int

    class FakeChunker(DocumentChunker):
        metadata_model = FakeMetadata

        def chunk(self, file_bytes: bytes) -> Sequence[Chunk]:
            return []

    # Temporarily swap the paper chunker to simulate adding a new document type.
    custom_type = DocumentType.PAPER
    original = chunker_registry[custom_type]
    try:
        register_chunker(custom_type, FakeChunker)
        assert chunker_registry[custom_type] is FakeChunker
        assert isinstance(get_chunker(custom_type), FakeChunker)
    finally:
        chunker_registry[custom_type] = original


def test_concrete_chunk_types_are_chunk_subclasses() -> None:
    """PaperChunk, BookChunk, and DocumentationChunk inherit from Chunk."""
    assert issubclass(PaperChunk, Chunk)
    assert issubclass(BookChunk, Chunk)
    assert issubclass(DocumentationChunk, Chunk)

    paper = PaperChunk(
        content="c",
        metadata=PaperChunkMetadata(section="Intro", subsection=None, page=1),
        token_count=1,
    )
    assert isinstance(paper, Chunk)

    book = BookChunk(
        content="c",
        metadata=BookChunkMetadata(chapter=1, heading=None),
        token_count=1,
    )
    assert isinstance(book, Chunk)

    doc = DocumentationChunk(
        content="c",
        metadata=DocumentationChunkMetadata(page="p", section=None),
        token_count=1,
    )
    assert isinstance(doc, Chunk)
