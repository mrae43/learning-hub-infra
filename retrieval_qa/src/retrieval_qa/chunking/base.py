"""Base document-type chunker abstraction and registry."""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import ClassVar

from pydantic import BaseModel

from core.exceptions import IngestionError
from core.types.chunk import Chunk
from core.types.document import DocumentType


class DocumentChunker(ABC):
    """Abstract base for a document-type chunker."""

    metadata_model: ClassVar[type[BaseModel]]

    @abstractmethod
    def chunk(self, file_bytes: bytes) -> Sequence[Chunk]:
        """Chunk ``file_bytes`` into a sequence of ``Chunk`` objects.

        Args:
            file_bytes: Raw uploaded file contents.

        Returns:
            Chunks ordered by their appearance in the document.
        """
        ...


chunker_registry: dict[DocumentType, type[DocumentChunker]] = {}


def register_chunker(
    document_type: DocumentType,
    chunker_class: type[DocumentChunker],
) -> type[DocumentChunker]:
    """Register a chunker class for a document type.

    Returns:
        The registered chunker class (so it can be used as a decorator).
    """
    chunker_registry[document_type] = chunker_class
    return chunker_class


def get_chunker_class(document_type: DocumentType) -> type[DocumentChunker]:
    """Return the chunker class registered for ``document_type``.

    Raises:
        IngestionError: No chunker is registered for ``document_type``.
    """
    try:
        return chunker_registry[document_type]
    except KeyError as exc:
        raise IngestionError(f"No chunker registered for {document_type.value}") from exc


def get_chunker(document_type: DocumentType) -> DocumentChunker:
    """Return a chunker instance for ``document_type``.

    Raises:
        IngestionError: No chunker is registered for ``document_type``.
    """
    return get_chunker_class(document_type)()


__all__ = [
    "DocumentChunker",
    "chunker_registry",
    "get_chunker",
    "get_chunker_class",
    "register_chunker",
]
