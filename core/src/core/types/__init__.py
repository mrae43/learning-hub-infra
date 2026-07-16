"""Shared Pydantic boundary types."""

from core.types.chat import ChatMessage
from core.types.chunk import (
    BookChunkMetadata,
    DocumentationChunkMetadata,
    PaperChunkMetadata,
)
from core.types.document import (
    DocumentStatus,
    DocumentStatusResponse,
    DocumentType,
)

__all__ = [
    "BookChunkMetadata",
    "ChatMessage",
    "DocumentStatus",
    "DocumentStatusResponse",
    "DocumentType",
    "DocumentationChunkMetadata",
    "PaperChunkMetadata",
]
