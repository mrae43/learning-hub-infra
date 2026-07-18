"""Document-type chunkers.

Provides a registry-based extension point for new document types. Adding a
chunker for a new ``DocumentType`` only requires implementing
``DocumentChunker`` and calling ``register_chunker``; no existing chunker or
pipeline code needs to change.
"""

from core.types.chunk import Chunk
from retrieval_qa.chunking.base import (
    DocumentChunker,
    chunker_registry,
    get_chunker,
    get_chunker_class,
    register_chunker,
)
from retrieval_qa.chunking.book_chunker import BookChunk, BookChunker, chunk_book
from retrieval_qa.chunking.documentation_chunker import (
    DocumentationChunk,
    DocumentationChunker,
    chunk_documentation,
)
from retrieval_qa.chunking.paper_chunker import PaperChunk, PaperChunker, chunk_paper

__all__ = [
    "BookChunk",
    "BookChunker",
    "Chunk",
    "DocumentChunker",
    "DocumentationChunk",
    "DocumentationChunker",
    "PaperChunk",
    "PaperChunker",
    "chunk_book",
    "chunk_documentation",
    "chunk_paper",
    "chunker_registry",
    "get_chunker",
    "get_chunker_class",
    "register_chunker",
]
