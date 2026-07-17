"""Background ingestion pipeline for uploaded documents."""

from collections.abc import Sequence
from typing import assert_never
from uuid import UUID

from pydantic import ValidationError
from sqlalchemy.orm import Session

from core.clients.embeddings_client import EmbeddingsClient
from core.database.schema import Chunk, Document, Embedding
from core.exceptions import IngestionError
from core.types.chunk import (
    BookChunkMetadata,
    DocumentationChunkMetadata,
    PaperChunkMetadata,
)
from core.types.document import DocumentStatus, DocumentType
from retrieval_qa.chunking.book_chunker import BookChunk, chunk_book
from retrieval_qa.chunking.paper_chunker import PaperChunk, chunk_paper

_Chunk = PaperChunk | BookChunk


def _validate_type_metadata(
    document_type: DocumentType,
    type_metadata: dict[str, object],
) -> None:
    """Validate type_metadata against the Pydantic model for the document type.

    This is the second layer of enforcement (the first being Pydantic
    construction in the chunker itself).  Running it here, just before the
    SQLAlchemy write, ensures that no code path — even one that bypasses the
    chunker — can persist invalid metadata to the JSONB column.

    Raises ``IngestionError`` when validation fails.
    """
    try:
        match document_type:
            case DocumentType.PAPER:
                PaperChunkMetadata.model_validate(type_metadata)
            case DocumentType.BOOK:
                BookChunkMetadata.model_validate(type_metadata)
            case DocumentType.DOCUMENTATION:
                DocumentationChunkMetadata.model_validate(type_metadata)
            case _ as unreachable:
                assert_never(unreachable)
    except ValidationError as exc:
        raise IngestionError(
            f"type_metadata validation failed for {document_type.value}: {exc}"
        ) from exc


def _chunk_inputs(
    chunks: Sequence[_Chunk],
) -> list[tuple[str, dict[str, object], int]]:
    """Convert chunker output to the (content, type_metadata, token_count) shape.

    Both paper and book chunkers expose the same interface, so this helper
    avoids repeating the tuple construction.
    """
    return [(chunk.content, chunk.metadata.model_dump(), chunk.token_count) for chunk in chunks]


def _chunk_document(
    document_type: DocumentType,
    file_bytes: bytes,
) -> list[tuple[str, dict[str, object], int]]:
    """Return (content, type_metadata, token_count) tuples for a document.

    Dispatches to the document-type-specific chunker. The ``match`` /
    ``assert_never`` pair provides compile-time exhaustiveness: adding a new
    ``DocumentType`` member produces a type-check error at every dispatch site.
    """
    match document_type:
        case DocumentType.PAPER:
            return _chunk_inputs(chunk_paper(file_bytes))
        case DocumentType.BOOK:
            return _chunk_inputs(chunk_book(file_bytes))
        case DocumentType.DOCUMENTATION:
            raise IngestionError(f"Chunker not implemented for document type: {document_type}")
        case _ as unreachable:
            assert_never(unreachable)


def _embed_chunks(
    session: Session,
    client: EmbeddingsClient,
    chunks: Sequence[Chunk],
    model_name: str,
) -> list[tuple[Chunk, list[float]]]:
    """Embed chunk contents and return (chunk, vector) pairs."""
    if not chunks:
        return []

    texts = [chunk.content for chunk in chunks]
    try:
        vectors = client.embed(texts)
    except Exception as exc:
        raise IngestionError(f"Embedding call failed: {exc}") from exc

    if len(vectors) != len(chunks):
        raise IngestionError(
            f"Embedding response length mismatch: expected {len(chunks)}, got {len(vectors)}"
        )

    return list(zip(chunks, vectors, strict=True))


def run_ingestion(
    document_id: UUID,
    title: str,
    document_type: DocumentType,
    source_filename: str,
    file_bytes: bytes,
    session: Session,
    embeddings_client: EmbeddingsClient,
    model_name: str,
) -> None:
    """Run the ingestion pipeline inside an open transaction.

    The caller is responsible for committing or rolling back ``session``.
    On success, ``session`` contains a ready document with chunks and
    embeddings attached. On failure, this function raises ``IngestionError``
    and the caller should roll back.

    Args:
        document_id: UUID of the document row created at upload time.
        title: Document title supplied by the user.
        document_type: Document type (e.g. ``DocumentType.PAPER``).
        source_filename: Original upload filename.
        file_bytes: Raw uploaded file contents.
        session: SQLAlchemy session bound to the documents database.
        embeddings_client: Client that produces one vector per chunk text.
        model_name: Model name to store alongside each embedding row.
    """
    _ = title, source_filename  # retained for future metadata use; not needed now

    document = session.get(Document, document_id)
    if document is None:
        raise IngestionError(f"Document {document_id} not found")

    try:
        # validating phase: ensure the file is parseable for the document type.
        chunk_inputs = _chunk_document(document_type, file_bytes)

        document.status = DocumentStatus.CHUNKING
        session.flush()

        db_chunks: list[Chunk] = []
        for position, (content, type_metadata, token_count) in enumerate(chunk_inputs):
            _validate_type_metadata(document_type, type_metadata)
            chunk = Chunk(
                document_id=document_id,
                position=position,
                content=content,
                token_count=token_count,
                type_metadata=type_metadata,
            )
            session.add(chunk)
            db_chunks.append(chunk)

        document.status = DocumentStatus.EMBEDDING
        session.flush()

        embedded = _embed_chunks(session, embeddings_client, db_chunks, model_name)
        for chunk, vector in embedded:
            # The composite PK (chunk_id, model_name) means we can add each
            # embedding directly; re-embedding the same chunk under the same
            # model would raise, but chunks are immutable in the happy path.
            session.add(
                Embedding(
                    chunk_id=chunk.chunk_id,
                    model_name=model_name,
                    embedding=vector,
                )
            )

        document.status = DocumentStatus.READY
        session.flush()
    except IngestionError:
        raise
    except Exception as exc:
        raise IngestionError(f"Unexpected ingestion failure: {exc}") from exc


__all__ = ["run_ingestion"]
