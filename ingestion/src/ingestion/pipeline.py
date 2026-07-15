"""Background ingestion pipeline for uploaded documents."""

from collections.abc import Sequence
from uuid import UUID

from sqlalchemy.orm import Session

from core.clients.embeddings_client import EmbeddingsClient
from core.database.schema import Chunk, Document, Embedding
from core.exceptions import IngestionError
from core.types.document import DocumentStatus
from retrieval_qa.chunking.paper_chunker import chunk_paper


def _chunk_document(
    document_type: str,
    file_bytes: bytes,
) -> list[tuple[str, dict[str, object], int]]:
    """Return (content, type_metadata, token_count) tuples for a document.

    Only paper chunking is implemented in MVP; other document types are
    deliberately rejected so the schema stays document-type-aware without
    guessing at chunkers that are out of scope.
    """
    if document_type == "paper":
        chunks = chunk_paper(file_bytes)
        return [(chunk.content, chunk.metadata.model_dump(), chunk.token_count) for chunk in chunks]

    raise IngestionError(f"Chunker not implemented for document type: {document_type}")


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
    document_type: str,
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
        document_type: Lower-case document type (e.g. ``"paper"``).
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
