"""Background ingestion pipeline for uploaded documents."""

from collections.abc import Sequence

from pydantic import ValidationError
from sqlalchemy.orm import Session

from core.clients import Embedder
from core.database.schema import Chunk as ChunkRow
from core.database.schema import Document, Embedding
from core.exceptions import IngestionError
from core.types.chunk import Chunk
from core.types.document import DocumentStatus, DocumentType
from ingestion.models import PendingIngestion
from retrieval_qa.chunking import chunker_registry, get_chunker_class


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
        chunker_class = chunker_registry[document_type]
    except KeyError as exc:
        raise IngestionError(f"No chunker registered for {document_type.value}") from exc
    try:
        chunker_class.metadata_model.model_validate(type_metadata)
    except ValidationError as exc:
        raise IngestionError(
            f"type_metadata validation failed for {document_type.value}: {exc}"
        ) from exc


def _chunk_inputs(
    chunks: Sequence[Chunk],
) -> list[tuple[str, dict[str, object], int]]:
    """Convert chunker output to the (content, type_metadata, token_count) shape.

    All chunkers expose the same ``Chunk`` protocol, so this helper avoids
    repeating the tuple construction for each document type.
    """
    return [(chunk.content, chunk.metadata.model_dump(), chunk.token_count) for chunk in chunks]


def _chunk_document(
    document_type: DocumentType,
    file_bytes: bytes,
) -> list[tuple[str, dict[str, object], int]]:
    """Return (content, type_metadata, token_count) tuples for a document.

    Dispatches to the document-type-specific chunker via the registry. Adding
    a new ``DocumentType`` only requires registering a new chunker; this
    function does not need to change.
    """
    chunker_class = get_chunker_class(document_type)
    return _chunk_inputs(chunker_class().chunk(file_bytes))


def _embed_chunks(
    session: Session,
    client: Embedder,
    chunks: Sequence[ChunkRow],
    model_name: str,
) -> list[tuple[ChunkRow, list[float]]]:
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
    pending: PendingIngestion,
    session: Session,
    embeddings_client: Embedder,
    model_name: str,
) -> None:
    """Run the ingestion pipeline inside an open transaction.

    The caller is responsible for committing or rolling back ``session``.
    On success, ``session`` contains a ready document with chunks and
    embeddings attached. On failure, this function raises ``IngestionError``
    and the caller should roll back.

    Args:
        pending: Document-identity fields for the ingestion.
        session: SQLAlchemy session bound to the documents database.
        embeddings_client: Provider that produces one vector per chunk text.
        model_name: Model name to store alongside each embedding row.
    """
    document = session.get(Document, pending.document_id)
    if document is None:
        raise IngestionError(f"Document {pending.document_id} not found")

    try:
        # validating phase: ensure the file is parseable for the document type.
        chunk_inputs = _chunk_document(pending.document_type, pending.file_bytes)

        document.status = DocumentStatus.CHUNKING
        session.flush()

        db_chunks: list[ChunkRow] = []
        for position, (content, type_metadata, token_count) in enumerate(chunk_inputs):
            _validate_type_metadata(pending.document_type, type_metadata)
            chunk = ChunkRow(
                document_id=pending.document_id,
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
