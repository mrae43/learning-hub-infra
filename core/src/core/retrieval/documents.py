"""Document-relative text lookup for non-text passages (ADR-0019, ticket #253).

Depth Dive anchors ``image``/``diagram``/``table`` captured passages via
``document_id`` + ``ordinal`` (ADR-0021). Anchored non-text retrieval fetches
document-relative text context near the figure/table plus semantic neighbors on
the passage's embeddable representation (depth-dive spec §8).

The store is text-only today: non-text assets have no rows of their own, and no
``ordinal`` → position mapping exists. This primitive therefore returns the
best-effort document-relative context available now — the document's parent
chunks (its structural units) in reading order — leaving the "near the
figure/table" narrowing to the future ingestion-redesign map, whose storage
contract (spec §10) adds ordinal identity and nearest-chunk position. The
semantic-neighbor half is composed by the harness from
:func:`core.retrieval.neighbors.search_dense_neighbors` on the passage's
embeddable representation.

Visibility matches the pipeline (ADR-0014): only chunks of ``status='ready'``
documents are returned. ``None`` signals an unresolvable anchor (the document
does not exist or is not ready), mirroring :func:`fetch_parent_chunk`.
"""

from uuid import UUID

from sqlalchemy import text
from sqlalchemy.exc import InterfaceError, OperationalError
from sqlalchemy.orm import Session

from core.exceptions import UpstreamUnavailable
from core.types.responses import CitedPassage

_DOCUMENT_READY_SQL = text(
    """
    SELECT 1
    FROM documents
    WHERE document_id = :document_id
      AND status = 'ready'
    """
)

_DOCUMENT_PARENT_CHUNKS_SQL = text(
    """
    SELECT
        chunks.chunk_id AS chunk_id,
        chunks.content  AS text
    FROM chunks
    JOIN documents ON documents.document_id = chunks.document_id
    WHERE chunks.document_id = :document_id
      AND documents.status = 'ready'
      AND chunks.parent_chunk_id IS NULL
    ORDER BY chunks.position
    LIMIT :limit
    """
)


def fetch_document_chunks(
    *,
    session: Session,
    document_id: UUID,
    limit: int,
) -> list[CitedPassage] | None:
    """Fetch a ready document's parent chunks in document order.

    The document-relative text context for an anchored non-text passage: the
    document's structural units (parent chunks, ``parent_chunk_id IS NULL``) in
    reading order, capped at ``limit``. Until the ingestion-redesign map adds
    ordinal → position metadata, this is the whole-document stand-in for "text
    context near the figure/table".

    Args:
        session: SQLAlchemy session bound to the documents database.
        document_id: The non-text passage's ``document_id`` anchor
            (ADR-0021).
        limit: Maximum number of parent chunks to return.

    Returns:
        The document's parent chunks as ``CitedPassage`` instances ordered by
        ``position``, or ``None`` when the document does not exist or is not in
        ``status='ready'`` (the same visibility gate as the retrieval pipeline,
        ADR-0014). Callers treat ``None`` as an unresolvable anchor, never as
        an error. An empty list means the ready document has no parent chunks.

    Raises:
        UpstreamUnavailable: The database is unreachable (maps to 503 per
            ADR-0014 § Error contract). Other DB-level errors propagate
            verbatim.
    """
    try:
        # Two queries: the first distinguishes an unresolvable anchor (document
        # absent or not ready → None) from a resolvable anchor with no parent
        # chunks (→ []), which callers treat differently.
        ready = session.execute(_DOCUMENT_READY_SQL, {"document_id": document_id}).first()
        if ready is None:
            return None
        rows = session.execute(
            _DOCUMENT_PARENT_CHUNKS_SQL,
            {"document_id": document_id, "limit": limit},
        ).fetchall()
        return [
            CitedPassage(chunk_id=row._mapping["chunk_id"], text=row._mapping["text"])
            for row in rows
        ]

    except (OperationalError, InterfaceError) as exc:
        raise UpstreamUnavailable(f"Database unreachable: {exc}") from exc


__all__ = ["fetch_document_chunks"]
