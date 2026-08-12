"""Parent-chunk fetch for anchored retrieval (ADR-0019).

Resolves a chunk anchor to its parent context: Depth Dive anchors
``text``/``code`` captured passages via ``chunk_id`` (ADR-0014) and needs
the enclosing Parent Chunk (CONTEXT.md) for the dive's corpus context.
Harness A's cited passages carry parent ids after parent-swap, so an anchor
may equally point at a parent row — the fetch resolves both to the passage
that reaches generation.
"""

from uuid import UUID

from sqlalchemy import text
from sqlalchemy.exc import InterfaceError, OperationalError
from sqlalchemy.orm import Session

from core.exceptions import UpstreamUnavailable
from core.types.responses import CitedPassage

_PARENT_CHUNK_SQL = text(
    """
    SELECT
        COALESCE(parents.chunk_id, children.chunk_id) AS chunk_id,
        COALESCE(parents.content, children.content)   AS text
    FROM chunks children
    JOIN documents ON documents.document_id = children.document_id
    LEFT JOIN chunks parents ON children.parent_chunk_id = parents.chunk_id
    WHERE children.chunk_id = :chunk_id
      AND documents.status = 'ready'
    """
)


def fetch_parent_chunk(*, session: Session, chunk_id: UUID) -> CitedPassage | None:
    """Resolve a chunk anchor to the passage that carries its parent context.

    Args:
        session: SQLAlchemy session bound to the documents database.
        chunk_id: The anchor chunk id (a captured-passage ``chunk_id``
            anchor per ADR-0014, or a cited-passage id from Harness A).

    Returns:
        The enclosing parent chunk as a ``CitedPassage`` when the anchor is
        a child chunk; the chunk itself when it is a parent or standalone
        chunk (``parent_chunk_id IS NULL``); ``None`` when no visible chunk
        has that id — the chunk does not exist, or its document is not in
        ``status='ready'`` (the same visibility gate as the retrieval
        pipeline, ADR-0014). Callers treat ``None`` as an unresolvable
        anchor, never as an error.

    Raises:
        UpstreamUnavailable: The database is unreachable (maps to 503 per
            ADR-0014 § Error contract). Other DB-level errors propagate
            verbatim.
    """
    try:
        row = session.execute(_PARENT_CHUNK_SQL, {"chunk_id": chunk_id}).first()
        if row is None:
            return None
        return CitedPassage(chunk_id=row._mapping["chunk_id"], text=row._mapping["text"])

    except (OperationalError, InterfaceError) as exc:
        raise UpstreamUnavailable(f"Database unreachable: {exc}") from exc


__all__ = ["fetch_parent_chunk"]
