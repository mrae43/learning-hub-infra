"""Retrieval query against the pgvector embeddings table.

Implements the retrieval half of Harness A's RAG pipeline (ADR-0014):

- ``ORDER BY embedding <=> :query_vector LIMIT :k`` against the
  ``embeddings`` table, scoped to ``model_name = :model_name`` (ADR-0014's
  provenance column, not a dimensional selector).
- ``SET LOCAL hnsw.ef_search = :val`` inside the retrieval transaction so the
  HNSW candidate list size is operator-controlled via ``settings.hnsw_ef_search``
  without leaking into the API contract.
- ``JOIN documents ON status='ready'`` is the chunk-visibility gate: chunks
  of documents still in ``validating``/``chunking``/``embedding`` (or
  ``failed``) are not queryable (ADR-0014).
- Fetches full chunk content (for prompt construction + ``CitedPassage.text``)
  and ``chunk_id`` (for ``CitedPassage.chunk_id``) in the same query — no
  second round-trip, since the chunk content is already needed for the prompt.

The cosine distance operator ``<=>`` matches the HNSW index's
``vector_cosine_ops`` opclass declared in the Alembic migration (ADR-0014).
"""

from uuid import UUID

from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.exc import InterfaceError, OperationalError
from sqlalchemy.orm import Session

from core.exceptions import UpstreamUnavailable


class RetrievedChunk(BaseModel):
    """A retrieved chunk with the fields needed downstream.

    ``text`` mirrors ``CitedPassage.text`` (full chunk content) so the QA
    controller can build both the prompt and the response from one shape
    without re-fetching.
    """

    chunk_id: UUID
    text: str


_RETRIEVE_SQL = text(
    """
    SET LOCAL hnsw.ef_search = :ef_search;

    SELECT
        chunks.chunk_id AS chunk_id,
        chunks.content  AS text
    FROM embeddings
    JOIN chunks  ON chunks.chunk_id  = embeddings.chunk_id
    JOIN documents ON documents.document_id = chunks.document_id
    WHERE embeddings.model_name = :model_name
      AND documents.status = 'ready'
    ORDER BY embeddings.embedding <=> CAST(:query_vector AS vector(1536))
    LIMIT :top_k
    """
)


def retrieve_relevant_chunks(
    *,
    query_vector: list[float],
    session: Session,
    model_name: str,
    ef_search: int,
    top_k: int,
) -> list[RetrievedChunk]:
    """Retrieve the top-k closest chunks for ``query_vector`` by cosine distance.

    Issues ``SET LOCAL hnsw.ef_search`` inside the caller's transaction and
    returns chunks only from documents whose status is ``ready``.

    Args:
        query_vector: A 1536-dim query embedding.
        session: SQLAlchemy session bound to the documents database. The
            ``SET LOCAL`` is scoped to the current transaction, so callers
            should not commit before consuming the results.
        model_name: Embedding model name to filter on (provenance column).
        ef_search: HNSW candidate-list size for this query.
        top_k: Maximum number of chunks to return.

    Returns:
        Chunks in ascending cosine-distance order. An empty list signals the
        not-found case (empty corpus or no relevant chunks), which is a valid
        response per ADR-0009 — never raised as an exception.

    Raises:
        UpstreamUnavailable: The database is unreachable (maps to 503 per
            ADR-0014 § Error contract). Other DB-level errors (e.g. a
            dimension mismatch from a bad query vector) propagate verbatim.
    """
    try:
        result = session.execute(
            _RETRIEVE_SQL,
            {
                "ef_search": ef_search,
                "model_name": model_name,
                "query_vector": str(query_vector),
                "top_k": top_k,
            },
        )
    except (OperationalError, InterfaceError) as exc:
        raise UpstreamUnavailable(f"Database unreachable: {exc}") from exc
    rows = result.fetchall()
    chunks: list[RetrievedChunk] = []
    for row in rows:
        chunk_id = row._mapping["chunk_id"]
        text_content = row._mapping["text"]
        chunks.append(RetrievedChunk(chunk_id=chunk_id, text=text_content))
    return chunks


__all__ = ["RetrievedChunk", "retrieve_relevant_chunks"]
