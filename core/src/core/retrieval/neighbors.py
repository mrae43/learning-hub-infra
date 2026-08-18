"""Dense semantic-neighbor search against the pgvector embeddings table.

The dense-only counterpart of the full hybrid pipeline (ADR-0019): given a
query vector, return the closest embedded chunks from the global corpus —
no sparse path, no RRF fusion, no parent-swap, no reranking. Depth Dive
composes it for the "up to K semantic neighbors" half of anchored retrieval
and for the unanchored-passage corpus-similarity gate; the gate's threshold
and grounded/unverified judgement stay harness policy in ``depth_dive``
(ADR-0019), which is why this primitive exposes cosine similarity as the
score instead of rank-based RRF contributions.

Visibility matches the pipeline (ADR-0014): embeddings are scoped to
``model_name`` and only chunks of ``status='ready'`` documents are
queryable. Query embedding stays caller-side via ``core.clients.Embedder``;
the primitive is vector-in.
"""

from sqlalchemy import text
from sqlalchemy.exc import InterfaceError, OperationalError
from sqlalchemy.orm import Session

from core.exceptions import UpstreamUnavailable
from core.telemetry import stage_span
from core.types.responses import ScoredChunk
from core.types.retrieval_config import RetrievalConfig

_DENSE_NEIGHBORS_SQL = text(
    """
    SELECT
        chunks.chunk_id          AS chunk_id,
        chunks.content           AS text,
        chunks.parent_chunk_id   AS parent_chunk_id,
        1 - (embeddings.embedding <=> CAST(:query_vector AS vector(1536))) AS score
    FROM embeddings
    JOIN chunks  ON chunks.chunk_id  = embeddings.chunk_id
    JOIN documents ON documents.document_id = chunks.document_id
    WHERE embeddings.model_name = :model_name
      AND documents.status = 'ready'
    ORDER BY embeddings.embedding <=> CAST(:query_vector AS vector(1536))
    LIMIT :top_k
    """
)


@stage_span("retrieve")
def search_dense_neighbors(
    *,
    session: Session,
    query_vector: list[float],
    config: RetrievalConfig,
) -> list[ScoredChunk]:
    """Return the closest embedded chunks to ``query_vector``, closest first.

    Dense-only search over the global corpus: no sparse fusion, no
    parent-swap, no reranking. Issues ``SET LOCAL hnsw.ef_search`` inside
    the caller's transaction, matching the full pipeline.

    Args:
        session: SQLAlchemy session bound to the documents database. The
            ``SET LOCAL`` is scoped to the current transaction, so callers
            should not commit before consuming the results.
        query_vector: A 1536-dim query embedding (caller-side, via
            ``core.clients.Embedder``).
        config: Retrieval configuration. Uses ``model_name`` (embedding
            provenance filter), ``ef_search`` (HNSW candidate-list size),
            and ``top_k`` (maximum number of neighbors); the
            ``hybrid_search`` and ``reranker`` toggles do not apply to
            this primitive.

    Returns:
        Up to ``config.top_k`` ``ScoredChunk`` instances ordered by cosine
        closeness, with ``score`` set to cosine similarity
        (``1 - cosine distance``; 1.0 identical direction, 0.0 orthogonal,
        -1.0 opposite) so callers can apply their own similarity
        thresholds. ``parent_chunk_id`` is carried through for child
        chunks. An empty list is a valid result (empty corpus or no
        embeddings under ``config.model_name``), never an exception.

    Raises:
        UpstreamUnavailable: The database is unreachable (maps to 503 per
            ADR-0014 § Error contract). Other DB-level errors (e.g. a
            dimension mismatch from a bad query vector) propagate verbatim.
    """
    try:
        session.execute(
            text("SET LOCAL hnsw.ef_search = :ef_search"), {"ef_search": config.ef_search}
        )
        result = session.execute(
            _DENSE_NEIGHBORS_SQL,
            {
                "model_name": config.model_name,
                "query_vector": str(query_vector),
                "top_k": config.top_k,
            },
        )

        neighbors: list[ScoredChunk] = []
        for row in result.fetchall():
            neighbors.append(
                ScoredChunk(
                    chunk_id=row._mapping["chunk_id"],
                    text=row._mapping["text"],
                    score=row._mapping["score"],
                    parent_chunk_id=row._mapping.get("parent_chunk_id"),
                )
            )
        return neighbors

    except (OperationalError, InterfaceError) as exc:
        raise UpstreamUnavailable(f"Database unreachable: {exc}") from exc


__all__ = ["search_dense_neighbors"]
