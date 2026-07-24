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

When ``RetrievalConfig.hybrid_search`` is True (default, ADR-0016):

- A parallel sparse search via ``to_tsvector`` / ``websearch_to_tsquery`` on
  chunk content recovers exact-match queries (function names, API endpoints,
  error codes) that pure dense retrieval misses.
- Dense and sparse results are fused via Reciprocal Rank Fusion (RRF) with
  k=60, producing a single ranked set with no duplicates.
- After fusion, matched child chunks are swapped to their parent chunks
  (parent-swap): the parent's content replaces the child's in the cited
  passage, and duplicate parents are deduplicated.
- When one path returns zero results, the other path's results alone still
  produce output (graceful degradation).

When ``hybrid_search`` is False, retrieval falls back to dense-only
(existing behavior).

When ``RetrievalConfig.reranker`` is True (default, ADR-0016):

- After RRF fusion and parent swap, the top-20 parent passages are reranked
  via the configured ``Reranker``. Only the top-k (default 5) reach the prompt.
- When the reranker is ``NoopReranker`` (or ``reranker`` is False), the
  RRF top-k is used directly.
- When the reranker raises ``RerankerRateLimitError`` (Cohere 429), the system
  logs the error and falls back to RRF top-k (no crash, no 5xx).
"""

import logging
from collections import OrderedDict
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.exc import InterfaceError, OperationalError
from sqlalchemy.orm import Session

from core.clients.reranker_client import Reranker
from core.exceptions import RerankerRateLimitError, UpstreamUnavailable
from core.types.responses import CitedPassage
from core.types.retrieval_config import RetrievalConfig

logger = logging.getLogger(__name__)

_RRF_K = 60

# When reranking is active, fetch more candidates from each retrieval path
# to give the reranker a bigger pool to re-rank from (ADR-0016).
_RERANK_CANDIDATE_COUNT = 20

_DENSE_SQL = text(
    """
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

_SPARSE_SQL = text(
    """
    SELECT
        chunks.chunk_id AS chunk_id,
        chunks.content  AS text,
        ts_rank(
            to_tsvector('english', chunks.content),
            websearch_to_tsquery('english', :query_text)
        ) AS rank
    FROM chunks
    JOIN documents ON documents.document_id = chunks.document_id
    WHERE documents.status = 'ready'
      AND to_tsvector('english', chunks.content)
          @@ websearch_to_tsquery('english', :query_text)
    ORDER BY rank DESC
    LIMIT :top_k
    """
)

_PARENT_SWAP_SQL = text(
    """
    SELECT
        children.chunk_id       AS child_chunk_id,
        COALESCE(parents.chunk_id, children.chunk_id) AS chunk_id,
        COALESCE(parents.content, children.content)   AS text
    FROM chunks children
    LEFT JOIN chunks parents ON children.parent_chunk_id = parents.chunk_id
    WHERE children.chunk_id IN :chunk_ids
    """
)


def _dense_retrieve(
    session: Session,
    query_vector: list[float],
    config: RetrievalConfig,
    limit: int | None = None,
) -> OrderedDict[UUID, float]:
    """Run dense (pgvector cosine) search and return chunk_id -> RRF score."""
    top_k = limit if limit is not None else config.top_k
    result = session.execute(
        _DENSE_SQL,
        {
            "model_name": config.model_name,
            "query_vector": str(query_vector),
            "top_k": top_k,
        },
    )
    scored: OrderedDict[UUID, float] = OrderedDict()
    for rank, row in enumerate(result.fetchall(), 1):
        chunk_id = row._mapping["chunk_id"]
        scored[chunk_id] = 1.0 / (_RRF_K + rank)
    return scored


def _sparse_retrieve(
    session: Session,
    query_text: str,
    config: RetrievalConfig,
    limit: int | None = None,
) -> OrderedDict[UUID, float]:
    """Run sparse (tsvector ts_rank) search and return chunk_id -> RRF score."""
    top_k = limit if limit is not None else config.top_k
    result = session.execute(
        _SPARSE_SQL,
        {
            "query_text": query_text,
            "top_k": top_k,
        },
    )
    scored: OrderedDict[UUID, float] = OrderedDict()
    for rank, row in enumerate(result.fetchall(), 1):
        chunk_id = row._mapping["chunk_id"]
        scored[chunk_id] = 1.0 / (_RRF_K + rank)
    return scored


def _rrf_fuse(
    dense_scored: OrderedDict[UUID, float],
    sparse_scored: OrderedDict[UUID, float],
    top_k: int,
) -> OrderedDict[UUID, float]:
    """Fuse dense and sparse scores via Reciprocal Rank Fusion.

    When one path returns zero results, the other path's results alone produce
    the output (graceful degradation). Duplicate chunk_ids (present in both
    paths) receive a combined score.

    Args:
        dense_scored: chunk_id -> RRF score from dense path.
        sparse_scored: chunk_id -> RRF score from sparse path.
        top_k: Maximum number of fused results to return.

    Returns:
        An OrderedDict of chunk_id -> combined RRF score, sorted descending
        by score and limited to top_k.
    """
    combined: dict[UUID, float] = {}

    for chunk_id, score in dense_scored.items():
        combined[chunk_id] = combined.get(chunk_id, 0.0) + score
    for chunk_id, score in sparse_scored.items():
        combined[chunk_id] = combined.get(chunk_id, 0.0) + score

    sorted_pairs = sorted(combined.items(), key=lambda item: item[1], reverse=True)
    return OrderedDict(sorted_pairs[:top_k])


def _parent_swap(
    session: Session,
    scored_chunks: OrderedDict[UUID, float],
) -> list[CitedPassage]:
    """Swap matched child chunks to their parents and deduplicate.

    For each chunk in scored_chunks (ordered by descending RRF score),
    resolve its parent. If the chunk has a parent, use the parent's content
    and chunk_id. If multiple children belong to the same parent, only the
    highest-scoring child is retained.

    Chunks that are themselves parents (``parent_chunk_id IS NULL``) are
    returned as-is.

    Args:
        session: SQLAlchemy session.
        scored_chunks: chunk_id -> RRF score, ordered descending by score.

    Returns:
        A deduplicated list of CitedPassage instances preserving RRF rank order.
    """
    if not scored_chunks:
        return []

    chunk_ids = tuple(scored_chunks.keys())
    result = session.execute(
        _PARENT_SWAP_SQL,
        {"chunk_ids": chunk_ids},
    )

    # Build child -> (parent_chunk_id, parent_text) map
    child_to_parent: dict[UUID, tuple[UUID, str]] = {}
    for row in result.fetchall():
        child_id = row._mapping["child_chunk_id"]
        parent_id = row._mapping["chunk_id"]
        parent_text = row._mapping["text"]
        child_to_parent[child_id] = (parent_id, parent_text)

    # Walk the scored chunks in RRF rank order, deduplicating by parent
    seen_parents: set[UUID] = set()
    passages: list[CitedPassage] = []

    for child_chunk_id in scored_chunks:
        parent_info = child_to_parent.get(child_chunk_id)
        if parent_info is None:
            continue
        parent_id, parent_text = parent_info

        if parent_id in seen_parents:
            continue
        seen_parents.add(parent_id)

        passages.append(CitedPassage(chunk_id=parent_id, text=parent_text))

    return passages


def retrieve_relevant_chunks(
    *,
    query_vector: list[float],
    session: Session,
    config: RetrievalConfig,
    query_text: str | None = None,
    reranker: Reranker | None = None,
) -> list[CitedPassage]:
    """Retrieve the top-k closest chunks for ``query_vector`` by cosine distance.

    Issues ``SET LOCAL hnsw.ef_search`` inside the caller's transaction and
    returns chunks only from documents whose status is ``ready``.

    When ``config.hybrid_search`` is True (default), also runs a sparse
    tsvector search against chunk content and fuses results via Reciprocal
    Rank Fusion, then performs parent-swap to replace matched child chunks
    with their enclosing parent chunks (ADR-0016).

    When ``config.reranker`` is True and a ``reranker`` is supplied, the
    top-20 RRF-fused parent passages are reranked and narrowed to ``top_k``.
    On rate-limit errors, the system logs a warning and falls back to RRF
    top-k (no crash).

    Args:
        query_vector: A 1536-dim query embedding.
        session: SQLAlchemy session bound to the documents database. The
            ``SET LOCAL`` is scoped to the current transaction, so callers
            should not commit before consuming the results.
        config: Retrieval configuration (model name, ef_search, top_k,
            hybrid_search toggle, reranker toggle).
        query_text: The original user query string, required for the sparse
            search path when ``config.hybrid_search`` is True. Ignored when
            hybrid_search is False.
        reranker: Reranker instance for cross-encoder reranking. When None
            or when ``config.reranker`` is False, the RRF top-k is used
            directly.

    Returns:
        Chunks in descending relevance order. An empty list signals the
        not-found case (empty corpus or no relevant chunks), which is a valid
        response per ADR-0009 — never raised as an exception.

    Raises:
        UpstreamUnavailable: The database is unreachable (maps to 503 per
            ADR-0014 § Error contract). Other DB-level errors (e.g. a
            dimension mismatch from a bad query vector) propagate verbatim.
    """
    try:
        session.execute(
            text("SET LOCAL hnsw.ef_search = :ef_search"), {"ef_search": config.ef_search}
        )

        should_rerank = config.reranker and reranker is not None
        retrieval_limit = _RERANK_CANDIDATE_COUNT if should_rerank else config.top_k

        if config.hybrid_search and query_text:
            dense_scored = _dense_retrieve(session, query_vector, config, limit=retrieval_limit)
            sparse_scored = _sparse_retrieve(session, query_text, config, limit=retrieval_limit)
            fused = _rrf_fuse(dense_scored, sparse_scored, retrieval_limit)
            passages = _parent_swap(session, fused)
        else:
            # Dense-only path (hybrid_search=False or no query_text provided)
            rows = session.execute(
                _DENSE_SQL,
                {
                    "model_name": config.model_name,
                    "query_vector": str(query_vector),
                    "top_k": retrieval_limit,
                },
            )
            passages = [
                CitedPassage(
                    chunk_id=row._mapping["chunk_id"],
                    text=row._mapping["text"],
                )
                for row in rows.fetchall()
            ]

        if should_rerank and passages and reranker is not None:
            try:
                passages = reranker.rerank(
                    query=query_text or "",
                    passages=passages,
                    top_k=config.top_k,
                )
            except RerankerRateLimitError as exc:
                logger.warning("Reranker rate-limited, falling back to RRF top-k: %s", exc)
                passages = passages[: config.top_k]

        return passages

    except (OperationalError, InterfaceError) as exc:
        raise UpstreamUnavailable(f"Database unreachable: {exc}") from exc


__all__ = ["CitedPassage", "retrieve_relevant_chunks"]
