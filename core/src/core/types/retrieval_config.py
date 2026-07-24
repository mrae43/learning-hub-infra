"""Retrieval configuration model.

Encapsulates the parameters that control pgvector similarity search:
embedding model provenance filter, HNSW candidate-list size, and result
limit. Carried as a single object across the retrieval boundary so that
adding a new parameter (e.g. ``rerank_k``, ``min_score``) does not require
threading it through every call site.
"""

from pydantic import BaseModel, ConfigDict


class RetrievalConfig(BaseModel):
    """Configuration for a pgvector similarity-search query.

    Attributes:
        model_name: Active embedding model name (provenance filter on
            ``embeddings.model_name``).
        ef_search: HNSW query-time candidate-list size (``hnsw.ef_search``).
        top_k: Maximum number of chunks to return.
        hybrid_search: When True (default), combine dense (pgvector cosine)
            and sparse (tsvector ts_rank) results via Reciprocal Rank Fusion
            and perform parent-swap before returning (ADR-0016).
            When False, fall back to dense-only retrieval.
        reranker: When True (default), the top-20 RRF-fused parent passages
            are reranked through a ``Reranker`` and narrowed to ``top_k``.
            When False, the RRF top-k is used directly (ADR-0016).
    """

    model_config = ConfigDict(extra="forbid")

    model_name: str
    ef_search: int
    top_k: int
    hybrid_search: bool = True
    reranker: bool = True


__all__ = ["RetrievalConfig"]
