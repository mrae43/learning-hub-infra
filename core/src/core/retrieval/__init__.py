"""Shared corpus retrieval layer (ADR-0019).

The corpus retrieval primitives both harnesses consume: Retrieval QA runs
the full hybrid pipeline, and Depth Dive builds its anchored/unanchored
retrieval on the dense-neighbor and parent-chunk primitives.
``retrieval_qa`` and ``depth_dive`` never import each other; both import
from here (ADR-0011).

Public surface:

- ``retrieve_relevant_chunks`` — the full pipeline (dense + sparse, RRF
  fusion, parent-swap, rerank orchestration).
- ``search_dense_neighbors`` — dense semantic-neighbor search scored by
  cosine similarity.
- ``fetch_parent_chunk`` — parent-chunk fetch for an anchored chunk id.
- ``fetch_document_chunks`` — document-relative parent-chunk fetch for an
  anchored non-text ``document_id`` (ticket #253).

Sparse search and RRF fusion stay private internals of the pipeline module.
"""

from core.retrieval.documents import fetch_document_chunks
from core.retrieval.neighbors import search_dense_neighbors
from core.retrieval.parents import fetch_parent_chunk
from core.retrieval.query import retrieve_relevant_chunks

__all__ = [
    "fetch_document_chunks",
    "fetch_parent_chunk",
    "retrieve_relevant_chunks",
    "search_dense_neighbors",
]
