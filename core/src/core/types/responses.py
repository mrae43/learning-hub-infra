"""Harness A request/response Pydantic models (ADR-0014)."""

from uuid import UUID

from pydantic import BaseModel, Field


class HarnessARequest(BaseModel):
    """Request body for ``POST /query``.

    Bare string, no filters: ADR-0014 locks the cross-corpus default retrieval
    behavior and rejects adding ``document_id``/``document_type`` filters as
    API-YAGNI. The empty-string query is rejected here so FastAPI returns 422
    for it (alongside the missing-field case).
    """

    query: str = Field(min_length=1, max_length=2000)


class CitedPassage(BaseModel):
    """A retrieved passage cited in a grounded answer.

    ``text`` carries the *full* chunk content (not a truncated preview);
    presentation (truncation for UI display) is the client's responsibility.
    """

    chunk_id: UUID
    text: str


class ScoredChunk(BaseModel):
    """A retrieved chunk with its relevance score.

    ``chunk_id``, ``text`` mirror ``CitedPassage`` so downstream consumers
    that access ``.chunk_id`` and ``.text`` continue to work.
    ``parent_chunk_id`` is the chunk id of the parent for child chunks, or
    ``None`` for standalone/parent chunks.
    """

    chunk_id: UUID
    text: str
    score: float = Field(
        description=(
            "Relevance score. In ``RetrievalResult.dense`` and "
            "``RetrievalResult.sparse`` this is the per-path RRF "
            "contribution (``1 / (_RRF_K + rank)``); in "
            "``RetrievalResult.fused`` it is the combined RRF score."
        )
    )
    parent_chunk_id: UUID | None = None


class RetrievalResult(BaseModel):
    """Result of a retrieval operation preserving pre-fusion candidate depth.

    ``dense``: results from the dense (pgvector) path with their scores.
    ``sparse``: results from the sparse (tsvector) path with their scores.
    ``fused``: the final fused/reranked list of ``ScoredChunk`` instances.

    Production callers access ``.fused`` (behaviourally identical to the
    previous ``list[CitedPassage]``). Test and introspection code can
    inspect ``.dense`` and ``.sparse`` separately.
    """

    dense: list[ScoredChunk]
    sparse: list[ScoredChunk]
    fused: list[ScoredChunk]


class HarnessAResponse(BaseModel):
    """Response body for ``POST /query``.

    Exactly three fields, no observability block: ``answer`` is always
    populated (model-generated refusal text when ``grounded=False``);
    ``cited_passages`` is empty on the not-found branch. Per ADR-0014 / ADR-0009
    the not-found case is a valid response, not an exception.
    """

    answer: str
    cited_passages: list[CitedPassage]
    grounded: bool
