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

    query: str = Field(min_length=1)


class CitedPassage(BaseModel):
    """A retrieved passage cited in a grounded answer.

    ``text`` carries the *full* chunk content (not a truncated preview);
    presentation (truncation for UI display) is the client's responsibility.
    """

    chunk_id: UUID
    text: str


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
