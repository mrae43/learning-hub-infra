"""Retry-once web-search wrapper with fallback (ADR-0013, ticket #244).

The assembly agent calls :func:`run_web_search` when the framing brief carries
a ``search_intent``. The wrapper bounds the failure path: one retry for a
failed or result-less search, then a graceful fallback that flags the missing
external grounding with a user-facing note instead of silently degrading.
"""

from pydantic import BaseModel, ConfigDict, Field

from depth_dive.web_search.client import WebSearchClient, WebSearchError, WebSearchResult

FALLBACK_NOTE = (
    "External grounding was sought but could not be retrieved; "
    "this dive reflects the ingested corpus only."
)
"""User-facing note attached to a dive whose web search failed twice (ADR-0013)."""


class SearchOutcome(BaseModel):
    """The outcome of the web-search step for one dive (ADR-0013).

    ``attempted`` is True whenever the wrapper issued at least one search;
    ``failed`` is True when the retry also failed or returned nothing useful,
    in which case ``note`` carries the user-facing fallback message and
    ``results`` is empty. A successful outcome carries the external material
    for the generation step (ticket #245).
    """

    model_config = ConfigDict(extra="forbid")

    attempted: bool
    failed: bool = False
    note: str | None = None
    results: list[WebSearchResult] = Field(default_factory=list)


def run_web_search(query: str, client: WebSearchClient) -> SearchOutcome:
    """Run one web search with retry-once and fallback (ADR-0013).

    Args:
        query: The framing brief's ``search_intent`` for this dive.
        client: The web-search provider.

    Returns:
        A ``SearchOutcome``. An empty or whitespace-only query is not a search
        (``attempted=False``). A failed or result-less attempt is retried
        exactly once; a second failure yields ``failed=True`` with a
        user-facing ``note`` and no results. Failures never propagate — the
        dive still renders, just without external material.
    """
    if not query.strip():
        return SearchOutcome(attempted=False)
    first = _attempt(query, client)
    if first is not None and first:
        return SearchOutcome(attempted=True, results=first)
    second = _attempt(query, client)
    if second is not None and second:
        return SearchOutcome(attempted=True, results=second)
    return SearchOutcome(attempted=True, failed=True, note=FALLBACK_NOTE)


def _attempt(query: str, client: WebSearchClient) -> list[WebSearchResult] | None:
    """Run one search attempt; ``None`` means the provider call failed.

    An empty result list is a valid return but counts as "no useful results"
    (ADR-0013), so the wrapper treats it like a failure and retries.
    """
    try:
        return client.search(query)
    except WebSearchError:
        return None


__all__ = ["FALLBACK_NOTE", "SearchOutcome", "run_web_search"]
