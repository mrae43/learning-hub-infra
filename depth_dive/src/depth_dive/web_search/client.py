"""Web-search provider clients for Depth Dive (ADR-0012, ADR-0013).

Depth Dive is permitted to extend beyond the ingested corpus via web search.
The framing agent judges per-request whether external grounding would help
(ADR-0012); this module owns the provider call. A failed search is a *valid,
degraded* outcome — the retry-once wrapper converts ``WebSearchError`` into a
fallback with a user-facing note (ADR-0013), so web-search failures never map
to the route's 502/503 upstream errors.
"""

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from openai import OpenAI, OpenAIError
from openai.types.responses import (
    Response,
    ResponseFunctionWebSearch,
    ResponseOutputMessage,
    ResponseOutputText,
)
from openai.types.responses.response_output_text import AnnotationURLCitation
from pydantic import BaseModel, ConfigDict

from core.config.settings import settings


class WebSearchError(Exception):
    """A web-search provider call failed.

    Depth Dive's own failure type (ADR-0013). ``WebSearchError`` is *not* a
    ``RetrievalError`` subclass on purpose: a failed search is a graceful
    fallback for the dive, not an upstream error the route should map to
    502/503.
    """


class WebSearchResult(BaseModel):
    """One external-material item surfaced by web search (ADR-0012).

    ``snippet`` is the short, quote-safe span actually cited by the search,
    and ``url`` is its source — short quotes only, always paired with a
    source citation (ADR-0012).
    """

    model_config = ConfigDict(extra="forbid")

    title: str
    url: str
    snippet: str


@runtime_checkable
class WebSearchClient(Protocol):
    """Protocol for synchronous web-search providers.

    Consumers depend on this protocol rather than a concrete client so that
    hosted API clients, in-memory test doubles, and future providers are
    interchangeable.
    """

    def search(self, query: str) -> list[WebSearchResult]:
        """Search the web for external grounding material.

        Args:
            query: The search intent from the framing brief.

        Returns:
            One ``WebSearchResult`` per surfaced source; ``[]`` when no useful
            results were found.

        Raises:
            WebSearchError: The provider call failed (timeout, API error,
                failed search).
        """
        ...


class OpenAIWebSearchClient:
    """Web-search provider backed by OpenAI's Responses API ``web_search`` tool.

    Each ``search()`` call issues one response with the ``web_search`` tool
    enabled and surfaces the model's ``url_citation`` annotations as
    ``WebSearchResult`` items — the snippet is the exact cited span of the
    answer text, which keeps external material quote-safe (ADR-0012).
    """

    def __init__(self, api_key: str | None = None, model: str | None = None) -> None:
        """Initialize the client.

        Args:
            api_key: OpenAI API key. Defaults to ``settings.openai_api_key``.
            model: Model used for the search response. Defaults to
                ``settings.web_search_model``.
        """
        self._model = model or settings.web_search_model
        self._api_key = api_key or settings.openai_api_key
        self._client: OpenAI | None = None

    def _get_client(self) -> OpenAI:
        if self._client is None:
            self._client = OpenAI(api_key=self._api_key, timeout=30.0, max_retries=0)
        return self._client

    def search(self, query: str) -> list[WebSearchResult]:
        """Search the web and return the cited external material.

        Args:
            query: The search intent from the framing brief.

        Returns:
            One ``WebSearchResult`` per ``url_citation`` annotation on the
            answer text.

        Raises:
            WebSearchError: The client could not be constructed (missing API
                key), the API was unreachable, returned a bad status, or the
                ``web_search_call`` itself failed.
        """
        try:
            response = self._get_client().responses.create(
                model=self._model,
                tools=[{"type": "web_search"}],
                input=query,
            )
        except OpenAIError as exc:
            # ``OpenAIError`` is the SDK's base failure type: it covers
            # construction-time errors (e.g. a missing API key), connection
            # failures, and bad API statuses alike. Everything converts to
            # ``WebSearchError`` so the retry wrapper can fall back gracefully
            # (ADR-0013) instead of leaking a 5xx to the route.
            raise WebSearchError(f"Web search API failed: {exc}") from exc
        return self._parse_response(response)

    @staticmethod
    def _parse_response(response: Response) -> list[WebSearchResult]:
        """Extract cited web sources from a Responses API output.

        A ``web_search_call`` item whose status is ``failed`` means the search
        itself failed even though the HTTP call succeeded. Otherwise the cited
        sources live in ``url_citation`` annotations on the answer text, with
        the snippet being the exact cited span.

        Args:
            response: The raw ``Response`` from the OpenAI SDK.

        Returns:
            One ``WebSearchResult`` per cited URL.

        Raises:
            WebSearchError: The search tool reported a failed search.
        """
        if any(
            isinstance(item, ResponseFunctionWebSearch) and item.status == "failed"
            for item in response.output
        ):
            raise WebSearchError("Web search tool reported a failed search")
        results: list[WebSearchResult] = []
        for item in response.output:
            if not isinstance(item, ResponseOutputMessage):
                continue
            for part in item.content:
                if not isinstance(part, ResponseOutputText):
                    continue
                for annotation in part.annotations:
                    if not isinstance(annotation, AnnotationURLCitation):
                        continue
                    if not (0 <= annotation.start_index <= annotation.end_index <= len(part.text)):
                        continue
                    snippet = part.text[annotation.start_index : annotation.end_index]
                    results.append(
                        WebSearchResult(
                            title=annotation.title,
                            url=annotation.url,
                            snippet=snippet,
                        )
                    )
        return results


class StubWebSearchClient:
    """Deterministic, in-memory web-search client for tests and local dev.

    Implements ``WebSearchClient`` without calling a hosted API: returns its
    configured results for every query, or raises ``WebSearchError`` for a
    configured number of calls before succeeding — letting the retry wrapper
    be exercised without any network access.
    """

    def __init__(
        self,
        *,
        results: Sequence[WebSearchResult] | None = None,
        failures: int = 0,
    ) -> None:
        """Initialize the stub.

        Args:
            results: Results to return for every query. Defaults to ``[]``.
            failures: Number of consecutive ``WebSearchError`` failures to
                raise before returning results.
        """
        self._results = list(results or [])
        self._failures = failures
        self.calls: list[str] = []

    def search(self, query: str) -> list[WebSearchResult]:
        """Return the configured results, raising while failures remain."""
        self.calls.append(query)
        if self._failures > 0:
            self._failures -= 1
            raise WebSearchError("stub search failure")
        return list(self._results)


__all__ = [
    "OpenAIWebSearchClient",
    "StubWebSearchClient",
    "WebSearchClient",
    "WebSearchError",
    "WebSearchResult",
]
