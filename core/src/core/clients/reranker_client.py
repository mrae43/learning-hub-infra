"""Client for the Cohere Rerank API.

Provides a swappable ``Reranker`` protocol, a ``CohereReranker`` that calls
the Cohere Rerank API, and a ``NoopReranker`` that returns passages unchanged.
"""

from typing import Protocol, runtime_checkable

import cohere

from core.config.settings import settings
from core.exceptions import RerankerRateLimitError, UpstreamBadResponse, UpstreamUnavailable
from core.types.responses import CitedPassage


@runtime_checkable
class Reranker(Protocol):
    """Protocol for synchronous passage reranking providers.

    Consumers depend on this protocol rather than a concrete client so that
    hosted API clients, noop test doubles, and future provider implementations
    are interchangeable.
    """

    def rerank(
        self,
        query: str,
        passages: list[CitedPassage],
        top_k: int,
    ) -> list[CitedPassage]:
        """Rerank passages by relevance to the query and return the top-k.

        Args:
            query: The user's original query string.
            passages: Candidate passages to rerank.
            top_k: Maximum number of passages to return.

        Returns:
            The top_k passages in descending relevance order.
        """
        ...


class CohereReranker:
    """Reranker backed by the Cohere Rerank API.

    Calls the Cohere Rerank endpoint with the query and passage texts, then
    returns the top_k passages in reranked order.

    Rate-limit errors surface as ``RerankerRateLimitError`` so the caller
    can fall back to RRF top-k without crashing (ADR-0016).
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
    ) -> None:
        """Initialize the reranker.

        Args:
            api_key: Cohere API key. Defaults to ``settings.cohere_api_key``.
            model: Rerank model ID. Defaults to ``settings.reranker_model``.
        """
        self._model = model or settings.reranker_model
        self._api_key = api_key or settings.cohere_api_key
        self._client: cohere.ClientV2 | None = None

    def _get_client(self) -> cohere.ClientV2:
        if self._client is None:
            if self._api_key is None:
                raise UpstreamUnavailable("Cohere API key not configured (COHERE_API_KEY)")
            self._client = cohere.ClientV2(api_key=self._api_key)
        return self._client

    def rerank(
        self,
        query: str,
        passages: list[CitedPassage],
        top_k: int,
    ) -> list[CitedPassage]:
        """Rerank passages via the Cohere Rerank API and return the top-k.

        Args:
            query: The user's original query string.
            passages: Candidate passages to rerank.
            top_k: Maximum number of passages to return.

        Returns:
            The top_k passages in descending Cohere relevance order.

        Raises:
            RerankerRateLimitError: Cohere returned a 429 (rate limit).
                Caller should fall back to RRF top-k.
            UpstreamBadResponse: Cohere returned an unexpected response shape
                or a 4xx/5xx other than 429.
            UpstreamUnavailable: Cohere could not be reached or timed out.
        """
        if not passages:
            return []

        texts = [p.text for p in passages]

        try:
            response = self._get_client().rerank(
                model=self._model,
                query=query,
                documents=texts,
                top_n=min(top_k, len(passages)),
            )
        except cohere.errors.TooManyRequestsError as exc:
            raise RerankerRateLimitError(f"Cohere Rerank rate-limited (429): {exc}") from exc
        except cohere.core.api_error.ApiError as exc:
            raise UpstreamBadResponse(f"Cohere Rerank API returned bad status: {exc}") from exc
        except Exception as exc:
            raise UpstreamUnavailable(f"Cohere Rerank API unreachable: {exc}") from exc

        try:
            results = response.results
        except AttributeError as exc:
            raise UpstreamBadResponse(
                f"Cohere Rerank API returned unexpected response shape: {exc}"
            ) from exc

        reranked: list[CitedPassage] = []
        for result in results:
            idx = result.index
            if 0 <= idx < len(passages):
                reranked.append(passages[idx])

        return reranked[:top_k]


class NoopReranker:
    """Identity reranker that returns passages unchanged.

    Used when reranking is disabled or for deterministic tests.
    Implements ``Reranker``.
    """

    def rerank(
        self,
        query: str,
        passages: list[CitedPassage],
        top_k: int,
    ) -> list[CitedPassage]:
        """Return the first top_k passages unchanged.

        Args:
            query: Ignored.
            passages: Candidate passages.
            top_k: Maximum number of passages to return.

        Returns:
            The first top_k passages.
        """
        return passages[:top_k]


__all__ = ["CohereReranker", "NoopReranker", "Reranker"]
