"""Client for the hosted OpenAI embeddings API."""

from collections.abc import Sequence

from openai import APIConnectionError, APIStatusError, AsyncOpenAI, OpenAI

from core.config.settings import settings
from core.exceptions import UpstreamBadResponse, UpstreamUnavailable


class EmbeddingsClient:
    """Thin wrapper around OpenAI's embeddings endpoint.

    The client is intentionally narrow: one model, one dimensionality. MVP
    uses ``text-embedding-3-small`` producing 1536-dim vectors (ADR-0014).
    """

    def __init__(self, api_key: str | None = None, model: str | None = None) -> None:
        """Initialize the client.

        Args:
            api_key: OpenAI API key. Defaults to ``settings.openai_api_key``.
            model: Embedding model ID. Defaults to ``settings.embedding_model``.
        """
        self._model = model or settings.embedding_model
        self._client: OpenAI | None = None
        self._async_client: AsyncOpenAI | None = None
        self._api_key = api_key or settings.openai_api_key

    def _get_client(self) -> OpenAI:
        if self._client is None:
            self._client = OpenAI(api_key=self._api_key)
        return self._client

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed a batch of texts synchronously.

        Args:
            texts: Input strings to embed.

        Returns:
            One 1536-dim vector per input string.

        Raises:
            UpstreamBadResponse: The API returned an unexpected response or a
                bad HTTP status (4xx/5xx). Maps to 502.
            UpstreamUnavailable: The API could not be reached or timed out.
                Maps to 503.
        """
        try:
            response = self._get_client().embeddings.create(
                input=list(texts),
                model=self._model,
            )
        except APIConnectionError as exc:
            raise UpstreamUnavailable(f"Embeddings API unreachable: {exc}") from exc
        except APIStatusError as exc:
            raise UpstreamBadResponse(f"Embeddings API returned bad status: {exc}") from exc

        try:
            return [item.embedding for item in response.data]
        except (AttributeError, IndexError) as exc:
            raise UpstreamBadResponse(
                f"Embeddings API returned unexpected response shape: {exc}"
            ) from exc

    async def aembed(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed a batch of texts asynchronously."""
        if self._async_client is None:
            self._async_client = AsyncOpenAI(api_key=self._api_key)
        try:
            response = await self._async_client.embeddings.create(
                input=list(texts),
                model=self._model,
            )
        except APIConnectionError as exc:
            raise UpstreamUnavailable(f"Embeddings API unreachable: {exc}") from exc
        except APIStatusError as exc:
            raise UpstreamBadResponse(f"Embeddings API returned bad status: {exc}") from exc

        try:
            return [item.embedding for item in response.data]
        except (AttributeError, IndexError) as exc:
            raise UpstreamBadResponse(
                f"Embeddings API returned unexpected response shape: {exc}"
            ) from exc
