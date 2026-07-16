"""Client for the hosted OpenAI chat-completions inference API.

Per ADR-0001, MVP generation uses a hosted inference API (Claude/OpenAI);
OpenAI is the chosen vendor here so embeddings (ADR-0004) and inference share
one client SDK and one auth/billing account. The client is intentionally
narrow: one model, plain chat-completions shape, no streaming/tools.
"""

from collections.abc import Sequence
from typing import Any

from openai import APIConnectionError, APIStatusError, OpenAI

from core.config.settings import settings
from core.exceptions import UpstreamBadResponse, UpstreamUnavailable


class LLMClient:
    """Thin wrapper around OpenAI's chat-completions endpoint.

    The client returns the first choice's message content as a plain string —
    Harness A's contract is plain text output only (CONTEXT.md).
    """

    def __init__(self, api_key: str | None = None, model: str | None = None) -> None:
        """Initialize the client.

        Args:
            api_key: OpenAI API key. Defaults to ``settings.openai_api_key``.
            model: Chat-completion model ID. Defaults to
                ``settings.inference_model``.
        """
        self._model = model or settings.inference_model
        self._api_key = api_key or settings.openai_api_key
        self._client: OpenAI | None = None

    def _get_client(self) -> OpenAI:
        if self._client is None:
            self._client = OpenAI(api_key=self._api_key)
        return self._client

    def chat(self, messages: Sequence[dict[str, Any]]) -> str:
        """Generate a completion synchronously and return the message content.

        Args:
            messages: OpenAI chat-completions message list (role/content).

        Returns:
            The first choice's message content as a string.

        Raises:
            UpstreamBadResponse: The API returned an unexpected response shape
                or a bad HTTP status (4xx/5xx). Maps to 502.
            UpstreamUnavailable: The API could not be reached or timed out.
                Maps to 503.
        """
        try:
            completion = self._get_client().chat.completions.create(
                model=self._model,
                # The OpenAI SDK expects its own TypedDict union for messages;
                # we accept plain dicts at our boundary (ADR-0003 hand-rolled
                # pipeline) and pass them through to the SDK.
                messages=list(messages),  # type: ignore[arg-type]
            )
        except APIConnectionError as exc:
            raise UpstreamUnavailable(f"Inference API unreachable: {exc}") from exc
        except APIStatusError as exc:
            raise UpstreamBadResponse(f"Inference API returned bad status: {exc}") from exc

        try:
            content = completion.choices[0].message.content
        except (AttributeError, IndexError) as exc:
            raise UpstreamBadResponse(
                f"Inference API returned unexpected response shape: {exc}"
            ) from exc

        if content is None:
            raise UpstreamBadResponse("Inference API returned empty message content")
        return content
