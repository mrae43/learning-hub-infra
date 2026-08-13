"""FastAPI dependency providers.

Factories here let route handlers depend on protocols (``Embedder``,
``CompletionProvider``, ``Reranker``, ``WebSearchClient``) rather than concrete
clients, following the dependency inversion principle (ADR-0011, SOLID review).
"""

from core.clients import (
    CohereReranker,
    CompletionProvider,
    Embedder,
    EmbeddingsClient,
    LLMClient,
    NoopReranker,
    Reranker,
)
from core.config.settings import settings
from depth_dive.web_search.client import OpenAIWebSearchClient, WebSearchClient


def get_embedder() -> Embedder:
    """Return the configured synchronous embeddings provider."""
    return EmbeddingsClient(
        api_key=settings.openai_api_key,
        model=settings.embedding_model,
    )


def get_completion_provider() -> CompletionProvider:
    """Return the configured synchronous chat-completion provider."""
    return LLMClient(
        api_key=settings.openai_api_key,
        model=settings.inference_model,
    )


def get_reranker() -> Reranker:
    """Return the configured synchronous reranker.

    Returns ``NoopReranker`` when ``cohere_api_key`` is not configured,
    allowing the system to run without a Cohere API key during development.
    """
    if settings.cohere_api_key:
        return CohereReranker()
    return NoopReranker()


def get_web_search_client() -> WebSearchClient:
    """Return the configured web-search provider for Depth Dive."""
    return OpenAIWebSearchClient(api_key=settings.openai_api_key)


__all__ = [
    "get_completion_provider",
    "get_embedder",
    "get_reranker",
    "get_web_search_client",
]
