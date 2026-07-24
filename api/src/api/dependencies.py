"""FastAPI dependency providers.

Factories here let route handlers depend on protocols (``Embedder``,
``CompletionProvider``, ``Reranker``) rather than concrete clients, following
the dependency inversion principle (ADR-0011, SOLID review).
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
        return CohereReranker(
            api_key=settings.cohere_api_key,
            model=settings.reranker_model,
        )
    return NoopReranker()


__all__ = ["get_completion_provider", "get_embedder", "get_reranker"]
