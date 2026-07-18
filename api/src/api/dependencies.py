"""FastAPI dependency providers.

Factories here let route handlers depend on protocols (``Embedder``,
``CompletionProvider``) rather than concrete clients, following the dependency
inversion principle (ADR-0011, SOLID review).
"""

from core.clients import (
    CompletionProvider,
    Embedder,
    EmbeddingsClient,
    LLMClient,
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


__all__ = ["get_completion_provider", "get_embedder"]
