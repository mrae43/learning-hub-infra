"""Hosted API clients and provider protocols."""

from core.clients.embeddings_client import (
    Embedder,
    EmbeddingsClient,
    InMemoryEmbedder,
)
from core.clients.llm_client import (
    CompletionProvider,
    LLMClient,
    MockCompletionProvider,
)
from core.clients.reranker_client import (
    CohereReranker,
    NoopReranker,
    Reranker,
)

__all__ = [
    "CohereReranker",
    "CompletionProvider",
    "Embedder",
    "EmbeddingsClient",
    "InMemoryEmbedder",
    "LLMClient",
    "MockCompletionProvider",
    "NoopReranker",
    "Reranker",
]
