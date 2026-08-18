"""Unit tests for the embeddings client (hosted API mocked)."""

import asyncio
from unittest.mock import MagicMock

import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from core.clients import embeddings_client as embeddings_module
from core.clients.embeddings_client import Embedder, EmbeddingsClient, InMemoryEmbedder
from core.config.settings import DEFAULT_OPENAI_BASE_URL, settings


def test_embed_returns_one_vector_per_input(monkeypatch: pytest.MonkeyPatch) -> None:
    """embed() returns one 1536-dim vector per input string."""
    fake_response = MagicMock()
    fake_response.data = [
        MagicMock(embedding=[0.1] * 1536),
        MagicMock(embedding=[0.2] * 1536),
    ]

    fake_create = MagicMock(return_value=fake_response)
    fake_client = MagicMock()
    fake_client.embeddings.create = fake_create

    client = EmbeddingsClient(api_key="sk-test", model="text-embedding-3-small")
    monkeypatch.setattr(client, "_get_client", lambda: fake_client)

    vectors = client.embed(["hello", "world"])

    assert len(vectors) == 2
    assert len(vectors[0]) == 1536
    fake_create.assert_called_once_with(
        input=["hello", "world"],
        model="text-embedding-3-small",
    )


def test_embeddings_client_satisfies_embedder_protocol() -> None:
    """EmbeddingsClient is a structural ``Embedder`` implementation."""
    client = EmbeddingsClient(api_key="sk-test", model="text-embedding-3-small")
    assert isinstance(client, Embedder)


def test_sync_client_threads_explicit_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    """The sync OpenAI client is constructed with the caller's base_url."""
    fake_response = MagicMock()
    fake_response.data = [MagicMock(embedding=[0.1] * 1536)]
    captured: dict[str, object] = {}

    def _fake_openai(**kwargs: object) -> MagicMock:
        captured.update(kwargs)
        client = MagicMock()
        client.embeddings.create = MagicMock(return_value=fake_response)
        return client

    monkeypatch.setattr(embeddings_module, "OpenAI", _fake_openai)
    client = EmbeddingsClient(
        api_key="sk-test", model="text-embedding-3-small", base_url="http://mock/v1"
    )
    client.embed(["hello"])

    assert captured["base_url"] == "http://mock/v1"


def test_sync_client_threads_settings_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    """With no explicit base_url, settings.openai_base_url is used."""
    fake_response = MagicMock()
    fake_response.data = [MagicMock(embedding=[0.1] * 1536)]
    captured: dict[str, object] = {}

    def _fake_openai(**kwargs: object) -> MagicMock:
        captured.update(kwargs)
        client = MagicMock()
        client.embeddings.create = MagicMock(return_value=fake_response)
        return client

    monkeypatch.setattr(settings, "openai_base_url", "http://mock/v1")
    monkeypatch.setattr(embeddings_module, "OpenAI", _fake_openai)
    client = EmbeddingsClient(api_key="sk-test", model="text-embedding-3-small")
    client.embed(["hello"])

    assert captured["base_url"] == "http://mock/v1"


def test_client_ignores_empty_base_url_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """An empty OPENAI_BASE_URL env (compose ${VAR:-}) falls back to the SDK default.

    The OpenAI SDK would otherwise treat the empty env value as the endpoint
    and break every request, so the client resolves the default itself.
    """
    monkeypatch.setenv("OPENAI_BASE_URL", "")
    monkeypatch.setattr(settings, "openai_base_url", None)
    client = EmbeddingsClient(api_key="sk-test", model="text-embedding-3-small")
    assert client._base_url == DEFAULT_OPENAI_BASE_URL


def test_async_client_threads_explicit_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    """The async OpenAI client is constructed with the caller's base_url."""
    fake_response = MagicMock()
    fake_response.data = [MagicMock(embedding=[0.1] * 1536)]
    captured: dict[str, object] = {}

    async def _fake_create(*args: object, **kwargs: object) -> MagicMock:
        return fake_response

    def _fake_async_openai(**kwargs: object) -> MagicMock:
        captured.update(kwargs)
        client = MagicMock()
        client.embeddings.create = _fake_create
        return client

    monkeypatch.setattr(embeddings_module, "AsyncOpenAI", _fake_async_openai)
    client = EmbeddingsClient(
        api_key="sk-test", model="text-embedding-3-small", base_url="http://mock/v1"
    )
    vectors = asyncio.run(client.aembed(["hello"]))

    assert vectors == [[0.1] * 1536]
    assert captured["base_url"] == "http://mock/v1"


def test_embed_records_embed_stage_span(
    monkeypatch: pytest.MonkeyPatch,
    in_memory_tracing: InMemorySpanExporter,
) -> None:
    """The embed seam records an ``embed`` stage span (issue #286)."""
    fake_response = MagicMock()
    fake_response.data = [MagicMock(embedding=[0.1] * 1536)]
    fake_client = MagicMock()
    fake_client.embeddings.create = MagicMock(return_value=fake_response)

    client = EmbeddingsClient(api_key="sk-test", model="text-embedding-3-small")
    monkeypatch.setattr(client, "_get_client", lambda: fake_client)
    client.embed(["hello"])

    spans = in_memory_tracing.get_finished_spans()
    assert [s.name for s in spans] == ["embed"]


def test_in_memory_embedder_returns_one_vector_per_text() -> None:
    """InMemoryEmbedder returns one deterministic vector per input."""
    embedder = InMemoryEmbedder(dimension=8, scale=0.1)
    vectors = embedder.embed(["alpha", "beta"])

    assert len(vectors) == 2
    assert len(vectors[0]) == 8
    assert vectors[0] == [0.1] * 8
    assert vectors[1] == [0.2] * 8


def test_in_memory_embedder_satisfies_embedder_protocol() -> None:
    """InMemoryEmbedder is a structural ``Embedder`` implementation."""
    embedder = InMemoryEmbedder()
    assert isinstance(embedder, Embedder)
