"""Tests for the API dependency factories' OpenAI base-URL threading.

The ``load`` compose profile points ``OPENAI_BASE_URL`` at the mock upstream;
every OpenAI-backed factory must hand the setting to its client so a single
config change redirects all three providers (embeddings, generation, web
search) without touching the app code.
"""

import pytest

from api.dependencies import get_completion_provider, get_embedder, get_web_search_client
from core.clients import EmbeddingsClient, LLMClient
from core.config.settings import DEFAULT_OPENAI_BASE_URL, settings
from depth_dive.web_search.client import OpenAIWebSearchClient


def test_get_embedder_threads_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    """get_embedder constructs its client with settings.openai_base_url."""
    monkeypatch.setattr(settings, "openai_base_url", "http://mock/v1")
    client = get_embedder()
    assert isinstance(client, EmbeddingsClient)
    assert client._base_url == "http://mock/v1"


def test_get_completion_provider_threads_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    """get_completion_provider constructs its client with settings.openai_base_url."""
    monkeypatch.setattr(settings, "openai_base_url", "http://mock/v1")
    client = get_completion_provider()
    assert isinstance(client, LLMClient)
    assert client._base_url == "http://mock/v1"


def test_get_web_search_client_threads_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    """get_web_search_client constructs its client with settings.openai_base_url."""
    monkeypatch.setattr(settings, "openai_base_url", "http://mock/v1")
    client = get_web_search_client()
    assert isinstance(client, OpenAIWebSearchClient)
    assert client._base_url == "http://mock/v1"


def test_factories_default_to_sdk_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without a configured base URL, clients use the SDK's default endpoint."""
    monkeypatch.setattr(settings, "openai_base_url", None)
    embedder = get_embedder()
    completion = get_completion_provider()
    web_search = get_web_search_client()
    assert isinstance(embedder, EmbeddingsClient)
    assert isinstance(completion, LLMClient)
    assert isinstance(web_search, OpenAIWebSearchClient)
    assert embedder._base_url == DEFAULT_OPENAI_BASE_URL
    assert completion._base_url == DEFAULT_OPENAI_BASE_URL
    assert web_search._base_url == DEFAULT_OPENAI_BASE_URL
