"""Unit tests for the embeddings client (hosted API mocked)."""

from unittest.mock import MagicMock

import pytest

from core.clients.embeddings_client import EmbeddingsClient


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
