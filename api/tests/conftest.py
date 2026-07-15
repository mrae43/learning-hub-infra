"""Shared fixtures for api tests."""

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(override_route_db_session: object, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """A TestClient with DB sessions and embeddings client mocked.

    Override ``monkeypatch`` again in a test to swap the embeddings client
    behaviour (e.g. make it raise).
    """
    # Default: a mocked embeddings client that returns fixed 1536-dim vectors.
    monkeypatch.setattr("ingestion.tasks.EmbeddingsClient", _default_fake_client)
    from api.server import create_app

    return TestClient(create_app())


def _default_fake_client(*args: object, **kwargs: object) -> MagicMock:
    client = MagicMock()

    def _embed(texts: list[str]) -> list[list[float]]:
        return [[0.01 * (i + 1)] * 1536 for i in range(len(texts))]

    client.embed.side_effect = _embed
    return client
