"""Shared fixtures for api tests."""

from collections.abc import Generator
from contextlib import contextmanager
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
    # Real API calls belong only in the eval job (coding-standards.md); both
    # the ingestion task and the query route get a mocked embeddings + LLM
    # client. Tests override ``LLMClient`` / ``EmbeddingsClient`` to swap
    # behaviour (e.g. make them raise, or return a grounded answer).
    monkeypatch.setattr("ingestion.tasks.EmbeddingsClient", _default_fake_client)
    monkeypatch.setattr("api.routes.retrieval_qa.EmbeddingsClient", _default_fake_client)
    monkeypatch.setattr("api.routes.retrieval_qa.LLMClient", _default_fake_llm_client)
    from api.server import create_app

    return TestClient(create_app())


@pytest.fixture
def mock_client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """A TestClient with ``db_session`` mocked to yield a MagicMock session.

    For tests that exercise request validation or upstream-error mapping
    where the DB session is opened but not meaningfully used (the embeddings
    call raise, or retrieval is monkeypatched away). Lets the HTTP-level
    behaviour run without a real Postgres+pgvector instance.
    """
    monkeypatch.setattr("ingestion.tasks.EmbeddingsClient", _default_fake_client)
    monkeypatch.setattr("api.routes.retrieval_qa.EmbeddingsClient", _default_fake_client)
    monkeypatch.setattr("api.routes.retrieval_qa.LLMClient", _default_fake_llm_client)

    @contextmanager
    def _mock_db_session() -> Generator[MagicMock, None, None]:
        session = MagicMock()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    monkeypatch.setattr("api.routes.ingest.db_session", _mock_db_session)
    monkeypatch.setattr("api.routes.documents.db_session", _mock_db_session)
    monkeypatch.setattr("api.routes.retrieval_qa.db_session", _mock_db_session)

    from api.server import create_app

    return TestClient(create_app())


def _default_fake_client(*args: object, **kwargs: object) -> MagicMock:
    client = MagicMock()

    def _embed(texts: list[str]) -> list[list[float]]:
        return [[0.01 * (i + 1)] * 1536 for i in range(len(texts))]

    client.embed.side_effect = _embed
    return client


def _default_fake_llm_client(*args: object, **kwargs: object) -> MagicMock:
    """A mocked LLM client returning a fixed grounded/refusal answer."""
    client = MagicMock()
    client.chat.return_value = "I could not find anything relevant in the corpus."
    return client
