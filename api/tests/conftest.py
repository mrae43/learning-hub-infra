"""Shared fixtures and utilities for api tests."""

from collections.abc import Callable, Generator
from contextlib import contextmanager
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

IngestADocument = Callable[[str], str]


def _patch_clients(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch embeddings + LLM clients with defaults for all test fixtures."""
    monkeypatch.setattr("ingestion.tasks.EmbeddingsClient", _default_fake_client)
    monkeypatch.setattr("api.routes.retrieval_qa.EmbeddingsClient", _default_fake_client)
    monkeypatch.setattr("api.routes.retrieval_qa.LLMClient", _default_fake_llm_refusal_client)


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
    _patch_clients(monkeypatch)
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
    _patch_clients(monkeypatch)

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


@pytest.fixture
def ingest_a_paper(client: TestClient, sample_paper_pdf: bytes) -> IngestADocument:
    """Fixture returning a callable that ingests a paper and awaits ready."""

    def _ingest(title: str = "Paper") -> str:
        response = client.post(
            "/ingest",
            files={"file": ("sample.pdf", sample_paper_pdf, "application/pdf")},
            data={"title": title, "document_type": "paper"},
        )
        assert response.status_code == 202
        document_id = response.json()["document_id"]
        status = client.get(f"/documents/{document_id}").json()["status"]
        assert status == "ready", f"document never reached ready: {status}"
        return str(document_id)

    return _ingest


@pytest.fixture
def ingest_a_book(client: TestClient, sample_book_pdf: bytes) -> IngestADocument:
    """Fixture returning a callable that ingests a book and awaits ready."""

    def _ingest(title: str = "Book") -> str:
        response = client.post(
            "/ingest",
            files={"file": ("sample.pdf", sample_book_pdf, "application/pdf")},
            data={"title": title, "document_type": "book"},
        )
        assert response.status_code == 202
        document_id = response.json()["document_id"]
        status = client.get(f"/documents/{document_id}").json()["status"]
        assert status == "ready", f"document never reached ready: {status}"
        return str(document_id)

    return _ingest


def _default_fake_client(*args: object, **kwargs: object) -> MagicMock:
    client = MagicMock()

    def _embed(texts: list[str]) -> list[list[float]]:
        return [[0.01 * (i + 1)] * 1536 for i in range(len(texts))]

    client.embed.side_effect = _embed
    return client


def _default_fake_llm_refusal_client(*args: object, **kwargs: object) -> MagicMock:
    """A mocked LLM client returning a fixed refusal (not-grounded) answer."""
    client = MagicMock()
    client.chat.return_value = "I could not find anything relevant in the corpus."
    return client
