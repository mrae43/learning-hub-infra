"""Shared fixtures and utilities for api tests.

Test fixtures:
* ``patched_empty_retrieval`` — patches ``retrieve_relevant_chunks`` to return ``[]``.
* ``patched_retrieve_chunks`` — factory that patches retrieval with caller-supplied chunks.
"""

from collections.abc import Callable, Generator, Sequence
from contextlib import contextmanager
from typing import Any, cast
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.dependencies import get_completion_provider, get_embedder
from api.server import create_app
from core.clients import InMemoryEmbedder, MockCompletionProvider

IngestADocument = Callable[[str], str]


def _default_fake_embedder() -> InMemoryEmbedder:
    """A deterministic embedder that returns fixed 1536-dim vectors."""
    return InMemoryEmbedder(dimension=1536, scale=0.01)


def _default_fake_llm_provider() -> MockCompletionProvider:
    """A mocked completion provider that returns a fixed grounded answer."""
    return MockCompletionProvider("Grounded answer derived from retrieved passages.")


def _default_fake_llm_refusal_provider() -> MockCompletionProvider:
    """A mocked completion provider returning a fixed refusal answer."""
    return MockCompletionProvider("I could not find anything relevant in the corpus.")


@pytest.fixture
def patched_empty_retrieval(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch ``retrieve_relevant_chunks`` to return an empty list (not-grounded)."""
    monkeypatch.setattr(
        "api.controllers.qa_controller.retrieve_relevant_chunks",
        lambda **kwargs: [],
    )


@pytest.fixture
def patched_retrieve_chunks(
    monkeypatch: pytest.MonkeyPatch,
) -> Callable[[Sequence[object]], None]:
    """Return a factory that patches retrieval with the given chunks.

    Usage in a test::

        def test_foo(patched_retrieve_chunks: ...) -> None:
            patched_retrieve_chunks([CitedPassage(chunk_id=uuid4(), text="...")])
    """

    def _factory(chunks: Sequence[object]) -> None:
        monkeypatch.setattr(
            "api.controllers.qa_controller.retrieve_relevant_chunks",
            lambda **kwargs: chunks,
        )

    return _factory


def set_dependency_override(client: TestClient, dependency: Any, override: Any) -> None:
    """Set a FastAPI dependency override on ``client.app``.

    ``TestClient.app`` is typed as a generic ASGI callable, so this helper
    casts it to ``FastAPI`` before touching ``dependency_overrides``.
    """
    cast(FastAPI, client.app).dependency_overrides[dependency] = override


@pytest.fixture
def client(override_route_db_session: object, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """A TestClient with DB sessions and provider dependencies mocked.

    Real API calls belong only in the eval job (coding-standards.md). Both
    the ingestion task and the query route get mocked ``Embedder`` and
    ``CompletionProvider`` dependencies via FastAPI dependency overrides.
    """
    app = create_app()
    app.dependency_overrides[get_embedder] = _default_fake_embedder
    app.dependency_overrides[get_completion_provider] = _default_fake_llm_refusal_provider
    return TestClient(app)


@pytest.fixture
def mock_client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """A TestClient with ``db_session`` mocked to yield a MagicMock session.

    For tests that exercise request validation or upstream-error mapping
    where the DB session is opened but not meaningfully used (the embeddings
    call raises, or retrieval is monkeypatched away). Lets the HTTP-level
    behaviour run without a real Postgres+pgvector instance.
    """

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

    app = create_app()
    app.dependency_overrides[get_embedder] = _default_fake_embedder
    app.dependency_overrides[get_completion_provider] = _default_fake_llm_refusal_provider
    return TestClient(app)


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


@pytest.fixture
def ingest_a_documentation(client: TestClient, sample_documentation_md: bytes) -> IngestADocument:
    """Fixture returning a callable that ingests documentation and awaits ready."""

    def _ingest(title: str = "Docs") -> str:
        response = client.post(
            "/ingest",
            files={"file": ("docs.md", sample_documentation_md, "text/markdown")},
            data={"title": title, "document_type": "documentation"},
        )
        assert response.status_code == 202
        document_id = response.json()["document_id"]
        status = client.get(f"/documents/{document_id}").json()["status"]
        assert status == "ready", f"document never reached ready: {status}"
        return str(document_id)

    return _ingest
