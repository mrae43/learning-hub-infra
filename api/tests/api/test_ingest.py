"""Integration tests for the POST /ingest route."""

import uuid
from collections.abc import Callable
from typing import cast
from unittest.mock import MagicMock
from urllib.parse import urlparse

import pytest
from fastapi.testclient import TestClient
from httpx2 import Response

from api.dependencies import get_completion_provider, get_embedder
from api.tests.conftest import _default_fake_llm_provider, set_dependency_override


def _assert_ingest_response(
    response: Response,
    expected_title: str,
    expected_doc_type: str,
    expected_filename: str,
) -> str:
    """Assert the POST /ingest response contract: 202 + Location header + body."""
    assert response.status_code == 202
    body = response.json()
    document_id = cast(str, body["document_id"])
    location = response.headers.get("location")
    assert location is not None, "Location header must be present"
    parsed = urlparse(location)
    assert parsed.path == f"/documents/{document_id}", (
        f"Location {location!r} should point at /documents/{document_id}"
    )
    assert body["status"] == "validating"
    assert body["title"] == expected_title
    assert body["document_type"] == expected_doc_type
    assert body["source_filename"] == expected_filename
    assert body["error_message"] is None
    return document_id


def test_ingest_returns_202_with_location_header(
    client: TestClient,
    sample_paper_pdf: bytes,
) -> None:
    """A valid paper upload returns 202 and a Location header."""
    response = client.post(
        "/ingest",
        files={"file": ("sample.pdf", sample_paper_pdf, "application/pdf")},
        data={"title": "Sample Paper", "document_type": "paper"},
    )
    _assert_ingest_response(response, "Sample Paper", "paper", "sample.pdf")


def test_ingest_pipeline_transitions_to_ready(
    client: TestClient,
    sample_paper_pdf: bytes,
) -> None:
    """Polling shows the document reach ready after the background task runs."""
    response = client.post(
        "/ingest",
        files={"file": ("sample.pdf", sample_paper_pdf, "application/pdf")},
        data={"title": "Sample Paper", "document_type": "paper"},
    )
    document_id = _assert_ingest_response(response, "Sample Paper", "paper", "sample.pdf")

    status = client.get(f"/documents/{document_id}").json()["status"]
    assert status == "ready"


def test_ingest_book_pipeline_transitions_to_ready(
    client: TestClient,
    sample_book_pdf: bytes,
) -> None:
    """A book upload reaches ready after the background task runs."""
    response = client.post(
        "/ingest",
        files={"file": ("sample.pdf", sample_book_pdf, "application/pdf")},
        data={"title": "Sample Book", "document_type": "book"},
    )
    document_id = _assert_ingest_response(response, "Sample Book", "book", "sample.pdf")

    status = client.get(f"/documents/{document_id}").json()["status"]
    assert status == "ready"


def test_ingest_book_epub_pipeline_transitions_to_ready(
    client: TestClient,
    sample_book_epub: bytes,
) -> None:
    """An EPUB book upload reaches ready after the background task runs."""
    response = client.post(
        "/ingest",
        files={"file": ("sample.epub", sample_book_epub, "application/epub+zip")},
        data={"title": "Sample Book", "document_type": "book"},
    )
    document_id = _assert_ingest_response(response, "Sample Book", "book", "sample.epub")

    status = client.get(f"/documents/{document_id}").json()["status"]
    assert status == "ready"


def test_ingest_documentation_pipeline_transitions_to_ready(
    client: TestClient,
    sample_documentation_md: bytes,
) -> None:
    """A Markdown documentation upload reaches ready after the background task runs."""
    response = client.post(
        "/ingest",
        files={"file": ("docs.md", sample_documentation_md, "text/markdown")},
        data={"title": "Sample Docs", "document_type": "documentation"},
    )
    document_id = _assert_ingest_response(response, "Sample Docs", "documentation", "docs.md")

    status = client.get(f"/documents/{document_id}").json()["status"]
    assert status == "ready"


def test_ingest_oversized_file_returns_413(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    sample_paper_pdf: bytes,
) -> None:
    """A file exceeding max_upload_bytes returns 413."""
    monkeypatch.setattr("api.routes.ingest.settings.max_upload_bytes", 10)
    response = client.post(
        "/ingest",
        files={"file": ("sample.pdf", sample_paper_pdf, "application/pdf")},
        data={"title": "Big Paper", "document_type": "paper"},
    )
    assert response.status_code == 413


def test_ingest_unsupported_extension_returns_415(
    client: TestClient,
    sample_paper_pdf: bytes,
) -> None:
    """An unsupported file extension returns 415."""
    response = client.post(
        "/ingest",
        files={"file": ("sample.txt", b"plain text", "text/plain")},
        data={"title": "Text Paper", "document_type": "paper"},
    )
    assert response.status_code == 415


def test_ingest_missing_title_returns_422(
    client: TestClient,
    sample_paper_pdf: bytes,
) -> None:
    """Missing form fields return 422."""
    response = client.post(
        "/ingest",
        files={"file": ("sample.pdf", sample_paper_pdf, "application/pdf")},
        data={"document_type": "paper"},
    )
    assert response.status_code == 422


def test_ingest_missing_document_type_returns_422(
    client: TestClient,
    sample_paper_pdf: bytes,
) -> None:
    """Missing document_type returns 422."""
    response = client.post(
        "/ingest",
        files={"file": ("sample.pdf", sample_paper_pdf, "application/pdf")},
        data={"title": "No Type"},
    )
    assert response.status_code == 422


def test_ingest_pipeline_failure_marks_failed(
    client: TestClient,
    sample_paper_pdf: bytes,
) -> None:
    """A mocked embeddings provider that raises surfaces as status=failed."""

    def _failing_embedder() -> MagicMock:
        embedder = MagicMock()
        embedder.embed.side_effect = RuntimeError("upstream down")
        return embedder

    set_dependency_override(client, get_embedder, _failing_embedder)

    response = client.post(
        "/ingest",
        files={"file": ("sample.pdf", sample_paper_pdf, "application/pdf")},
        data={"title": "Failing Paper", "document_type": "paper"},
    )
    document_id = _assert_ingest_response(response, "Failing Paper", "paper", "sample.pdf")

    status_response = client.get(f"/documents/{document_id}")
    assert status_response.status_code == 200
    body = status_response.json()
    assert body["status"] == "failed"
    assert body["error_message"] is not None


def test_ingest_then_query_end_to_end(
    client: TestClient,
    ingest_a_paper: Callable[[str], str],
) -> None:
    """Ingest a paper, then query it expecting grounded=true with cited passages."""
    set_dependency_override(client, get_completion_provider, _default_fake_llm_provider)

    ingest_a_paper("RAG Paper")

    response = client.post("/query", json={"query": "Tell me about retrieval strategies."})
    assert response.status_code == 200
    body = response.json()
    assert body["grounded"] is True
    assert len(body["cited_passages"]) >= 1
    for passage in body["cited_passages"]:
        uuid.UUID(passage["chunk_id"])
        assert isinstance(passage["text"], str)
        assert passage["text"]
    assert body["answer"]


def test_ingest_book_then_query_end_to_end(
    client: TestClient,
    ingest_a_book: Callable[[str], str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ingest a book, then query it expecting grounded=true with cited passages."""
    set_dependency_override(client, get_completion_provider, _default_fake_llm_provider)
    # Raise top_k so all tied fake-embedding chunks are returned.
    monkeypatch.setattr("api.routes.retrieval_qa.settings.query_top_k", 100)

    ingest_a_book("Roman History")

    response = client.post("/query", json={"query": "Tell me about Rome."})
    assert response.status_code == 200
    body = response.json()
    assert body["grounded"] is True
    assert len(body["cited_passages"]) >= 1
    assert any("Rome" in passage["text"] for passage in body["cited_passages"])
    for passage in body["cited_passages"]:
        uuid.UUID(passage["chunk_id"])
        assert isinstance(passage["text"], str)
        assert passage["text"]
    assert body["answer"]
