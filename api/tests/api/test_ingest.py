"""Integration tests for the POST /ingest route."""

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient


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

    assert response.status_code == 202
    assert "location" in {k.lower() for k in response.headers}
    body = response.json()
    assert body["status"] == "validating"
    assert body["title"] == "Sample Paper"
    assert body["document_type"] == "paper"
    assert body["source_filename"] == "sample.pdf"
    assert body["error_message"] is None


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
    document_id = response.json()["document_id"]

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
    monkeypatch: pytest.MonkeyPatch,
    sample_paper_pdf: bytes,
) -> None:
    """A mocked embeddings client that raises surfaces as status=failed."""

    def _failing_client(*args: object, **kwargs: object) -> MagicMock:
        client = MagicMock()
        client.embed.side_effect = RuntimeError("upstream down")
        return client

    monkeypatch.setattr("ingestion.tasks.EmbeddingsClient", _failing_client)

    response = client.post(
        "/ingest",
        files={"file": ("sample.pdf", sample_paper_pdf, "application/pdf")},
        data={"title": "Failing Paper", "document_type": "paper"},
    )
    assert response.status_code == 202
    document_id = response.json()["document_id"]

    status_response = client.get(f"/documents/{document_id}")
    assert status_response.status_code == 200
    body = status_response.json()
    assert body["status"] == "failed"
    assert body["error_message"] is not None
