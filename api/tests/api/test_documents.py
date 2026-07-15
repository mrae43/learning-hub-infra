"""Integration tests for the GET /documents/{document_id} route."""

import uuid

from fastapi.testclient import TestClient


def test_get_document_returns_404_for_unknown_id(
    client: TestClient,
) -> None:
    """An unknown document id returns 404."""
    response = client.get(f"/documents/{uuid.uuid4()}")
    assert response.status_code == 404


def test_get_document_returns_422_for_malformed_uuid(client: TestClient) -> None:
    """A malformed UUID returns 422."""
    response = client.get("/documents/not-a-uuid")
    assert response.status_code == 422
