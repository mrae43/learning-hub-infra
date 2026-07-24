"""Tests for the GET /health route.

Two fixture variants are used:
- ``mock_client``: ``db_session`` is mocked (yields a ``MagicMock``), and
  ``get_engine`` is overridden per test. Runs anywhere, including CI without
  a Postgres instance.
- ``client``: a real test Postgres+pgvector session is wired through; the
  happy-path integration test uses it. Skips locally when the test database
  is unavailable.
"""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import Engine, create_engine


def test_health_returns_200_with_mock_engine(
    mock_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """GET /health returns 200 with ``{"status": "ok"}`` via a mocked engine."""
    engine = create_engine("sqlite:///:memory:")
    monkeypatch.setattr("api.routes.health.get_engine", lambda: engine)

    response = mock_client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_health_returns_503_when_get_engine_raises(
    mock_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """GET /health returns 503 when ``get_engine()`` raises an exception."""

    def _broken() -> None:
        msg = "could not connect to server: Connection refused"
        raise ConnectionRefusedError(msg)

    monkeypatch.setattr("api.routes.health.get_engine", _broken)

    response = mock_client.get("/health")

    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "unhealthy"
    assert "detail" in body
    assert body["detail"]


def test_health_returns_200_via_real_test_db(
    client: TestClient, test_engine: Engine, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Integration test: GET /health returns 200 via the real test database."""

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
