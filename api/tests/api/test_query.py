"""Tests for the POST /query route.

Two fixture variants are used:
- ``mock_client``: ``db_session`` is mocked (yields a ``MagicMock``), for tests
  that exercise request validation or upstream-error mapping where the DB is
  not meaningfully used (the embeddings call raises, or retrieval is
  monkeypatched away). Runs anywhere, including CI without a Postgres instance.
- ``client``: a real test Postgres+pgvector session is wired through; the
  happy-path retrieval and citations tests use it. These tests skip locally
  when the test database is unavailable (mirroring the ``test_ingest`` suite).
"""

import uuid
from collections.abc import Callable
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from core.database.schema import Chunk
from core.exceptions import UpstreamBadResponse, UpstreamUnavailable
from core.types.responses import CitedPassage


def _default_fake_llm_client(*args: object, **kwargs: object) -> MagicMock:
    """A mocked LLM client that returns a fixed grounded answer."""
    client = MagicMock()
    client.chat.return_value = "Grounded answer derived from retrieved passages."
    return client


def _refusal_fake_llm_client(*args: object, **kwargs: object) -> MagicMock:
    """A mocked LLM client that returns a refusal answer."""
    client = MagicMock()
    client.chat.return_value = "I could not find anything relevant in the corpus."
    return client


def _fake_chunk(*, text: str = "chunk text") -> CitedPassage:
    """Build a CitedPassage chunk for test fixtures."""
    return CitedPassage(chunk_id=uuid.uuid4(), text=text)


# ============================================================
# 422 — request validation (FastAPI / Pydantic defaults)
# ============================================================


def test_query_missing_body_returns_422(mock_client: TestClient) -> None:
    """POST /query without a body returns 422 by FastAPI default."""
    response = mock_client.post("/query")
    assert response.status_code == 422


def test_query_empty_query_string_returns_422(mock_client: TestClient) -> None:
    """An empty-string query is rejected by HarnessARequest and returns 422."""
    response = mock_client.post("/query", json={"query": ""})
    assert response.status_code == 422


def test_query_non_string_query_returns_422(mock_client: TestClient) -> None:
    """A body with the wrong shape returns 422."""
    response = mock_client.post("/query", json={"not_query": "x"})
    assert response.status_code == 422


# ============================================================
# 502 / 503 — upstream error mapping
# ============================================================


def test_query_embeddings_bad_response_returns_502(
    mock_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A mocked embeddings client returning a bad response maps to 502."""

    def _raises_bad_client(*args: object, **kwargs: object) -> MagicMock:
        client = MagicMock()
        client.embed.side_effect = UpstreamBadResponse("bad upstream response")
        return client

    monkeypatch.setattr("api.routes.retrieval_qa.EmbeddingsClient", _raises_bad_client)

    response = mock_client.post("/query", json={"query": "anything"})

    assert response.status_code == 502
    body = response.json()
    assert "detail" in body
    assert isinstance(body["detail"], str)
    assert body["detail"]


def test_query_embeddings_unavailable_returns_503(
    mock_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A mocked embeddings client that's unreachable/timeout maps to 503."""

    def _raises_unavailable_client(*args: object, **kwargs: object) -> MagicMock:
        client = MagicMock()
        client.embed.side_effect = UpstreamUnavailable("timeout")
        return client

    monkeypatch.setattr("api.routes.retrieval_qa.EmbeddingsClient", _raises_unavailable_client)

    response = mock_client.post("/query", json={"query": "anything"})

    assert response.status_code == 503
    body = response.json()
    assert "detail" in body
    assert isinstance(body["detail"], str)
    assert body["detail"]


def test_query_inference_bad_response_returns_502(
    mock_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A mocked inference client returning a bad response maps to 502."""

    def _raises_bad_llm(*args: object, **kwargs: object) -> MagicMock:
        client = MagicMock()
        client.chat.side_effect = UpstreamBadResponse("bad upstream response")
        return client

    monkeypatch.setattr("api.routes.retrieval_qa.LLMClient", _raises_bad_llm)
    monkeypatch.setattr(
        "api.controllers.qa_controller.retrieve_relevant_chunks",
        lambda **kwargs: [_fake_chunk(text="some chunk")],
    )

    response = mock_client.post("/query", json={"query": "anything"})

    assert response.status_code == 502
    body = response.json()
    assert "detail" in body
    assert body["detail"]


def test_query_inference_unavailable_returns_503(
    mock_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A mocked inference client that's unreachable/timeout maps to 503."""

    def _raises_unavailable_llm(*args: object, **kwargs: object) -> MagicMock:
        client = MagicMock()
        client.chat.side_effect = UpstreamUnavailable("timeout")
        return client

    monkeypatch.setattr("api.routes.retrieval_qa.LLMClient", _raises_unavailable_llm)
    monkeypatch.setattr(
        "api.controllers.qa_controller.retrieve_relevant_chunks",
        lambda **kwargs: [_fake_chunk(text="some chunk")],
    )

    response = mock_client.post("/query", json={"query": "anything"})

    assert response.status_code == 503
    body = response.json()
    assert "detail" in body
    assert body["detail"]


# ============================================================
# 200 — empty-corpus / not-grounded branch (retrieval monkeypatched)
# ============================================================


def test_query_empty_corpus_returns_not_grounded(
    mock_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An empty retrieval result yields 200 grounded=false with empty passages."""
    monkeypatch.setattr("api.routes.retrieval_qa.LLMClient", _refusal_fake_llm_client)
    monkeypatch.setattr(
        "api.controllers.qa_controller.retrieve_relevant_chunks",
        lambda **kwargs: [],
    )

    response = mock_client.post("/query", json={"query": "What is pgvector?"})

    assert response.status_code == 200
    body = response.json()
    assert body["grounded"] is False
    assert body["cited_passages"] == []
    assert isinstance(body["answer"], str)
    assert body["answer"]  # non-empty refusal text


def test_query_response_has_exactly_three_fields(
    mock_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The 200 body exposes only answer, cited_passages, grounded (per ADR-0014)."""
    monkeypatch.setattr("api.routes.retrieval_qa.LLMClient", _refusal_fake_llm_client)
    monkeypatch.setattr(
        "api.controllers.qa_controller.retrieve_relevant_chunks",
        lambda **kwargs: [],
    )

    response = mock_client.post("/query", json={"query": "anything"})

    assert response.status_code == 200
    assert set(response.json().keys()) == {"answer", "cited_passages", "grounded"}


def test_query_grounds_with_mocked_retrieval(
    mock_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The route glue produces grounded=True and cited_passages from the retrieved chunks."""
    monkeypatch.setattr("api.routes.retrieval_qa.LLMClient", _default_fake_llm_client)
    chunk = _fake_chunk(text="relevant passage text")
    monkeypatch.setattr(
        "api.controllers.qa_controller.retrieve_relevant_chunks",
        lambda **kwargs: [chunk],
    )

    response = mock_client.post("/query", json={"query": "Tell me about retrieval strategies."})

    assert response.status_code == 200
    body = response.json()
    assert body["grounded"] is True
    assert len(body["cited_passages"]) == 1
    passage = body["cited_passages"][0]
    assert set(passage.keys()) == {"chunk_id", "text"}
    assert uuid.UUID(passage["chunk_id"]) == chunk.chunk_id
    assert passage["text"] == chunk.text


# ============================================================
# 200 — end-to-end against a real test DB (skips without Postgres)
# ============================================================


def test_query_end_to_end_grounds_with_real_chunks(
    client: TestClient,
    ingest_a_paper: Callable[[str], str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A relevant query against a ready corpus returns 200 grounded=true with citations."""
    monkeypatch.setattr("api.routes.retrieval_qa.LLMClient", _default_fake_llm_client)
    ingest_a_paper("RAG Paper")

    response = client.post("/query", json={"query": "Tell me about retrieval strategies."})

    assert response.status_code == 200
    body = response.json()
    assert body["grounded"] is True
    assert len(body["cited_passages"]) >= 1
    for passage in body["cited_passages"]:
        assert uuid.UUID(passage["chunk_id"])  # parses as a UUID
        assert isinstance(passage["text"], str)
        assert passage["text"]  # full chunk content
    assert body["answer"]


def test_query_end_to_end_irrelevant_returns_not_grounded(
    client: TestClient,
    ingest_a_paper: Callable[[str], str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Asking an irrelevant question yields 200 grounded=false with empty passages."""
    # Force the not-found branch even against a ready corpus by returning []
    # from retrieval; the LLM stub returns a refusal.
    monkeypatch.setattr("api.routes.retrieval_qa.LLMClient", _refusal_fake_llm_client)
    monkeypatch.setattr(
        "api.controllers.qa_controller.retrieve_relevant_chunks",
        lambda **kwargs: [],
    )
    ingest_a_paper("RAG Paper")

    response = client.post("/query", json={"query": "What is the capital of France?"})

    assert response.status_code == 200
    body = response.json()
    assert body["grounded"] is False
    assert body["cited_passages"] == []
    assert body["answer"]


def test_query_end_to_end_empty_corpus_returns_not_grounded(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Querying an empty corpus yields 200 grounded=false with empty passages."""
    monkeypatch.setattr("api.routes.retrieval_qa.LLMClient", _refusal_fake_llm_client)

    response = client.post("/query", json={"query": "What is pgvector?"})

    assert response.status_code == 200
    body = response.json()
    assert body["grounded"] is False
    assert body["cited_passages"] == []
    assert body["answer"]


def test_query_end_to_end_cross_document_returns_book_citation(
    client: TestClient,
    test_session: Session,
    ingest_a_paper: Callable[[str], str],
    ingest_a_book: Callable[[str], str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A query answered by the book returns a cited passage from the book."""
    monkeypatch.setattr("api.routes.retrieval_qa.LLMClient", _default_fake_llm_client)
    # Raise top_k so the tied fake embeddings return chunks from both documents.
    monkeypatch.setattr("api.routes.retrieval_qa.settings.query_top_k", 100)

    paper_id = ingest_a_paper("RAG Paper")
    book_id = ingest_a_book("Roman History")

    book_chunk_ids = {
        str(chunk.chunk_id)
        for chunk in test_session.query(Chunk).filter(Chunk.document_id == book_id).all()
    }
    paper_chunk_ids = {
        str(chunk.chunk_id)
        for chunk in test_session.query(Chunk).filter(Chunk.document_id == paper_id).all()
    }

    response = client.post("/query", json={"query": "Tell me about Rome."})

    assert response.status_code == 200
    body = response.json()
    assert body["grounded"] is True
    cited_ids = {passage["chunk_id"] for passage in body["cited_passages"]}
    assert cited_ids & book_chunk_ids
    assert not (cited_ids - paper_chunk_ids - book_chunk_ids)


def test_query_end_to_end_cross_document_returns_paper_citation(
    client: TestClient,
    test_session: Session,
    ingest_a_paper: Callable[[str], str],
    ingest_a_book: Callable[[str], str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A query answered by the paper returns a cited passage from the paper."""
    monkeypatch.setattr("api.routes.retrieval_qa.LLMClient", _default_fake_llm_client)
    # Raise top_k so the tied fake embeddings return chunks from both documents.
    monkeypatch.setattr("api.routes.retrieval_qa.settings.query_top_k", 100)

    paper_id = ingest_a_paper("RAG Paper")
    book_id = ingest_a_book("Roman History")

    book_chunk_ids = {
        str(chunk.chunk_id)
        for chunk in test_session.query(Chunk).filter(Chunk.document_id == book_id).all()
    }
    paper_chunk_ids = {
        str(chunk.chunk_id)
        for chunk in test_session.query(Chunk).filter(Chunk.document_id == paper_id).all()
    }

    response = client.post("/query", json={"query": "Tell me about retrieval strategies."})

    assert response.status_code == 200
    body = response.json()
    assert body["grounded"] is True
    cited_ids = {passage["chunk_id"] for passage in body["cited_passages"]}
    assert cited_ids & paper_chunk_ids
    assert not (cited_ids - paper_chunk_ids - book_chunk_ids)


def test_query_end_to_end_cross_document_returns_documentation_citation(
    client: TestClient,
    test_session: Session,
    ingest_a_paper: Callable[[str], str],
    ingest_a_documentation: Callable[[str], str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A query answered by documentation returns a cited passage from the docs."""
    monkeypatch.setattr("api.routes.retrieval_qa.LLMClient", _default_fake_llm_client)
    # Raise top_k so the tied fake embeddings return chunks from both documents.
    monkeypatch.setattr("api.routes.retrieval_qa.settings.query_top_k", 100)

    paper_id = ingest_a_paper("RAG Paper")
    docs_id = ingest_a_documentation("API Docs")

    docs_chunk_ids = {
        str(chunk.chunk_id)
        for chunk in test_session.query(Chunk).filter(Chunk.document_id == docs_id).all()
    }
    paper_chunk_ids = {
        str(chunk.chunk_id)
        for chunk in test_session.query(Chunk).filter(Chunk.document_id == paper_id).all()
    }

    response = client.post("/query", json={"query": "Tell me about users."})

    assert response.status_code == 200
    body = response.json()
    assert body["grounded"] is True
    cited_ids = {passage["chunk_id"] for passage in body["cited_passages"]}
    assert cited_ids & docs_chunk_ids
    assert not (cited_ids - paper_chunk_ids - docs_chunk_ids)
