"""Unit tests for the QA controller (clients + retrieval mocked)."""

from typing import cast
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from api.controllers.qa_controller import run_query
from core.exceptions import UpstreamBadResponse, UpstreamUnavailable
from core.types.responses import CitedPassage, HarnessAResponse
from core.types.retrieval_config import RetrievalConfig


def _fake_embeddings_client(vector: list[float], side_effect: object | None = None) -> MagicMock:
    """Build a MagicMock embeddings client that returns ``vector`` for one text."""
    client = MagicMock()
    if side_effect is not None:
        client.embed.side_effect = side_effect
    else:
        client.embed.return_value = [vector]
    return client


def _fake_llm_client(answer: str, side_effect: object | None = None) -> MagicMock:
    """Build a MagicMock LLM client that returns ``answer``."""
    client = MagicMock()
    if side_effect is not None:
        client.chat.side_effect = side_effect
    else:
        client.chat.return_value = answer
    return client


def _vector(value: float = 0.5) -> list[float]:
    return [value] * 1536


def test_run_query_grounds_when_retrieval_returns_chunks(monkeypatch: pytest.MonkeyPatch) -> None:
    """Relevant chunks yield grounded=True with cited_passages populated."""
    retrieved_chunks = [
        CitedPassage(chunk_id=uuid4(), text="pgvector supports HNSW indexes."),
        CitedPassage(chunk_id=uuid4(), text="Cosine distance uses the <=> operator."),
    ]

    fake_session = MagicMock()
    monkeypatch.setattr(
        "api.controllers.qa_controller.retrieve_relevant_chunks",
        lambda **kwargs: retrieved_chunks,
    )

    response = run_query(
        query="How does pgvector rank chunks?",
        session=fake_session,
        embeddings_client=_fake_embeddings_client(_vector()),
        llm_client=_fake_llm_client("Cosine distance via <=> against an HNSW index."),
        config=RetrievalConfig(model_name="text-embedding-3-small", ef_search=40, top_k=5),
    )

    assert isinstance(response, HarnessAResponse)
    assert response.grounded is True
    assert len(response.cited_passages) == 2
    for passage, original in zip(response.cited_passages, retrieved_chunks, strict=True):
        assert passage.chunk_id == original.chunk_id
        assert passage.text == original.text
    assert response.answer == "Cosine distance via <=> against an HNSW index."


def test_run_query_not_grounded_when_retrieval_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    """An empty retrieval result yields grounded=False with empty passages and a refusal answer."""
    monkeypatch.setattr(
        "api.controllers.qa_controller.retrieve_relevant_chunks",
        lambda **kwargs: [],
    )

    response = run_query(
        query="an irrelevant question",
        session=MagicMock(),
        embeddings_client=_fake_embeddings_client(_vector()),
        llm_client=_fake_llm_client("I couldn't find anything relevant in the corpus."),
        config=RetrievalConfig(model_name="text-embedding-3-small", ef_search=40, top_k=5),
    )

    assert response.grounded is False
    assert response.cited_passages == []
    assert response.answer == "I couldn't find anything relevant in the corpus."


def test_run_query_answer_field_always_populated(monkeypatch: pytest.MonkeyPatch) -> None:
    """The answer field is non-empty on both grounded and not-grounded branches."""
    monkeypatch.setattr(
        "api.controllers.qa_controller.retrieve_relevant_chunks",
        lambda **kwargs: [],
    )

    response = run_query(
        query="another irrelevant question",
        session=MagicMock(),
        embeddings_client=_fake_embeddings_client(_vector()),
        llm_client=_fake_llm_client("firm refusal text"),
        config=RetrievalConfig(model_name="text-embedding-3-small", ef_search=40, top_k=5),
    )

    assert response.answer == "firm refusal text"
    assert isinstance(response.answer, str)


def test_run_query_embeds_query_before_retrieval(monkeypatch: pytest.MonkeyPatch) -> None:
    """The embeddings client is called with the query string before retrieval runs."""
    captured: dict[str, object] = {}
    embeddings_client = _fake_embeddings_client(_vector(0.7))

    def _capture_retrieve(**kwargs: object) -> list[object]:
        captured.update(kwargs)
        return []

    monkeypatch.setattr(
        "api.controllers.qa_controller.retrieve_relevant_chunks",
        _capture_retrieve,
    )

    run_query(
        query="what is logits?",
        session=MagicMock(),
        embeddings_client=embeddings_client,
        llm_client=_fake_llm_client("answer"),
        config=RetrievalConfig(model_name="text-embedding-3-small", ef_search=42, top_k=3),
    )

    embeddings_client.embed.assert_called_once_with(["what is logits?"])
    config = cast(RetrievalConfig, captured["config"])
    assert config.model_name == "text-embedding-3-small"
    assert config.ef_search == 42
    assert config.top_k == 3
    assert captured["query_vector"] == _vector(0.7)


def test_run_query_propagates_upstream_unavailable_from_embeddings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Embeddings client unreachable surfaces UpstreamUnavailable for the route to map to 503."""
    monkeypatch.setattr(
        "api.controllers.qa_controller.retrieve_relevant_chunks",
        lambda **kwargs: [],
    )

    with pytest.raises(UpstreamUnavailable):
        run_query(
            query="x",
            session=MagicMock(),
            embeddings_client=_fake_embeddings_client(
                _vector(),
                side_effect=UpstreamUnavailable("down"),
            ),
            llm_client=_fake_llm_client("answer"),
            config=RetrievalConfig(model_name="text-embedding-3-small", ef_search=40, top_k=5),
        )


def test_run_query_propagates_upstream_bad_response_from_embeddings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Embeddings bad response surfaces UpstreamBadResponse for the route to map to 502."""
    monkeypatch.setattr(
        "api.controllers.qa_controller.retrieve_relevant_chunks",
        lambda **kwargs: [],
    )

    with pytest.raises(UpstreamBadResponse):
        run_query(
            query="x",
            session=MagicMock(),
            embeddings_client=_fake_embeddings_client(
                _vector(),
                side_effect=UpstreamBadResponse("bad"),
            ),
            llm_client=_fake_llm_client("answer"),
            config=RetrievalConfig(model_name="text-embedding-3-small", ef_search=40, top_k=5),
        )


def test_run_query_propagates_upstream_unavailable_from_inference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Inference client unreachable surfaces UpstreamUnavailable for the route to map to 503."""
    monkeypatch.setattr(
        "api.controllers.qa_controller.retrieve_relevant_chunks",
        lambda **kwargs: [MagicMock(chunk_id=uuid4(), text="chunk text")],
    )

    with pytest.raises(UpstreamUnavailable):
        run_query(
            query="x",
            session=MagicMock(),
            embeddings_client=_fake_embeddings_client(_vector()),
            llm_client=_fake_llm_client(
                "answer",
                side_effect=UpstreamUnavailable("down"),
            ),
            config=RetrievalConfig(model_name="text-embedding-3-small", ef_search=40, top_k=5),
        )


def test_run_query_propagates_upstream_bad_response_from_inference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Inference bad response surfaces UpstreamBadResponse for the route to map to 502."""
    monkeypatch.setattr(
        "api.controllers.qa_controller.retrieve_relevant_chunks",
        lambda **kwargs: [MagicMock(chunk_id=uuid4(), text="chunk text")],
    )

    with pytest.raises(UpstreamBadResponse):
        run_query(
            query="x",
            session=MagicMock(),
            embeddings_client=_fake_embeddings_client(_vector()),
            llm_client=_fake_llm_client(
                "answer",
                side_effect=UpstreamBadResponse("bad"),
            ),
            config=RetrievalConfig(model_name="text-embedding-3-small", ef_search=40, top_k=5),
        )


def test_run_query_passes_chunks_to_llm_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    """The injected-context prompt embeds referenced chunk text into the user message."""
    retrieved = [CitedPassage(chunk_id=uuid4(), text="chunk A text")]

    monkeypatch.setattr(
        "api.controllers.qa_controller.retrieve_relevant_chunks",
        lambda **kwargs: retrieved,
    )

    llm_client = _fake_llm_client("answer")
    run_query(
        query="what are chunks?",
        session=MagicMock(),
        embeddings_client=_fake_embeddings_client(_vector()),
        llm_client=llm_client,
        config=RetrievalConfig(model_name="text-embedding-3-small", ef_search=40, top_k=5),
    )

    llm_client.chat.assert_called_once()
    sent_messages = llm_client.chat.call_args.args[0]
    user_msg = next(m for m in sent_messages if m.role == "user")
    assert "chunk A text" in user_msg.content
    assert "what are chunks?" in user_msg.content


def test_run_query_no_chunks_prompt_does_not_include_passage_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When no chunks are retrieved the prompt asks for a refusal, without fake passages."""
    monkeypatch.setattr(
        "api.controllers.qa_controller.retrieve_relevant_chunks",
        lambda **kwargs: [],
    )

    llm_client = _fake_llm_client("refusal")
    run_query(
        query="topic the corpus does not cover",
        session=MagicMock(),
        embeddings_client=_fake_embeddings_client(_vector()),
        llm_client=llm_client,
        config=RetrievalConfig(model_name="text-embedding-3-small", ef_search=40, top_k=5),
    )

    llm_client.chat.assert_called_once()
    sent_messages = llm_client.chat.call_args.args[0]
    user_msg = next(m for m in sent_messages if m.role == "user")
    assert "topic the corpus does not cover" in user_msg.content
    assert "Passages:" not in user_msg.content
