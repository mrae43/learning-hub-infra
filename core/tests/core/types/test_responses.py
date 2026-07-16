"""Tests for the Harness A request/response Pydantic models."""

from uuid import UUID, uuid4

from core.types.responses import CitedPassage, HarnessARequest, HarnessAResponse


def test_harness_a_request_accepts_bare_query_string() -> None:
    """HarnessARequest accepts a single non-empty query field."""
    body = HarnessARequest(query="What is pgvector?")
    assert body.query == "What is pgvector?"


def test_harness_a_request_rejects_missing_query() -> None:
    """HarnessARequest rejects a body without a query field."""
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        HarnessARequest()  # type: ignore[call-arg]


def test_harness_a_request_rejects_empty_query() -> None:
    """HarnessARequest rejects an empty-string query."""
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        HarnessARequest(query="")


def test_cited_passage_carries_chunk_id_and_full_text() -> None:
    """CitedPassage serializes chunk_id (UUID) and the full chunk text."""
    chunk_id = uuid4()
    passage = CitedPassage(chunk_id=chunk_id, text="A full chunk of prose, not a preview.")
    assert passage.chunk_id == chunk_id
    assert passage.text == "A full chunk of prose, not a preview."


def test_cited_passage_serializes_chunk_id_as_uuid_string() -> None:
    """CitedPassage serializes to JSON with chunk_id as a UUID string."""
    chunk_id = uuid4()
    passage = CitedPassage(chunk_id=chunk_id, text="text")
    dumped = passage.model_dump(mode="json")
    assert dumped["chunk_id"] == str(chunk_id)


def test_harness_a_response_grounds_with_citations() -> None:
    """A grounded response carries non-empty cited_passages and grounded=True."""
    passage = CitedPassage(chunk_id=uuid4(), text="relevant chunk text")
    response = HarnessAResponse(
        answer="Based on the corpus, ...",
        cited_passages=[passage],
        grounded=True,
    )
    assert response.answer
    assert response.grounded is True
    assert len(response.cited_passages) == 1
    assert isinstance(response.cited_passages[0].chunk_id, UUID)


def test_harness_a_response_not_grounded_has_empty_passages() -> None:
    """A not-found response keeps answer populated and cited_passages empty."""
    response = HarnessAResponse(
        answer="I could not find anything relevant in the corpus.",
        cited_passages=[],
        grounded=False,
    )
    assert response.answer
    assert response.grounded is False
    assert response.cited_passages == []


def test_harness_a_response_has_exactly_three_fields_no_observability() -> None:
    """HarnessAResponse exposes only answer, cited_passages, grounded (per ADR-0014)."""
    fields = set(HarnessAResponse.model_fields)
    assert fields == {"answer", "cited_passages", "grounded"}


def test_harness_a_response_answer_is_required_str_not_nullable() -> None:
    """HarnessAResponse.answer is always populated (str, never str | None)."""
    answer_field = HarnessAResponse.model_fields["answer"]
    assert answer_field.is_required()
    # The annotation should be plain str, not str | None.
    assert answer_field.annotation is str
