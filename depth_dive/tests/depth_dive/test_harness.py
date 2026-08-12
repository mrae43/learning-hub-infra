"""Tests for the Depth Dive harness entrypoint (ADR-0020)."""

import pytest
from pydantic import TypeAdapter

from core.types.captured_passage import CapturedPassage, TextPassage
from core.types.depth_dive import HarnessBResponse, InteractiveAnimation, Treatment
from depth_dive.harness import run_dive
from depth_dive.transform import PassageTransformError

_VALID_TEXT = TextPassage(content="Attention is all you need.")
_captured_passage_adapter: TypeAdapter[CapturedPassage] = TypeAdapter(CapturedPassage)


def test_run_dive_returns_hardcoded_animation() -> None:
    """A valid text passage yields a HarnessBResponse with the demo animation."""
    response = run_dive(_VALID_TEXT)
    assert isinstance(response, HarnessBResponse)
    assert isinstance(response.output, InteractiveAnimation)
    assert response.output.output_type == "interactive_animation"
    assert len(response.output.elements) > 0
    assert len(response.output.steps) > 0
    assert response.output.initial_state


def test_run_dive_response_is_not_grounded_without_retrieval() -> None:
    """No retrieval/search runs in the tracer bullet: grounded=False, search off."""
    response = run_dive(_VALID_TEXT)
    assert response.grounded is False
    assert response.external_search_attempted is False
    assert response.external_search_failed is False
    assert response.external_search_note is None
    assert response.cited_passages == []


def test_run_dive_recommends_and_applies_worked_example() -> None:
    """The demo treats the passage as a worked_example; no routing overrides."""
    response = run_dive(_VALID_TEXT)
    assert response.recommended_treatments == [Treatment.WORKED_EXAMPLE]
    assert response.applied_treatments == [Treatment.WORKED_EXAMPLE]
    assert response.routing_note is None


def test_run_dive_output_is_static_across_passages() -> None:
    """The tracer bullet returns the identical hardcoded payload for any text."""
    first = run_dive(_VALID_TEXT)
    second = run_dive(TextPassage(content="A completely different passage."))
    assert first.model_dump() == second.model_dump()


def test_run_dive_validates_before_building() -> None:
    """A passage violating text bounds is rejected, not silently built."""
    with pytest.raises(PassageTransformError):
        run_dive(TextPassage(content=""))


def test_run_dive_accepts_type_checked_union_payload() -> None:
    """The union type parses request-shaped payloads before the harness runs."""
    passage = _captured_passage_adapter.validate_python(
        {"passage_type": "text", "content": "Typed union payload", "chunk_id": None}
    )
    response = run_dive(passage)
    assert response.output.output_type == "interactive_animation"
