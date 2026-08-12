"""Tests for the Depth Dive (Harness B) request/response and scene-graph models."""

from uuid import uuid4

import pytest
from pydantic import ValidationError

from core.types.captured_passage import TextPassage
from core.types.depth_dive import (
    AnimationStep,
    ElementState,
    HarnessBRequest,
    HarnessBResponse,
    InteractionHints,
    InteractiveAnimation,
    SceneElement,
    Treatment,
    Viewport,
)
from core.types.responses import CitedPassage


def _animation() -> InteractiveAnimation:
    """A minimal valid scene graph for request/response tests."""
    return InteractiveAnimation(
        output_type="interactive_animation",
        title="Self-attention",
        concept="attention",
        viewport=Viewport(width=800, height=520),
        elements=[SceneElement(id="title", type="text", x=400, y=36, text="Self-attention")],
        steps=[
            AnimationStep(
                id="setup",
                label="Show the title.",
                duration_ms=1500,
                element_states={"title": ElementState(opacity=1)},
            )
        ],
        initial_state={"title": ElementState(opacity=1)},
        interactions=InteractionHints(click_to_advance=True),
    )


def test_harness_b_request_accepts_text_passage() -> None:
    """HarnessBRequest parses a text Captured Passage and optional hints."""
    body = HarnessBRequest(
        captured_passage=TextPassage(content="Attention is all you need."),
        requested_treatments=[Treatment.WORKED_EXAMPLE],
        client_passage_id="ui-123",
    )
    assert isinstance(body.captured_passage, TextPassage)
    assert body.requested_treatments == [Treatment.WORKED_EXAMPLE]
    assert body.client_passage_id == "ui-123"
    assert body.requested_output_type is None


def test_harness_b_request_rejects_missing_passage() -> None:
    """HarnessBRequest requires a captured passage."""
    with pytest.raises(ValidationError):
        HarnessBRequest()  # type: ignore[call-arg]


def test_harness_b_request_rejects_unknown_output_type() -> None:
    """HarnessBRequest restricts requested_output_type to interactive_animation."""
    with pytest.raises(ValidationError):
        HarnessBRequest(
            captured_passage=TextPassage(content="x"),
            requested_output_type="carousel",  # type: ignore[arg-type]
        )


def test_harness_b_response_serializes_scene_graph() -> None:
    """A grounded HarnessBResponse round-trips output, treatments, and citations."""
    chunk_id = uuid4()
    response = HarnessBResponse(
        output=_animation(),
        recommended_treatments=[Treatment.WORKED_EXAMPLE],
        applied_treatments=[Treatment.WORKED_EXAMPLE],
        grounded=True,
        external_search_attempted=False,
        external_search_failed=False,
        cited_passages=[CitedPassage(chunk_id=chunk_id, text="full chunk")],
    )
    dumped = response.model_dump(mode="json")
    assert dumped["output"]["output_type"] == "interactive_animation"
    assert dumped["output"]["elements"][0]["id"] == "title"
    assert dumped["applied_treatments"] == ["worked_example"]
    assert dumped["cited_passages"][0]["chunk_id"] == str(chunk_id)
    assert dumped["grounded"] is True


def test_harness_b_response_rejects_unknown_treatment() -> None:
    """applied_treatments only accepts the three MVP treatments."""
    with pytest.raises(ValidationError):
        HarnessBResponse(
            output=_animation(),
            recommended_treatments=[Treatment.WORKED_EXAMPLE],
            applied_treatments=["voodoo"],  # type: ignore[list-item]
            grounded=False,
            external_search_attempted=False,
            external_search_failed=False,
            cited_passages=[],
        )


def test_harness_b_response_has_exact_field_set() -> None:
    """HarnessBResponse exposes exactly the spec §9 fields."""
    fields = set(HarnessBResponse.model_fields)
    assert fields == {
        "output",
        "recommended_treatments",
        "applied_treatments",
        "routing_note",
        "grounded",
        "external_search_attempted",
        "external_search_failed",
        "external_search_note",
        "cited_passages",
    }


def test_interactive_animation_requires_elements_steps_initial_state() -> None:
    """The scene graph requires non-empty elements/steps and an initial_state."""
    with pytest.raises(ValidationError):
        InteractiveAnimation(
            output_type="interactive_animation",
            title="T",
            concept="c",
            viewport=Viewport(width=800, height=520),
            elements=[],
            steps=[],
            initial_state={},
        )


def test_interactive_animation_forbids_extra_fields() -> None:
    """Undeclared scene-graph fields are rejected at the boundary."""
    with pytest.raises(ValidationError):
        InteractiveAnimation(
            output_type="interactive_animation",
            title="T",
            concept="c",
            viewport=Viewport(width=800, height=520),
            elements=[SceneElement(id="t", type="text", x=0, y=0)],
            steps=[AnimationStep(id="s", label="l", element_states={})],
            initial_state={},
            unexpected=True,  # type: ignore[call-arg]
        )


def test_element_state_is_sparse() -> None:
    """ElementState accepts only the four mutable fields."""
    state = ElementState(opacity=0.5, value=0.15)
    assert state.opacity == 0.5
    assert state.value == 0.15
    assert state.highlight is None
    assert state.text is None


def test_treatment_enum_has_three_mvp_values() -> None:
    """Treatment exposes exactly the three MVP treatments as str values."""
    assert [t.value for t in Treatment] == [
        "worked_example",
        "prediction_reveal",
        "segmented_carousel",
    ]
