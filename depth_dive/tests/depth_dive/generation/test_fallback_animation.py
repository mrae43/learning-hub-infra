"""Tests for the graceful-fallback scene graph builder (ticket #245)."""

from depth_dive.generation.fallback_animation import build_fallback_animation


def test_fallback_animation_is_valid_interactive_animation() -> None:
    """The fallback declares the interactive_animation output type."""
    animation = build_fallback_animation("a note")
    assert animation.output_type == "interactive_animation"
    assert animation.title
    assert animation.concept
    assert animation.viewport.width > 0
    assert animation.viewport.height > 0


def test_fallback_animation_carries_the_note_as_a_caption() -> None:
    """The note surfaces as a text element so the client renders it."""
    animation = build_fallback_animation("model output was malformed")
    note_elements = [e for e in animation.elements if e.type == "text"]
    assert any(e.text == "model output was malformed" for e in note_elements)


def test_fallback_animation_has_a_single_narrative_step() -> None:
    """The fallback ships exactly one labeled step."""
    animation = build_fallback_animation("a note")
    assert len(animation.steps) == 1
    step = animation.steps[0]
    assert step.id
    assert step.label
    assert step.duration_ms is not None
    assert step.duration_ms > 0


def test_fallback_animation_has_fully_seeded_initial_state() -> None:
    """initial_state seeds every declared element."""
    animation = build_fallback_animation("a note")
    assert animation.initial_state
    for element in animation.elements:
        assert element.id in animation.initial_state


def test_fallback_animation_steps_mutate_declared_elements() -> None:
    """Every state key in a step references a declared element ID."""
    animation = build_fallback_animation("a note")
    declared_ids = {element.id for element in animation.elements}
    for step in animation.steps:
        for element_id in step.element_states:
            assert element_id in declared_ids


def test_fallback_animation_element_ids_are_unique() -> None:
    """No two elements share an ID."""
    animation = build_fallback_animation("a note")
    ids = [element.id for element in animation.elements]
    assert len(ids) == len(set(ids))
