"""Tests for the hardcoded demo animation builder."""

from depth_dive.generation.demo_animation import build_demo_animation


def test_demo_animation_is_interactive_animation() -> None:
    """The demo payload declares the interactive_animation output type."""
    animation = build_demo_animation()
    assert animation.output_type == "interactive_animation"


def test_demo_animation_has_persistent_elements() -> None:
    """The scene graph ships a non-empty element set with stable IDs."""
    animation = build_demo_animation()
    assert len(animation.elements) > 0
    ids = {element.id for element in animation.elements}
    assert len(ids) == len(animation.elements)  # no duplicate element IDs


def test_demo_animation_has_ordered_steps() -> None:
    """The scene graph ships a non-empty, labeled step sequence."""
    animation = build_demo_animation()
    assert len(animation.steps) > 0
    for step in animation.steps:
        assert step.id
        assert step.label
        assert step.duration_ms is not None
        assert step.duration_ms > 0


def test_demo_animation_has_initial_state() -> None:
    """initial_state seeds every element with a state."""
    animation = build_demo_animation()
    assert animation.initial_state
    for element in animation.elements:
        assert element.id in animation.initial_state


def test_demo_animation_steps_mutate_element_ids() -> None:
    """Every state key in a step references a declared element ID."""
    animation = build_demo_animation()
    declared_ids = {element.id for element in animation.elements}
    for step in animation.steps:
        for element_id in step.element_states:
            assert element_id in declared_ids


def test_demo_animation_has_viewport_and_treatment_agnostic_hints() -> None:
    """The payload carries a viewport and click-to-advance interaction."""
    animation = build_demo_animation()
    assert animation.viewport.width > 0
    assert animation.viewport.height > 0
    assert animation.interactions.click_to_advance is True
