"""Builder for the graceful-fallback scene graph (ticket #245).

When the model's output cannot be parsed into a valid scene graph, the
generation agent returns a minimal ``InteractiveAnimation`` built here: a
single narrative step carrying the fallback note as a text element, so the
client always has a renderable artifact instead of a 500.
"""

from core.types.depth_dive import (
    AnimationStep,
    ElementState,
    ElementStyle,
    InteractionHints,
    InteractiveAnimation,
    SceneElement,
    Viewport,
)

_DEFAULT_VIEWPORT = Viewport(width=800, height=520)
_TITLE_STYLE = ElementStyle(fontSize=20, fontWeight="bold", textAnchor="middle")
_NOTE_STYLE = ElementStyle(fontSize=13, textAnchor="middle", fill="var(--muted)")


def build_fallback_animation(note: str) -> InteractiveAnimation:
    """Return a minimal valid scene graph carrying a fallback note.

    Args:
        note: Human-readable note explaining why the dive fell back.

    Returns:
        An ``InteractiveAnimation`` with one persistent text element holding
        the note, a single narrative step, and a fully-seeded ``initial_state``.
    """
    title_element = SceneElement(
        id="title",
        type="text",
        x=400,
        y=120,
        text="Depth Dive",
        style=_TITLE_STYLE,
    )
    note_element = SceneElement(
        id="note",
        type="text",
        x=400,
        y=200,
        text=note,
        style=_NOTE_STYLE,
    )
    step = AnimationStep(
        id="fallback",
        label=note,
        duration_ms=1500,
        element_states={
            "title": ElementState(opacity=1),
            "note": ElementState(opacity=1),
        },
    )
    return InteractiveAnimation(
        output_type="interactive_animation",
        title="Depth Dive",
        concept="depth dive",
        viewport=_DEFAULT_VIEWPORT,
        elements=[title_element, note_element],
        steps=[step],
        initial_state=dict(step.element_states.items()),
        interactions=InteractionHints(click_to_advance=True),
    )


__all__ = ["build_fallback_animation"]
