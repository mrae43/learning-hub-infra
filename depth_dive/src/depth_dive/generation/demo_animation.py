"""Builder for the hardcoded self-attention demo animation.

The scene graph mirrors the ``prototype/interactive-animation-contract``
worked example (`[river, bank, money]`): a persistent element set plus four
ordered steps that mutate ``element_states`` per element ID, seeded from
``initial_state``. Static by design — the tracer bullet returns this payload
for every valid text passage.
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
_TEXT_STYLE = ElementStyle(fontSize=13, textAnchor="middle", fill="var(--muted)")

_ACCENT = "var(--accent)"
_ACCENT_2 = "var(--accent-2)"
_OK = "var(--ok)"
_WARN = "var(--warn)"


# Match the prototype's defaults for every element the renderer interpolates.
def build_demo_animation() -> InteractiveAnimation:
    """Return the hardcoded self-attention ``interactive_animation`` payload."""
    elements = [
        SceneElement(
            id="title",
            type="text",
            x=400,
            y=36,
            text="Self-attention: bank",
            style=ElementStyle(fontSize=20, fontWeight="bold", textAnchor="middle"),
        ),
        SceneElement(id="tok-river", type="token", x=160, y=100, text="river"),
        SceneElement(id="tok-bank", type="token", x=400, y=100, text="bank"),
        SceneElement(id="tok-money", type="token", x=640, y=100, text="money"),
        SceneElement(id="q-label", type="text", x=160, y=200, text="Query (Q)", style=_TEXT_STYLE),
        SceneElement(id="k-label", type="text", x=400, y=200, text="Key (K)", style=_TEXT_STYLE),
        SceneElement(id="v-label", type="text", x=640, y=200, text="Value (V)", style=_TEXT_STYLE),
        SceneElement(id="q-bank", type="vector", x=160, y=240, label="Q_bank", color=_ACCENT),
        SceneElement(id="k-river", type="vector", x=400, y=240, label="K_river", color=_ACCENT_2),
        SceneElement(id="k-bank", type="vector", x=400, y=300, label="K_bank", color=_ACCENT_2),
        SceneElement(id="k-money", type="vector", x=400, y=360, label="K_money", color=_ACCENT_2),
        SceneElement(id="v-river", type="vector", x=640, y=240, label="V_river", color=_OK),
        SceneElement(id="v-bank", type="vector", x=640, y=300, label="V_bank", color=_OK),
        SceneElement(id="v-money", type="vector", x=640, y=360, label="V_money", color=_OK),
        SceneElement(id="score-river", type="score", x=280, y=260, value=0.15),
        SceneElement(id="score-bank", type="score", x=280, y=300, value=0.35),
        SceneElement(id="score-money", type="score", x=280, y=340, value=0.50),
        SceneElement(
            id="out-bank",
            type="vector",
            x=400,
            y=470,
            label="context(bank)",
            color=_WARN,
        ),
        SceneElement(id="caption", type="text", x=400, y=510, text="", style=_TEXT_STYLE),
    ]

    setup = AnimationStep(
        id="setup",
        label="Three input words. We want a context-aware meaning for 'bank'.",
        duration_ms=1500,
        element_states={
            "title": ElementState(opacity=1),
            "tok-river": ElementState(opacity=1),
            "tok-bank": ElementState(opacity=1, highlight=True),
            "tok-money": ElementState(opacity=1),
            "q-label": ElementState(opacity=0),
            "k-label": ElementState(opacity=0),
            "v-label": ElementState(opacity=0),
            "q-bank": ElementState(opacity=0),
            "k-river": ElementState(opacity=0),
            "k-bank": ElementState(opacity=0),
            "k-money": ElementState(opacity=0),
            "v-river": ElementState(opacity=0),
            "v-bank": ElementState(opacity=0),
            "v-money": ElementState(opacity=0),
            "score-river": ElementState(opacity=0),
            "score-bank": ElementState(opacity=0),
            "score-money": ElementState(opacity=0),
            "out-bank": ElementState(opacity=0),
            "caption": ElementState(opacity=1, text="Input: [river, bank, money]"),
        },
    )

    steps = [
        setup,
        AnimationStep(
            id="qkv",
            label="For every word, the model builds a Query, Key, and Value vector.",
            duration_ms=2000,
            element_states={
                "q-label": ElementState(opacity=1),
                "k-label": ElementState(opacity=1),
                "v-label": ElementState(opacity=1),
                "q-bank": ElementState(opacity=1),
                "k-river": ElementState(opacity=1),
                "k-bank": ElementState(opacity=1),
                "k-money": ElementState(opacity=1),
                "v-river": ElementState(opacity=1),
                "v-bank": ElementState(opacity=1),
                "v-money": ElementState(opacity=1),
                "caption": ElementState(text="Q, K, V are learned projections of each token."),
            },
        ),
        AnimationStep(
            id="scores",
            label="Query bank is compared to every Key to produce attention scores.",
            duration_ms=2500,
            element_states={
                "score-river": ElementState(opacity=1, value=0.15),
                "score-bank": ElementState(opacity=1, value=0.35),
                "score-money": ElementState(opacity=1, value=0.50),
                "caption": ElementState(text="Higher score = more relevant context for 'bank'."),
            },
        ),
        AnimationStep(
            id="weighted",
            label="Scores become weights; Values sum into the new 'bank' representation.",
            duration_ms=2500,
            element_states={
                "out-bank": ElementState(opacity=1),
                "score-river": ElementState(value=0.15),
                "score-bank": ElementState(value=0.35),
                "score-money": ElementState(value=0.50),
                "caption": ElementState(
                    text="context(bank) = 0.15*V_river + 0.35*V_bank + 0.50*V_money"
                ),
            },
        ),
    ]

    return InteractiveAnimation(
        output_type="interactive_animation",
        title="Self-attention: how a word sees its neighbors",
        concept="attention",
        viewport=_DEFAULT_VIEWPORT,
        elements=elements,
        steps=steps,
        initial_state=dict(setup.element_states.items()),
        interactions=InteractionHints(click_to_advance=True),
    )


__all__ = ["build_demo_animation"]
