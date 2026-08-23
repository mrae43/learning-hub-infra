"""Tests for the scene-graph -> self-contained HTML render stage (ticket #306).

Stage 2 of the ADR-0022 renderer. Tests assert external behavior: the HTML is
self-contained and deterministic, the model's text is escaped data (never
markup), and the step reveal / prediction reveal / carousel transition are
encoded as CSS-driven animation. The spec §7 worked example (self-attention
for ``[river, bank, money]``) is the reference fixture.
"""

import json
import re

from core.types.depth_dive import (
    AnimationStep,
    ElementState,
    InteractionHints,
    InteractiveAnimation,
    SceneElement,
    Viewport,
)
from depth_dive.render.html_renderer import render_html

_JSON_TAG = '<script type="application/json" id="depth-dive-scene">'


def _worked_example() -> InteractiveAnimation:
    """The spec §7 worked example: self-attention for [river, bank, money].

    Faithful to the ``.scratch/depth-dive/interactive-animation-contract.html``
    worked example, minus the prototype's demo-only fields (top-level
    ``version`` and ``treatment``).
    """
    return InteractiveAnimation.model_validate(
        {
            "output_type": "interactive_animation",
            "title": "Self-attention: how a word sees its neighbors",
            "concept": "attention",
            "viewport": {"width": 800, "height": 520},
            "elements": [
                {
                    "id": "title",
                    "type": "text",
                    "x": 400,
                    "y": 36,
                    "text": "Self-attention: bank",
                    "style": {"fontSize": 20, "fontWeight": "bold", "textAnchor": "middle"},
                },
                {"id": "tok-river", "type": "token", "x": 160, "y": 100, "text": "river"},
                {"id": "tok-bank", "type": "token", "x": 400, "y": 100, "text": "bank"},
                {"id": "tok-money", "type": "token", "x": 640, "y": 100, "text": "money"},
                {
                    "id": "q-label",
                    "type": "text",
                    "x": 160,
                    "y": 200,
                    "text": "Query (Q)",
                    "style": {"fontSize": 12, "textAnchor": "middle"},
                },
                {
                    "id": "k-label",
                    "type": "text",
                    "x": 400,
                    "y": 200,
                    "text": "Key (K)",
                    "style": {"fontSize": 12, "textAnchor": "middle"},
                },
                {
                    "id": "v-label",
                    "type": "text",
                    "x": 640,
                    "y": 200,
                    "text": "Value (V)",
                    "style": {"fontSize": 12, "textAnchor": "middle"},
                },
                {
                    "id": "q-bank",
                    "type": "vector",
                    "x": 160,
                    "y": 240,
                    "label": "Q_bank",
                    "color": "var(--depth-accent)",
                },
                {
                    "id": "k-river",
                    "type": "vector",
                    "x": 400,
                    "y": 240,
                    "label": "K_river",
                    "color": "var(--depth-accent-2)",
                },
                {
                    "id": "k-bank",
                    "type": "vector",
                    "x": 400,
                    "y": 300,
                    "label": "K_bank",
                    "color": "var(--depth-accent-2)",
                },
                {
                    "id": "k-money",
                    "type": "vector",
                    "x": 400,
                    "y": 360,
                    "label": "K_money",
                    "color": "var(--depth-accent-2)",
                },
                {
                    "id": "v-river",
                    "type": "vector",
                    "x": 640,
                    "y": 240,
                    "label": "V_river",
                    "color": "var(--depth-ok)",
                },
                {
                    "id": "v-bank",
                    "type": "vector",
                    "x": 640,
                    "y": 300,
                    "label": "V_bank",
                    "color": "var(--depth-ok)",
                },
                {
                    "id": "v-money",
                    "type": "vector",
                    "x": 640,
                    "y": 360,
                    "label": "V_money",
                    "color": "var(--depth-ok)",
                },
                {"id": "score-river", "type": "score", "x": 280, "y": 260, "value": 0.15},
                {"id": "score-bank", "type": "score", "x": 280, "y": 300, "value": 0.35},
                {"id": "score-money", "type": "score", "x": 280, "y": 340, "value": 0.5},
                {
                    "id": "out-bank",
                    "type": "vector",
                    "x": 400,
                    "y": 470,
                    "label": "context(bank)",
                    "color": "var(--depth-warn)",
                },
                {
                    "id": "caption",
                    "type": "text",
                    "x": 400,
                    "y": 510,
                    "text": "",
                    "style": {"fontSize": 13, "textAnchor": "middle", "fill": "var(--depth-muted)"},
                },
            ],
            "steps": [
                {
                    "id": "setup",
                    "label": "Three input words. We want a context-aware meaning for 'bank'.",
                    "duration_ms": 1500,
                    "element_states": {
                        "title": {"opacity": 1},
                        "tok-river": {"opacity": 1},
                        "tok-bank": {"opacity": 1, "highlight": True},
                        "tok-money": {"opacity": 1},
                        "q-label": {"opacity": 0},
                        "k-label": {"opacity": 0},
                        "v-label": {"opacity": 0},
                        "q-bank": {"opacity": 0},
                        "k-river": {"opacity": 0},
                        "k-bank": {"opacity": 0},
                        "k-money": {"opacity": 0},
                        "v-river": {"opacity": 0},
                        "v-bank": {"opacity": 0},
                        "v-money": {"opacity": 0},
                        "score-river": {"opacity": 0},
                        "score-bank": {"opacity": 0},
                        "score-money": {"opacity": 0},
                        "out-bank": {"opacity": 0},
                        "caption": {"opacity": 1, "text": "Input: [river, bank, money]"},
                    },
                },
                {
                    "id": "qkv",
                    "label": "For every word, the model builds a Query, Key, and Value vector.",
                    "duration_ms": 2000,
                    "element_states": {
                        "q-label": {"opacity": 1},
                        "k-label": {"opacity": 1},
                        "v-label": {"opacity": 1},
                        "q-bank": {"opacity": 1},
                        "k-river": {"opacity": 1},
                        "k-bank": {"opacity": 1},
                        "k-money": {"opacity": 1},
                        "v-river": {"opacity": 1},
                        "v-bank": {"opacity": 1},
                        "v-money": {"opacity": 1},
                        "caption": {"text": "Q, K, V are learned projections of each token."},
                    },
                },
                {
                    "id": "scores",
                    "label": "Query bank is compared to every Key to produce attention scores.",
                    "duration_ms": 2500,
                    "element_states": {
                        "score-river": {"opacity": 1, "value": 0.15},
                        "score-bank": {"opacity": 1, "value": 0.35},
                        "score-money": {"opacity": 1, "value": 0.5},
                        "caption": {"text": "Higher score = more relevant context for 'bank'."},
                    },
                },
                {
                    "id": "weighted",
                    "label": "Scores become weights; Values are summed into the new 'bank'.",
                    "duration_ms": 2500,
                    "element_states": {
                        "out-bank": {"opacity": 1},
                        "score-river": {"value": 0.15},
                        "score-bank": {"value": 0.35},
                        "score-money": {"value": 0.5},
                        "caption": {
                            "text": "context(bank) = 0.15*V_river + 0.35*V_bank + 0.50*V_money"
                        },
                    },
                },
            ],
            "initial_state": {
                "title": {"opacity": 1},
                "tok-river": {"opacity": 0},
                "tok-bank": {"opacity": 0},
                "tok-money": {"opacity": 0},
                "q-label": {"opacity": 0},
                "k-label": {"opacity": 0},
                "v-label": {"opacity": 0},
                "q-bank": {"opacity": 0},
                "k-river": {"opacity": 0},
                "k-bank": {"opacity": 0},
                "k-money": {"opacity": 0},
                "v-river": {"opacity": 0},
                "v-bank": {"opacity": 0},
                "v-money": {"opacity": 0},
                "score-river": {"opacity": 0},
                "score-bank": {"opacity": 0},
                "score-money": {"opacity": 0},
                "out-bank": {"opacity": 0},
                "caption": {"opacity": 0, "text": ""},
            },
            "interactions": {"click_to_advance": True},
        }
    )


def _base_animation() -> InteractiveAnimation:
    """A minimal two-element, two-step scene graph with an opacity ramp."""
    return InteractiveAnimation(
        output_type="interactive_animation",
        title="Attention",
        concept="attention",
        viewport=Viewport(width=800, height=520),
        elements=[
            SceneElement(id="title", type="text", x=400, y=36, text="Attention"),
            SceneElement(id="tok-bank", type="token", x=400, y=200, text="bank"),
        ],
        steps=[
            AnimationStep(
                id="step-1",
                label="Introduce the token.",
                duration_ms=1500,
                element_states={
                    "title": ElementState(opacity=1),
                    "tok-bank": ElementState(opacity=0),
                },
            ),
            AnimationStep(
                id="step-2",
                label="Reveal the token.",
                duration_ms=1500,
                element_states={
                    "title": ElementState(opacity=1),
                    "tok-bank": ElementState(opacity=1, highlight=True),
                },
            ),
        ],
        initial_state={
            "title": ElementState(opacity=1),
            "tok-bank": ElementState(opacity=0),
        },
        interactions=InteractionHints(click_to_advance=True),
    )


def _evil_animation() -> InteractiveAnimation:
    """A scene graph whose model text tries to break out of the document."""
    return InteractiveAnimation(
        output_type="interactive_animation",
        title="<script>alert('title')</script>",
        concept="<b>concept</b>",
        viewport=Viewport(width=800, height=520),
        elements=[
            SceneElement(
                id="note",
                type="text",
                x=400,
                y=200,
                text='<script>alert("xss")</script> & <b>bold</b>',
            ),
            SceneElement(
                id="tok",
                type="token",
                x=200,
                y=100,
                text="</script><script>alert(1)</script>",
            ),
        ],
        steps=[
            AnimationStep(
                id="evil",
                label="</script><img src=x onerror=alert(1)>",
                duration_ms=1500,
                element_states={
                    "note": ElementState(opacity=1),
                    "tok": ElementState(opacity=1),
                },
            )
        ],
        initial_state={
            "note": ElementState(opacity=1),
            "tok": ElementState(opacity=1),
        },
    )


def _segmented_animation() -> InteractiveAnimation:
    """A two-segment carousel scene graph."""
    return InteractiveAnimation(
        output_type="interactive_animation",
        title="Carousel",
        concept="carousel",
        viewport=Viewport(width=800, height=520),
        elements=[SceneElement(id="tok", type="token", x=400, y=200, text="bank")],
        steps=[
            AnimationStep(
                id="seg-0",
                label="Slide one.",
                duration_ms=1500,
                segment=0,
                element_states={"tok": ElementState(opacity=1)},
            ),
            AnimationStep(
                id="seg-1",
                label="Slide two.",
                duration_ms=1500,
                segment=1,
                element_states={"tok": ElementState(opacity=1)},
            ),
        ],
        initial_state={"tok": ElementState(opacity=0)},
        interactions=InteractionHints(click_to_advance=True, segments=["seg-0", "seg-1"]),
    )


def _all_types_animation() -> InteractiveAnimation:
    """A scene graph exercising every declared element type."""
    elements = [
        SceneElement(id="t", type="text", x=100, y=100, text="hi"),
        SceneElement(id="tok", type="token", x=200, y=100, text="tok"),
        SceneElement(id="v", type="vector", x=300, y=100, label="vec"),
        SceneElement(id="s", type="score", x=400, y=100, value=0.5),
        SceneElement(id="a", type="arrow", x=500, y=100),
        SceneElement(id="g", type="group", x=600, y=100),
    ]
    return InteractiveAnimation(
        output_type="interactive_animation",
        title="Types",
        concept="types",
        viewport=Viewport(width=800, height=520),
        elements=elements,
        steps=[
            AnimationStep(
                id="one",
                label="One.",
                duration_ms=1000,
                element_states={el.id: ElementState(opacity=1) for el in elements},
            )
        ],
        initial_state={el.id: ElementState(opacity=1) for el in elements},
        interactions=InteractionHints(click_to_advance=True),
    )


def _json_block(html_text: str) -> str:
    """Extract the embedded scene-graph JSON block from rendered HTML."""
    start = html_text.index(_JSON_TAG) + len(_JSON_TAG)
    end = html_text.index("</script>", start)
    return html_text[start:end]


# ============================================================
# Self-containment
# ============================================================


def test_render_html_is_self_contained() -> None:
    """The rendered HTML carries no external http(s) references."""
    html_text = render_html(_worked_example())
    assert "http://" not in html_text
    assert "https://" not in html_text
    assert 'src="http' not in html_text
    assert 'href="http' not in html_text


def test_render_html_has_no_external_url_references() -> None:
    """Every CSS url() reference is a same-document fragment."""
    html_text = render_html(_worked_example())
    for match in re.findall(r"url\([^)]*\)", html_text):
        assert match.startswith("url(#")


def test_render_html_is_byte_for_byte_deterministic() -> None:
    """The same scene graph always yields the same HTML."""
    assert render_html(_worked_example()) == render_html(_worked_example())


# ============================================================
# Model text is escaped data, never markup
# ============================================================


def test_render_html_escapes_model_text_in_the_body() -> None:
    """Model tags appear escaped in the body, never as raw markup."""
    html_text = render_html(_evil_animation())
    assert "&lt;script&gt;" in html_text
    assert "&lt;b&gt;concept&lt;/b&gt;" in html_text
    assert "<script>alert('title')</script>" not in html_text
    assert "<script>alert('xss')</script>" not in html_text


def test_render_html_closes_script_blocks_exactly_twice() -> None:
    """A model </script> can never close a script block early."""
    html_text = render_html(_evil_animation())
    assert html_text.count("</script>") == 2


def test_render_html_escapes_model_text_in_the_embedded_json() -> None:
    """The embedded JSON escapes < > & so model text stays data."""
    block = _json_block(render_html(_evil_animation()))
    assert "</script>" not in block
    assert "\\u003cscript\\u003e" in block


def test_render_html_embeds_a_parseable_scene_graph() -> None:
    """The embedded JSON block parses back to the scene graph."""
    animation = _worked_example()
    payload = json.loads(_json_block(render_html(animation)))
    assert payload["output_type"] == "interactive_animation"
    assert payload["title"] == animation.title
    assert [el["id"] for el in payload["elements"]] == [el.id for el in animation.elements]
    assert len(payload["steps"]) == len(animation.steps)


# ============================================================
# CSS-driven animation
# ============================================================


def test_render_html_encodes_step_reveal_as_css() -> None:
    """Each step is a CSS state; element opacity transitions via CSS."""
    html_text = render_html(_base_animation())
    assert "transition: opacity 0.3s ease;" in html_text
    assert '#depth-dive-stage[data-step="0"] [data-el="tok-bank"] { opacity: 0; }' in html_text
    assert '#depth-dive-stage[data-step="1"] [data-el="tok-bank"] { opacity: 1; }' in html_text


def test_render_html_encodes_highlight_as_css() -> None:
    """A highlighted element's box is styled by a CSS rule, not JS."""
    html_text = render_html(_base_animation())
    assert (
        '#depth-dive-stage[data-step="1"] [data-el="tok-bank"] rect, '
        '#depth-dive-stage[data-step="1"] [data-el="tok-bank"] line { '
        "fill: rgba(56,189,248,0.2); stroke: var(--depth-accent); "
        "stroke-width: 2; }" in html_text
    )


def test_render_html_honors_static_element_highlight() -> None:
    """A scene element declared highlight=True is highlighted by default."""
    animation = _base_animation()
    animation.elements[1].highlight = True
    html_text = render_html(animation)
    assert (
        '#depth-dive-stage[data-step="0"] [data-el="tok-bank"] rect, '
        '#depth-dive-stage[data-step="0"] [data-el="tok-bank"] line { '
        "fill: rgba(56,189,248,0.2); stroke: var(--depth-accent); "
        "stroke-width: 2; }" in html_text
    )


def test_render_html_encodes_prediction_reveal_as_css() -> None:
    """reveal_on_last_step elements stay hidden until the final step."""
    animation = _base_animation()
    animation.interactions = InteractionHints(
        click_to_advance=True, reveal_on_last_step=["tok-bank"]
    )
    html_text = render_html(animation)
    assert '#depth-dive-stage[data-step="0"] [data-el="tok-bank"] { opacity: 0; }' in html_text
    assert '#depth-dive-stage[data-step="1"] [data-el="tok-bank"] { opacity: 1; }' in html_text
    assert "reveal_on_last_step" in html_text


def test_render_html_encodes_carousel_transition_as_css() -> None:
    """Segments trigger a CSS slide-in keyframe animation."""
    html_text = render_html(_segmented_animation())
    assert "depth-dive-slide-in" in html_text
    assert "@keyframes depth-dive-slide" in html_text
    assert 'id="depth-dive-slide"' in html_text


def test_render_html_derives_segments_from_interaction_hints() -> None:
    """InteractionHints.segments alone drives the slide structure (no step.segment)."""
    animation = _segmented_animation()
    for step in animation.steps:
        step.segment = None
    html_text = render_html(animation)
    assert 'data-segments="0 1"' in html_text
    assert 'data-segment-count="2"' in html_text
    assert "depth-dive-slide-in" in html_text


def test_render_html_counts_segments_from_step_segment_values() -> None:
    """Numeric step.segment values drive the slide structure without a list."""
    animation = _segmented_animation()
    animation.interactions.segments = []
    html_text = render_html(animation)
    assert 'data-segments="0 1"' in html_text
    assert 'data-segment-count="2"' in html_text


def test_render_html_omits_carousel_markup_without_segments() -> None:
    """A non-carousel animation carries no slide row or segment attributes."""
    html_text = render_html(_base_animation())
    assert 'id="depth-dive-slide"' not in html_text
    assert "Slide: " not in html_text
    assert "data-segments=" not in html_text


# ============================================================
# Responsive layout
# ============================================================


def test_render_html_adapts_to_screen_sizes() -> None:
    """The page carries a viewport meta tag, a media query, and a scaling SVG."""
    html_text = render_html(_worked_example())
    assert 'name="viewport"' in html_text
    assert "@media" in html_text
    assert 'viewBox="0 0 800 520"' in html_text


# ============================================================
# Scene-graph fidelity
# ============================================================


def test_render_html_reflects_the_scene_graph() -> None:
    """Element ids, step labels, and narration reach the document."""
    animation = _worked_example()
    html_text = render_html(animation)
    for element in animation.elements:
        assert f'data-el="{element.id}"' in html_text
    for step in animation.steps:
        assert step.label in _json_block(html_text)
    assert "Self-attention: how a word sees its neighbors" in html_text
    assert 'id="depth-dive-stage" data-step="0"' in html_text


def test_render_html_starts_on_the_first_step_label() -> None:
    """The static narration shows the first step's label."""
    animation = _base_animation()
    html_text = render_html(animation)
    assert "Introduce the token." in html_text


def test_render_html_renders_every_element_type() -> None:
    """Text, token, vector, score, arrow, and group all render."""
    html_text = render_html(_all_types_animation())
    assert 'data-type="text"' in html_text
    assert 'data-type="token"' in html_text
    assert 'data-type="vector"' in html_text
    assert 'data-type="score"' in html_text
    assert 'data-type="arrow"' in html_text
    assert 'data-type="group"' in html_text


def test_render_html_defaults_unpositioned_elements() -> None:
    """Elements without x/y layout hints still render deterministically."""
    animation = _base_animation()
    animation.elements[1].x = None
    animation.elements[1].y = None
    html_text = render_html(animation)
    assert 'data-el="tok-bank"' in html_text
    assert render_html(animation) == html_text
