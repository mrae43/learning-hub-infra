"""Tests for the LLM generation turn (ADR-0020, ticket #245).

The hosted inference API is mocked (``MockCompletionProvider`` / a
``MagicMock``) per coding-standards.md; these tests cover prompt assembly,
scene-graph parsing, and the graceful fallback on malformed output.
"""

import json
from collections.abc import Sequence
from typing import Any
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from core.clients import MockCompletionProvider
from core.exceptions import UpstreamBadResponse
from core.types.captured_passage import TextPassage
from core.types.chat import ChatContentImagePart, ChatContentTextPart, ChatMessage
from core.types.depth_dive import InteractiveAnimation, Treatment
from core.types.responses import CitedPassage
from depth_dive.framing.framing_agent import run_framing
from depth_dive.generation.generation_agent import (
    MALFORMED_OUTPUT_NOTE,
    build_generation_prompt,
    parse_animation,
    run_generation,
)
from depth_dive.transform import ImageBlock, TableBlock, TextBlock
from depth_dive.web_search.client import WebSearchResult


def _valid_scene_graph() -> dict[str, Any]:
    """A scene-graph payload that satisfies the InteractiveAnimation contract."""
    return {
        "output_type": "interactive_animation",
        "title": "Self-attention",
        "concept": "attention",
        "viewport": {"width": 800, "height": 520},
        "elements": [
            {"id": "tok-bank", "type": "token", "x": 100, "y": 100, "text": "bank"},
            {"id": "label", "type": "text", "x": 100, "y": 200, "text": "query"},
        ],
        "steps": [
            {
                "id": "setup",
                "label": "A query vector attends to the keys.",
                "duration_ms": 2000,
                "element_states": {"tok-bank": {"opacity": 1}},
            }
        ],
        "initial_state": {"tok-bank": {"opacity": 0}, "label": {"opacity": 0}},
    }


def _citation(text: str = "a grounding passage") -> CitedPassage:
    return CitedPassage(chunk_id=uuid4(), text=text)


def _search_result(title: str = "Paper") -> WebSearchResult:
    return WebSearchResult(title=title, url="https://example.com/paper", snippet="short quote")


def _text_carrier(text: str = "Attention is all you need.") -> TextBlock:
    return TextBlock(text=text)


def _image_carrier() -> ImageBlock:
    return ImageBlock(media_type="image/png", base64="aGVsbG8=", width=4, height=4)


def _table_carrier() -> TableBlock:
    return TableBlock(
        headers=["l", "v"],
        rows=[["a", "1"], ["b", "2"]],
        image=ImageBlock(media_type="image/png", base64="aGVsbG8=", width=4, height=4),
    )


def _user_message(prompt: list[ChatMessage]) -> str:
    assert len(prompt) == 2
    assert prompt[0].role == "system"
    content = prompt[1].content
    if isinstance(content, str):
        return content
    return "\n".join(part.text for part in content if isinstance(part, ChatContentTextPart))


# ============================================================
# Prompt assembly (ticket #245 acceptance criterion 1)
# ============================================================


def test_prompt_includes_the_framing_brief() -> None:
    """The prompt carries the brief's concept, treatments, and search intent."""
    brief = run_framing(TextPassage(content="Attention is all you need."))
    user = _user_message(build_generation_prompt(brief, [], [], _text_carrier()))
    assert "Attention is all you need." in user
    assert "worked_example" in user
    assert "none" in user


def test_prompt_includes_cited_passages() -> None:
    """Grounding passages are included in the prompt."""
    brief = run_framing(TextPassage(content="A concept"))
    citations = [_citation("the enclosing section on attention")]
    user = _user_message(build_generation_prompt(brief, citations, [], _text_carrier()))
    assert "the enclosing section on attention" in user
    assert "1" in user


def test_prompt_notes_when_passage_is_ungrounded() -> None:
    """An ungrounded dive tells the model there are no grounding passages."""
    brief = run_framing(TextPassage(content="A concept"))
    user = _user_message(build_generation_prompt(brief, [], [], _text_carrier()))
    assert "could not be grounded" in user


def test_prompt_includes_web_search_results() -> None:
    """Web-search material is included in the prompt."""
    brief = run_framing(TextPassage(content="A concept"))
    results = [_search_result("The Illustrated Attention")]
    user = _user_message(build_generation_prompt(brief, [], results, _text_carrier()))
    assert "The Illustrated Attention" in user
    assert "short quote" in user
    assert "https://example.com/paper" in user


def test_prompt_notes_when_no_external_material() -> None:
    """A dive without search results tells the model no material was retrieved."""
    brief = run_framing(TextPassage(content="A concept"))
    user = _user_message(build_generation_prompt(brief, [], [], _text_carrier()))
    assert "no external material" in user


def test_prompt_is_treatment_specific() -> None:
    """The applied treatments carry their scaffolding guidance in the prompt."""
    brief = run_framing(
        TextPassage(content="A concept"),
        requested_treatments=[Treatment.SEGMENTED_CAROUSEL],
    )
    user = _user_message(build_generation_prompt(brief, [], [], _text_carrier()))
    assert "segmented_carousel" in user
    assert "swipeable segments" in user


def test_system_prompt_declares_the_scene_graph_contract() -> None:
    """The system message sets the JSON contract the model must satisfy."""
    brief = run_framing(TextPassage(content="A concept"))
    prompt = build_generation_prompt(brief, [], [], _text_carrier())
    assert prompt[0].role == "system"
    assert "output_type" in prompt[0].content
    assert "initial_state" in prompt[0].content
    assert "elements" in prompt[0].content


# ============================================================
# Captured-passage carrier wiring (ticket #254)
# ============================================================


def test_prompt_includes_the_captured_text_passage() -> None:
    """The text carrier's full content reaches the prompt, not just the concept."""
    brief = run_framing(TextPassage(content="Attention is all you need."))
    user = _user_message(
        build_generation_prompt(brief, [], [], _text_carrier("the full captured text"))
    )
    assert "Captured passage:" in user
    assert "the full captured text" in user


def test_prompt_includes_code_language_hint() -> None:
    """A code carrier carries its language hint alongside the content."""
    brief = run_framing(TextPassage(content="A concept"))
    carrier = TextBlock(text="def f():\n    return 1", language="python")
    user = _user_message(build_generation_prompt(brief, [], [], carrier))
    assert "def f():" in user
    assert "language: python" in user


def test_prompt_serializes_table_rows() -> None:
    """A table carrier's structured rows reach the prompt as text."""
    brief = run_framing(TextPassage(content="A concept"))
    user = _user_message(build_generation_prompt(brief, [], [], _table_carrier()))
    assert "l | v" in user
    assert "a | 1" in user
    assert "b | 2" in user


def test_prompt_attaches_image_part_for_image_carrier() -> None:
    """An image carrier travels as an image_url content part alongside the text."""
    brief = run_framing(TextPassage(content="A concept"))
    prompt = build_generation_prompt(brief, [], [], _image_carrier())
    assert prompt[1].role == "user"
    content = prompt[1].content
    assert isinstance(content, list)
    text_parts = [part for part in content if isinstance(part, ChatContentTextPart)]
    image_parts = [part for part in content if isinstance(part, ChatContentImagePart)]
    assert len(text_parts) == 1
    assert "an image is attached below" in text_parts[0].text
    assert len(image_parts) == 1
    assert image_parts[0].type == "image_url"
    assert image_parts[0].image_url.url == "data:image/png;base64,aGVsbG8="


def test_table_carrier_attaches_rendered_image() -> None:
    """A table carries both its structured rows and its rendered image."""
    brief = run_framing(TextPassage(content="A concept"))
    prompt = build_generation_prompt(brief, [], [], _table_carrier())
    content = prompt[1].content
    assert isinstance(content, list)
    text_parts = [part for part in content if isinstance(part, ChatContentTextPart)]
    image_parts = [part for part in content if isinstance(part, ChatContentImagePart)]
    assert len(text_parts) == 1
    assert "l | v" in text_parts[0].text
    assert "a | 1" in text_parts[0].text
    assert len(image_parts) == 1
    assert image_parts[0].image_url.url == "data:image/png;base64,aGVsbG8="


def test_run_generation_passes_image_carrier_to_provider() -> None:
    """The image carrier's bytes reach the provider as part of the user message."""
    provider = MagicMock(return_value=json.dumps(_valid_scene_graph()))
    brief = run_framing(TextPassage(content="A concept"))
    carrier = _image_carrier()
    run_generation(brief, [], [], carrier, completion_provider=provider)
    expected = build_generation_prompt(brief, [], [], carrier)
    provider.chat.assert_called_once_with(expected)


# ============================================================
# Parsing (ticket #245 acceptance criterion 2)
# ============================================================


def test_parse_animation_returns_valid_scene_graph() -> None:
    """A well-formed scene graph parses into an InteractiveAnimation."""
    animation = parse_animation(json.dumps(_valid_scene_graph()))
    assert animation is not None
    assert isinstance(animation, InteractiveAnimation)
    assert animation.output_type == "interactive_animation"
    assert animation.title == "Self-attention"
    assert len(animation.elements) == 2
    assert len(animation.steps) == 1
    assert animation.initial_state


def test_parse_animation_tolerates_markdown_code_fence() -> None:
    """A response wrapped in ```json fences still parses."""
    raw = f"```json\n{json.dumps(_valid_scene_graph())}\n```"
    animation = parse_animation(raw)
    assert animation is not None
    assert animation.title == "Self-attention"


def test_parse_animation_rejects_unspecified_initial_state() -> None:
    """An empty initial_state violates the contract and fails to parse."""
    payload = _valid_scene_graph()
    payload["initial_state"] = {}
    assert parse_animation(json.dumps(payload)) is None


def test_parse_animation_rejects_partial_initial_state() -> None:
    """initial_state must seed every declared element (spec §7)."""
    payload = _valid_scene_graph()
    payload["initial_state"] = {"tok-bank": {"opacity": 0}}
    assert parse_animation(json.dumps(payload)) is None


def test_parse_animation_rejects_undeclared_step_element_id() -> None:
    """A step referencing an undeclared element id fails to parse."""
    payload = _valid_scene_graph()
    payload["steps"][0]["element_states"] = {"ghost": {"opacity": 1}}
    assert parse_animation(json.dumps(payload)) is None


def test_parse_animation_returns_none_for_invalid_json() -> None:
    """Non-JSON model output fails to parse."""
    assert parse_animation("not json at all") is None
    assert parse_animation("") is None


def test_parse_animation_returns_none_for_schema_violation() -> None:
    """Valid JSON that violates the scene-graph contract fails validation."""
    assert parse_animation('{"foo": "bar"}') is None
    assert parse_animation("[]") is None


# ============================================================
# run_generation (ticket #245 acceptance criteria 2 and 3)
# ============================================================


def test_run_generation_returns_the_model_scene_graph() -> None:
    """A valid model response becomes the generated animation, no fallback."""
    provider = MockCompletionProvider(json.dumps(_valid_scene_graph()))
    brief = run_framing(TextPassage(content="Attention is all you need."))
    result = run_generation(brief, [_citation()], [], _text_carrier(), completion_provider=provider)
    assert result.fallback_note is None
    assert result.animation.title == "Self-attention"
    assert isinstance(result.animation, InteractiveAnimation)


def test_run_generation_passes_the_assembled_messages() -> None:
    """The provider receives exactly the messages the prompt builder made."""
    provider = MagicMock(return_value=json.dumps(_valid_scene_graph()))
    brief = run_framing(TextPassage(content="Attention is all you need."))
    citations = [_citation()]
    results = [_search_result()]
    run_generation(brief, citations, results, _text_carrier(), completion_provider=provider)
    expected = build_generation_prompt(brief, citations, results, _text_carrier())
    provider.chat.assert_called_once_with(expected)


def test_run_generation_falls_back_on_malformed_output() -> None:
    """Malformed model output yields the fallback scene graph plus a note."""
    provider = MockCompletionProvider("I cannot do that.")
    brief = run_framing(TextPassage(content="Attention is all you need."))
    result = run_generation(brief, [], [], _text_carrier(), completion_provider=provider)
    assert result.fallback_note == MALFORMED_OUTPUT_NOTE
    assert result.animation.output_type == "interactive_animation"
    assert any(e.text == MALFORMED_OUTPUT_NOTE for e in result.animation.elements)


def test_run_generation_falls_back_on_schema_violation() -> None:
    """JSON that fails contract validation also falls back."""
    provider = MockCompletionProvider('{"output_type": "interactive_animation"}')
    brief = run_framing(TextPassage(content="Attention is all you need."))
    result = run_generation(brief, [], [], _text_carrier(), completion_provider=provider)
    assert result.fallback_note == MALFORMED_OUTPUT_NOTE


def test_run_generation_propagates_upstream_errors() -> None:
    """An inference API failure is an upstream error, not a fallback."""

    class _FailingProvider:
        def chat(self, messages: Sequence[ChatMessage]) -> str:
            raise UpstreamBadResponse("bad status from the inference API")

    brief = run_framing(TextPassage(content="Attention is all you need."))
    with pytest.raises(UpstreamBadResponse):
        run_generation(brief, [], [], _text_carrier(), completion_provider=_FailingProvider())
