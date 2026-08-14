"""LLM-driven interactive-animation generation (ADR-0020, ticket #245).

The assembly agent's final turn: builds the generation prompt from the framing
brief, the model-ready carrier chosen by the Passage Transform, the cited
corpus passages, and any web-search results, calls the hosted inference API,
and parses the response into a validated ``InteractiveAnimation`` scene graph.
Malformed model output does not fail the dive — it falls back to a minimal
valid scene graph carrying a note
(:mod:`depth_dive.generation.fallback_animation`).
"""

import json
import re
from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, ValidationError

from core.clients import CompletionProvider
from core.types.chat import (
    ChatContentImagePart,
    ChatContentImageURL,
    ChatContentTextPart,
    ChatMessage,
)
from core.types.depth_dive import InteractiveAnimation, Treatment
from core.types.responses import CitedPassage
from depth_dive.framing.framing_agent import FramingBrief
from depth_dive.generation.fallback_animation import build_fallback_animation
from depth_dive.transform import ImageBlock, ModelReadyBlock, TableBlock, TextBlock
from depth_dive.web_search.client import WebSearchResult

# User-facing note carried by the fallback scene graph (ticket #245).
MALFORMED_OUTPUT_NOTE = (
    "The model could not produce a valid scene graph for this passage, so "
    "this minimal animation stands in."
)

SYSTEM_PROMPT = (
    "You are the assembly agent of a dual-coding explainer. Given a creative "
    "brief, optional grounding passages from the learner's ingested corpus, "
    "and optional web-search results, you produce one declarative "
    "interactive-animation scene graph that explains the concept by pairing "
    "short explanatory text with a visual.\n"
    "\n"
    "Respond with a single JSON object and nothing else (no prose, no code "
    "fences). The object must match this contract:\n"
    "{\n"
    '  "output_type": "interactive_animation",\n'
    '  "title": "short headline",\n'
    '  "concept": "the concept being animated",\n'
    '  "viewport": {"width": 800, "height": 520},\n'
    '  "elements": [\n'
    "    {\n"
    '      "id": "stable unique id",\n'
    '      "type": "text | token | vector | score | arrow | group",\n'
    '      "x": 0,\n'
    '      "y": 0,\n'
    '      "text": "label for text/token elements",\n'
    '      "label": "vector label",\n'
    '      "color": "vector color"\n'
    "    }\n"
    "  ],\n"
    '  "steps": [\n'
    "    {\n"
    '      "id": "step-1",\n'
    '      "label": "narration for this step",\n'
    '      "duration_ms": 2000,\n'
    '      "element_states": {"element_id": {"opacity": 1}}\n'
    "    }\n"
    "  ],\n"
    '  "initial_state": {"element_id": {"opacity": 0}}\n'
    "}\n"
    "\n"
    "Rules:\n"
    "- Every element id referenced in a step's element_states must be "
    "declared in elements, and initial_state must seed every declared element.\n"
    "- Ground the explanation in the captured passage content, the grounding "
    "passages, and web-search results when they are present; never invent "
    "facts that contradict them.\n"
    "- Concept-centric primitives over decorative shapes; keep narration "
    "short and concrete.\n"
)

_TREATMENT_INSTRUCTIONS: dict[Treatment, str] = {
    Treatment.WORKED_EXAMPLE: (
        "worked_example: a narration-driven walkthrough that reveals each next "
        "step only as the learner advances."
    ),
    Treatment.PREDICTION_REVEAL: (
        "prediction_reveal: ask the learner to predict the next step or the "
        "output before revealing it."
    ),
    Treatment.SEGMENTED_CAROUSEL: (
        "segmented_carousel: split the concept into a small number of swipeable segments."
    ),
}
# Per-treatment scaffolding instructions (depth-dive spec §6).


class GenerationResult(BaseModel):
    """The outcome of the LLM generation turn for one dive.

    ``animation`` is always a valid ``InteractiveAnimation``: either the model's
    parsed scene graph or, when the model output was malformed, the minimal
    fallback scene graph. ``fallback_note`` is set exactly when the fallback
    was used.
    """

    model_config = ConfigDict(extra="forbid")

    animation: InteractiveAnimation
    fallback_note: str | None = None


def build_generation_prompt(
    brief: FramingBrief,
    cited_passages: Sequence[CitedPassage],
    search_results: Sequence[WebSearchResult],
    carrier: ModelReadyBlock,
) -> list[ChatMessage]:
    """Assemble the chat-completions messages for the generation turn.

    Args:
        brief: The framing agent's creative brief for this dive.
        cited_passages: Corpus passages that grounded the dive; empty when the
            passage could not be verified against the ingested corpus.
        search_results: External material surfaced by web search; empty when no
            search ran or none returned results.
        carrier: The model-ready carrier chosen by the Passage Transform for
            the captured passage. Its content is carried into the prompt: text
            carriers render as text, image carriers attach as an image content
            part so the bytes reach the model, and table carriers attach both
            their structured rows (text) and their rendered image.

    Returns:
        A two-message list: a system message setting the scene-graph contract,
        and a user message carrying the captured passage, the brief, the
        grounding passages (or the unverified notice), and the web-search
        results (or the no-material notice). When the carrier carries image
        content (an image/diagram, or a table's render) the user message's
        content is a list of content parts (text plus the image); otherwise it
        is a plain string.
    """
    image_part = _carrier_image_part(carrier)
    user_text = _build_user_text(
        brief,
        cited_passages,
        search_results,
        carrier,
        has_image=image_part is not None,
    )
    if image_part is None:
        return [
            ChatMessage(role="system", content=SYSTEM_PROMPT),
            ChatMessage(role="user", content=user_text),
        ]
    return [
        ChatMessage(role="system", content=SYSTEM_PROMPT),
        ChatMessage(
            role="user",
            content=[ChatContentTextPart(text=user_text), image_part],
        ),
    ]


def _build_user_text(
    brief: FramingBrief,
    cited_passages: Sequence[CitedPassage],
    search_results: Sequence[WebSearchResult],
    carrier: ModelReadyBlock,
    *,
    has_image: bool,
) -> str:
    """Render the user-message text: captured passage, brief, grounding, search."""
    user_parts: list[str] = ["Captured passage:"]
    carrier_text = _carrier_text(carrier)
    if carrier_text is not None:
        user_parts.append(carrier_text)
        if has_image:
            user_parts.append("(a rendered image of the same passage is attached below)")
    else:
        user_parts.append("an image is attached below as the captured passage.")
    user_parts.extend(
        [
            "\nCreative brief:",
            f"- Concept: {brief.concept}",
            f"- Applied treatments: {_treatment_guidance(brief.applied_treatments)}",
            f"- Search intent: {brief.search_intent or 'none'}",
            f"- Routing note: {brief.routing_note or 'none'}",
        ]
    )
    user_parts.append("\nGrounding passages from the learner's corpus:")
    if cited_passages:
        user_parts.extend(
            f"- [{i}] {passage.text}" for i, passage in enumerate(cited_passages, start=1)
        )
    else:
        user_parts.append("- none: the passage could not be grounded in the ingested corpus")
    user_parts.append("\nWeb-search results:")
    if search_results:
        user_parts.extend(
            f"- [{i}] {result.title}: {result.snippet} ({result.url})"
            for i, result in enumerate(search_results, start=1)
        )
    else:
        user_parts.append("- none: no external material was retrieved")
    return "\n".join(user_parts)


def _carrier_text(carrier: ModelReadyBlock) -> str | None:
    """Render the carrier's text-visible content for the prompt.

    Text/code carriers render their content (a code carrier appends its
    language hint); table carriers render their structured rows. Image
    carriers have no text-visible form, so they return ``None``.
    """
    if isinstance(carrier, TextBlock):
        if carrier.language:
            return f"{carrier.text}\n(language: {carrier.language})"
        return carrier.text
    if isinstance(carrier, TableBlock):
        return _render_table_rows(carrier)
    return None


def _render_table_rows(carrier: TableBlock) -> str:
    """Serialize a table carrier's structured form for the prompt.

    Headers (when present) lead, then one line per row; cells joined with
    ``" | "`` and dict rows rendered as ``key: value`` cells. The row/header
    serialization is consistent with the assembly agent's embedding form
    (``key: value`` for dict cells), minus the passage caption, which the
    ``TableBlock`` carrier does not carry.
    """
    lines: list[str] = []
    if carrier.headers:
        lines.append(" | ".join(carrier.headers))
    for row in carrier.rows:
        if isinstance(row, dict):
            lines.append(" | ".join(f"{key}: {value}" for key, value in row.items()))
        else:
            lines.append(" | ".join(row))
    return "\n".join(lines)


def _carrier_image_part(carrier: ModelReadyBlock) -> ChatContentImagePart | None:
    """Return the carrier's image content part, or ``None`` for a text carrier.

    Image/diagram carriers travel as an ``image_url`` part holding a base64
    data URI so the bytes reach the model (OpenAI vision shape); a table
    carrier attaches its rendered image the same way alongside its structured
    rows (spec §5's "structured text + rendered image").
    """
    if isinstance(carrier, ImageBlock):
        return _image_part(carrier)
    if isinstance(carrier, TableBlock):
        return _image_part(carrier.image)
    return None


def _image_part(block: ImageBlock) -> ChatContentImagePart:
    """Build an ``image_url`` content part from a base64 image block."""
    return ChatContentImagePart(
        image_url=ChatContentImageURL(url=f"data:{block.media_type};base64,{block.base64}")
    )


def _treatment_guidance(treatments: Sequence[Treatment]) -> str:
    """Render the applied treatments plus their scaffolding guidance."""
    if not treatments:
        return "none"
    return "; ".join(_TREATMENT_INSTRUCTIONS.get(t, t.value) for t in treatments)


def parse_animation(raw: str) -> InteractiveAnimation | None:
    """Parse and validate a model response into a scene graph.

    Tolerates a raw JSON object or one wrapped in markdown code fences.
    Returns ``None`` when the payload is not valid JSON, does not satisfy the
    ``InteractiveAnimation`` contract, or violates its referential integrity:
    every element id referenced by a step must be declared, and
    ``initial_state`` must seed every declared element (spec §7).

    Args:
        raw: The model's message content.

    Returns:
        The validated ``InteractiveAnimation``, or ``None`` for malformed
        output.
    """
    try:
        data = _extract_json(raw)
        animation = InteractiveAnimation.model_validate(data)
    except (json.JSONDecodeError, ValueError, TypeError, ValidationError):
        return None
    if not _is_referentially_coherent(animation):
        return None
    return animation


def _is_referentially_coherent(animation: InteractiveAnimation) -> bool:
    """Return whether a scene graph's state references stay within its elements.

    ``initial_state`` must seed every declared element, and every element id
    named in a step's ``element_states`` must be declared (spec §7).
    """
    declared_ids = {element.id for element in animation.elements}
    if not set(animation.initial_state) >= declared_ids:
        return False
    return all(set(step.element_states) <= declared_ids for step in animation.steps)


def _extract_json(raw: str) -> object:
    """Parse the JSON object in ``raw``, tolerating markdown code fences.

    Raises:
        json.JSONDecodeError: The response contains no parseable JSON object.
    """
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match is None:
            raise
        return json.loads(match.group(0))


def run_generation(
    brief: FramingBrief,
    cited_passages: Sequence[CitedPassage],
    search_results: Sequence[WebSearchResult],
    carrier: ModelReadyBlock,
    *,
    completion_provider: CompletionProvider,
) -> GenerationResult:
    """Run the LLM generation turn and return a validated scene graph.

    Args:
        brief: The framing agent's creative brief for this dive.
        cited_passages: Corpus passages that grounded the dive (empty when
            unverified).
        search_results: External material surfaced by web search (empty when
            no search ran or none returned results).
        carrier: The model-ready carrier chosen by the Passage Transform; its
            content (text, structured rows, or image bytes) reaches the model
            in the user message.
        completion_provider: The chat-completions provider to call.

    Returns:
        A ``GenerationResult`` carrying the parsed scene graph, or the minimal
        fallback scene graph (with ``fallback_note``) when the model output
        was malformed.

    Raises:
        UpstreamBadResponse: The inference API returned an unexpected response
            (route maps to 502).
        UpstreamUnavailable: The inference API was unreachable (route maps to
            503).
    """
    messages = build_generation_prompt(brief, cited_passages, search_results, carrier)
    raw = completion_provider.chat(messages)
    animation = parse_animation(raw)
    if animation is None:
        return GenerationResult(
            animation=build_fallback_animation(MALFORMED_OUTPUT_NOTE),
            fallback_note=MALFORMED_OUTPUT_NOTE,
        )
    return GenerationResult(animation=animation)


__all__ = [
    "MALFORMED_OUTPUT_NOTE",
    "SYSTEM_PROMPT",
    "GenerationResult",
    "build_generation_prompt",
    "parse_animation",
    "run_generation",
]
