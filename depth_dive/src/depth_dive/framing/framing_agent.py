"""Framing agent implementation (ADR-0020, ticket #242).

Makes the "what to make" decision visible before generation starts: it picks
the recommended treatments, resolves the applied set through the precedence
rules from the depth-dive spec §6, decides whether web search would help this
specific dive (ADR-0012), and emits a :class:`FramingBrief`.

The MVP tracer bullet performs no content analysis — the recommendation and
search-intent decision are deterministic stands-in for the eventual LLM framing
turn (ADR-0012), keyed on the passage's type, its content, and its anchoring so
they stay per-request rather than following a fixed schedule by output type or
keyword.
"""

import re
from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict

from core.types.captured_passage import (
    CapturedPassage,
    CodePassage,
    DiagramPassage,
    ImagePassage,
    TablePassage,
    TextPassage,
)
from core.types.depth_dive import DeferredTreatment, Treatment, TreatmentHint

_SEARCH_INTENT_MAX_CHARS = 200
"""Cap on the length of a derived web-search intent."""

_CONCEPT_MAX_CHARS = 120
"""Cap on the length of the brief's concept summary."""


class FramingBrief(BaseModel):
    """Internal creative brief emitted by the framing agent (ADR-0020).

    Not exposed on ``HarnessBResponse``; the assembly agent consumes it. The
    creative fields beyond treatment routing (key takeaways, visual metaphors,
    required corpus context) are populated by the LLM-driven framing turn and
    are out of scope for the routing-and-search tracer bullet.
    """

    model_config = ConfigDict(extra="forbid")

    concept: str
    recommended_treatments: list[Treatment]
    applied_treatments: list[Treatment]
    routing_note: str | None = None
    search_intent: str | None = None


def run_framing(
    passage: CapturedPassage,
    *,
    requested_treatments: Sequence[TreatmentHint] | None = None,
    preferred_treatments: Sequence[TreatmentHint] | None = None,
) -> FramingBrief:
    """Produce the framing brief for one captured passage and request hints.

    Args:
        passage: The captured passage carried in the request.
        requested_treatments: Explicit, user-typed treatment ask (MVP or
            deferred/still-deferred names).
        preferred_treatments: UI-supplied, request-time-only treatment
            preferences.

    Returns:
        A ``FramingBrief`` carrying the recommended and applied treatments,
        a routing note when an override or fallback occurred, and the
        per-request web-search intent.
    """
    recommended = _recommend_treatments(passage)
    applied, notes = _resolve_treatments(requested_treatments, preferred_treatments, recommended)
    return FramingBrief(
        concept=_concept(passage),
        recommended_treatments=recommended,
        applied_treatments=applied,
        routing_note=_join_notes(notes),
        search_intent=_decide_search_intent(passage),
    )


def _recommend_treatments(passage: CapturedPassage) -> list[Treatment]:
    """Return the harness's blind recommendation for a passage.

    Keyed on passage type (spec §6 maps coding examples to ``worked_example``);
    a general explanatory walkthrough suits text, an image is best revealed
    through prediction, and a table spreads naturally onto swipeable slides.
    """
    if isinstance(passage, (CodePassage, TextPassage)):
        return [Treatment.WORKED_EXAMPLE]
    if isinstance(passage, (ImagePassage, DiagramPassage)):
        return [Treatment.PREDICTION_REVEAL]
    if isinstance(passage, TablePassage):
        return [Treatment.SEGMENTED_CAROUSEL]
    return [Treatment.WORKED_EXAMPLE]


def _resolve_treatments(
    requested: Sequence[TreatmentHint] | None,
    preferred: Sequence[TreatmentHint] | None,
    recommendation: list[Treatment],
) -> tuple[list[Treatment], list[str]]:
    """Apply the §6 precedence rules and collect a routing note.

    Explicit `requested` beats `preferred` beats the harness recommendation. An
    empty or absent hints list contributes nothing, so precedence falls through
    to the next level. Treatments the MVP set supports are kept; deferred
    treatments are dropped and reported so the caller can flag the fallback.
    Precedence overrides of the recommendation are also reported.
    """
    for hints, override_note in (
        (requested, "explicit request overrides the harness recommendation"),
        (preferred, "preferred treatments override the harness recommendation"),
    ):
        if hints:
            applied, notes = _filter_supported(hints)
            if not applied:
                applied = list(recommendation)
            elif applied != recommendation:
                notes.append(override_note)
            return applied, notes
    return list(recommendation), []


def _filter_supported(hints: Sequence[TreatmentHint]) -> tuple[list[Treatment], list[str]]:
    """Split hints into supported treatments and dropped deferred names.

    Returns the supported treatments plus a note describing any deferred
    treatments that were dropped.
    """
    supported: list[Treatment] = []
    dropped: list[str] = []
    for hint in hints:
        if isinstance(hint, Treatment):
            supported.append(hint)
        elif isinstance(hint, DeferredTreatment):
            dropped.append(hint.value)
    notes: list[str] = []
    if dropped:
        names = ", ".join(dropped)
        notes.append(f"deferred and unavailable, dropped: {names}")
    return supported, notes


def _join_notes(notes: Sequence[str]) -> str | None:
    """Join zero or more routing notes into one note, or ``None``."""
    return "; ".join(notes) if notes else None


def _decide_search_intent(passage: CapturedPassage) -> str | None:
    """Decide, per-request, whether external grounding would help.

    This is the tracer-bullet stand-in for the framing LLM's per-request
    judgment that ADR-0012 calls for — the real content assessment lands with
    the LLM framing turn. Until then it substitutes a deterministic rule keyed
    on the request's own properties (anchoring and content), not on output type
    or keyword: a passage anchored to the corpus is grounded by retrieval, so no
    external search is warranted; an unanchored passage has no corpus grounding,
    so a web-search intent is derived from whatever is searchable — text/code
    content or a non-text caption. No caption means nothing to search for.
    """
    if _is_anchored(passage):
        return None
    if isinstance(passage, (TextPassage, CodePassage)):
        return _snippet(passage.content)
    if isinstance(passage, (ImagePassage, DiagramPassage, TablePassage)):
        return _snippet(passage.caption or "")
    return None


def _is_anchored(passage: CapturedPassage) -> bool:
    """Return whether the passage carries a corpus anchor."""
    if isinstance(passage, (TextPassage, CodePassage)):
        return passage.chunk_id is not None
    return passage.document_id is not None


def _snippet(text: str, limit: int = _SEARCH_INTENT_MAX_CHARS) -> str | None:
    """Trim text to a single-line searchable snippet, or ``None`` if empty."""
    collapsed = re.sub(r"\s+", " ", text).strip()
    if not collapsed:
        return None
    return collapsed[:limit]


def _concept(passage: CapturedPassage) -> str:
    """Derive a short concept label for the brief from the passage."""
    if isinstance(passage, (TextPassage, CodePassage)):
        return _brief_snippet(passage.content) or "the captured passage"
    caption = _brief_snippet(passage.caption or "")
    if caption:
        return caption
    kind = "diagram" if isinstance(passage, DiagramPassage) else passage.passage_type
    return f"the captured {kind}"


def _brief_snippet(text: str) -> str | None:
    """Collapse and cap text for the brief's concept field."""
    return _snippet(text, limit=_CONCEPT_MAX_CHARS)


__all__ = ["FramingBrief", "run_framing"]
