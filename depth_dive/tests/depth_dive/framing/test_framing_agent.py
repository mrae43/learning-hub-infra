"""Tests for the Depth Dive framing agent (ADR-0020, ticket #242).

Covers treatment routing precedence (explicit ask > preferred > harness
recommendation), deferred/unsupported treatment fallback with a ``routing_note``,
and the per-request web-search-intent decision.
"""

from uuid import uuid4

from core.types.captured_passage import (
    CodePassage,
    ImagePassage,
    TablePassage,
    TextPassage,
)
from core.types.depth_dive import DeferredTreatment, Treatment
from depth_dive.framing.framing_agent import run_framing

_VALID_TEXT = TextPassage(content="Attention is all you need.")
_VALID_CODE = CodePassage(content="def f():\n    return 1", language="python")
_VALID_IMAGE = ImagePassage(content=b"\x00", media_type="image/png", caption="An attention plot")
_VALID_TABLE = TablePassage(rows=[["a", "1"], ["b", "2"]], headers=["l", "v"], caption="Scores")


def test_framing_recommends_worked_example_for_text() -> None:
    """The harness recommends worked_example for a text passage."""
    brief = run_framing(_VALID_TEXT)
    assert brief.recommended_treatments == [Treatment.WORKED_EXAMPLE]


def test_framing_recommends_worked_example_for_code() -> None:
    """The harness recommends worked_example for a code passage (spec §6)."""
    brief = run_framing(_VALID_CODE)
    assert brief.recommended_treatments == [Treatment.WORKED_EXAMPLE]


def test_framing_recommends_prediction_reveal_for_image() -> None:
    """The harness recommends prediction_reveal for an image passage."""
    brief = run_framing(_VALID_IMAGE)
    assert brief.recommended_treatments == [Treatment.PREDICTION_REVEAL]


def test_framing_recommends_segmented_carousel_for_table() -> None:
    """The harness recommends segmented_carousel for a table passage."""
    brief = run_framing(_VALID_TABLE)
    assert brief.recommended_treatments == [Treatment.SEGMENTED_CAROUSEL]


# ------------------------------------------------------------
# Routing precedence
# ------------------------------------------------------------


def test_framing_uses_recommendation_when_no_hints() -> None:
    """With no request hints, applied equals the harness recommendation."""
    brief = run_framing(_VALID_TEXT)
    assert brief.applied_treatments == brief.recommended_treatments
    assert brief.routing_note is None


def test_explicit_request_beats_harness_recommendation() -> None:
    """An explicit requested treatment wins over the recommendation with a note."""
    brief = run_framing(_VALID_TEXT, requested_treatments=[Treatment.SEGMENTED_CAROUSEL])
    assert brief.applied_treatments == [Treatment.SEGMENTED_CAROUSEL]
    assert brief.recommended_treatments == [Treatment.WORKED_EXAMPLE]
    assert brief.routing_note is not None
    assert "request" in brief.routing_note


def test_preferred_beats_recommendation_when_no_explicit_ask() -> None:
    """A UI-supplied preferred treatment wins over the recommendation."""
    brief = run_framing(_VALID_TEXT, preferred_treatments=[Treatment.PREDICTION_REVEAL])
    assert brief.applied_treatments == [Treatment.PREDICTION_REVEAL]
    assert brief.routing_note is not None


def test_explicit_request_beats_preferred() -> None:
    """When both are supplied, the explicit ask wins over the preferred set."""
    brief = run_framing(
        _VALID_TEXT,
        requested_treatments=[Treatment.SEGMENTED_CAROUSEL],
        preferred_treatments=[Treatment.PREDICTION_REVEAL],
    )
    assert brief.applied_treatments == [Treatment.SEGMENTED_CAROUSEL]
    assert Treatment.PREDICTION_REVEAL not in brief.applied_treatments


def test_empty_request_falls_through_to_preferred() -> None:
    """An empty explicit-request list is not an ask, so preferred still applies."""
    brief = run_framing(
        _VALID_TEXT,
        requested_treatments=[],
        preferred_treatments=[Treatment.SEGMENTED_CAROUSEL],
    )
    assert brief.applied_treatments == [Treatment.SEGMENTED_CAROUSEL]


def test_matching_explicit_request_produces_no_note() -> None:
    """A request that matches the recommendation needs no routing note."""
    brief = run_framing(_VALID_TEXT, requested_treatments=[Treatment.WORKED_EXAMPLE])
    assert brief.applied_treatments == [Treatment.WORKED_EXAMPLE]
    assert brief.routing_note is None


# ------------------------------------------------------------
# Deferred / unsupported treatment fallback
# ------------------------------------------------------------


def test_deferred_treatment_falls_back_with_note() -> None:
    """A requested deferred treatment is dropped and reported in the note."""
    brief = run_framing(
        _VALID_TEXT,
        requested_treatments=[DeferredTreatment.ANALOGY_MAPPING],
    )
    assert brief.applied_treatments == [Treatment.WORKED_EXAMPLE]
    assert brief.routing_note is not None
    assert DeferredTreatment.ANALOGY_MAPPING.value in brief.routing_note


def test_mixed_request_keeps_supported_and_drops_deferred() -> None:
    """A request mixing supported and deferred treatments keeps the supported ones."""
    brief = run_framing(
        _VALID_TEXT,
        requested_treatments=[
            Treatment.SEGMENTED_CAROUSEL,
            DeferredTreatment.ELABORATIVE_PROMPT,
        ],
    )
    assert brief.applied_treatments == [Treatment.SEGMENTED_CAROUSEL]
    assert brief.routing_note is not None
    assert DeferredTreatment.ELABORATIVE_PROMPT.value in brief.routing_note


def test_deferred_preferred_treatment_falls_back_with_note() -> None:
    """A deferred preferred treatment is dropped and reported."""
    brief = run_framing(
        _VALID_TEXT,
        preferred_treatments=[DeferredTreatment.INTERACTIVE_CONCEPT_MAP],
    )
    assert brief.applied_treatments == [Treatment.WORKED_EXAMPLE]
    assert brief.routing_note is not None
    assert DeferredTreatment.INTERACTIVE_CONCEPT_MAP.value in brief.routing_note


# ------------------------------------------------------------
# Web-search intent decision
# ------------------------------------------------------------


def test_anchored_text_gets_no_search_intent() -> None:
    """An anchored text passage relies on corpus retrieval, so no web intent."""
    passage = TextPassage(content="Attention is all you need.", chunk_id=uuid4())
    brief = run_framing(passage)
    assert brief.search_intent is None


def test_unanchored_text_gets_search_intent() -> None:
    """An unanchored text passage produces a search intent from its content."""
    brief = run_framing(_VALID_TEXT)
    assert brief.search_intent is not None
    assert brief.search_intent == "Attention is all you need."


def test_unanchored_code_gets_search_intent() -> None:
    """An unanchored code passage produces a search intent from its content."""
    brief = run_framing(_VALID_CODE)
    assert brief.search_intent is not None
    assert "def f" in brief.search_intent


def test_anchored_image_gets_no_search_intent() -> None:
    """An anchored image passage relies on corpus retrieval, so no web intent."""
    passage = ImagePassage(
        content=b"\x00",
        media_type="image/png",
        caption="An attention plot",
        document_id=uuid4(),
    )
    brief = run_framing(passage)
    assert brief.search_intent is None


def test_unanchored_image_uses_caption_as_search_intent() -> None:
    """An unanchored image passage with a caption uses that as the web intent."""
    brief = run_framing(_VALID_IMAGE)
    assert brief.search_intent == "An attention plot"


def test_unanchored_image_without_caption_gets_no_search_intent() -> None:
    """Without a caption an unanchored image has no derivable web intent."""
    passage = ImagePassage(content=b"\x00", media_type="image/png")
    brief = run_framing(passage)
    assert brief.search_intent is None


def test_unanchored_whitespace_text_gets_no_search_intent() -> None:
    """Whitespace-only unanchored text has nothing to search for."""
    brief = run_framing(TextPassage(content="   \n  "))
    assert brief.search_intent is None
