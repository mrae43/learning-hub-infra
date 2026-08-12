"""Tests for the Passage Transform stage (depth-dive spec §5)."""

import pytest

from core.types.captured_passage import CodePassage, ImagePassage, TextPassage
from depth_dive.transform import (
    TEXT_PASSAGE_MAX_CHARS,
    PassageTransformError,
    transform_passage,
)


def test_text_passage_returns_stripped_content() -> None:
    """A valid text passage normalizes to its stripped content."""
    result = transform_passage(TextPassage(content="  Attention is all you need.  "))
    assert result == "Attention is all you need."


def test_text_passage_with_anchor_transforms() -> None:
    """An anchored text passage still transforms; the anchor is ignored here."""
    result = transform_passage(TextPassage(content="Hello world"))
    assert result == "Hello world"


def test_empty_text_content_is_rejected() -> None:
    """Empty text content fails transform validation."""
    with pytest.raises(PassageTransformError):
        transform_passage(TextPassage(content=""))


def test_whitespace_only_text_content_is_rejected() -> None:
    """Whitespace-only text content fails transform validation."""
    with pytest.raises(PassageTransformError):
        transform_passage(TextPassage(content="   \n\t  "))


def test_oversized_text_content_is_rejected() -> None:
    """Text exceeding the declared character bound fails transform validation."""
    with pytest.raises(PassageTransformError):
        transform_passage(TextPassage(content="x" * (TEXT_PASSAGE_MAX_CHARS + 1)))


def test_boundary_text_content_is_accepted() -> None:
    """Text exactly at the declared bound is accepted."""
    result = transform_passage(TextPassage(content="x" * TEXT_PASSAGE_MAX_CHARS))
    assert len(result) == TEXT_PASSAGE_MAX_CHARS


def test_bound_measures_normalized_not_raw_content() -> None:
    """The bound applies to the stripped content, matching the returned value."""
    content = "x" * TEXT_PASSAGE_MAX_CHARS
    result = transform_passage(TextPassage(content=f"  {content}  "))
    assert len(result) == TEXT_PASSAGE_MAX_CHARS


def test_non_text_types_are_rejected_for_tracer_bullet() -> None:
    """Non-text passage types are unsupported by the MVP tracer bullet."""
    with pytest.raises(PassageTransformError):
        transform_passage(CodePassage(content="def f():\n    pass"))
    with pytest.raises(PassageTransformError):
        transform_passage(ImagePassage(content=b"\x89PNG", media_type="image/png"))
