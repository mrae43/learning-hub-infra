"""Tests for the Depth Dive harness entrypoint (ADR-0020)."""

import io

import pytest
from PIL import Image
from pydantic import TypeAdapter

from core.types.captured_passage import (
    CapturedPassage,
    ImagePassage,
    TablePassage,
    TextPassage,
)
from core.types.depth_dive import (
    HarnessBRequest,
    HarnessBResponse,
    InteractiveAnimation,
    Treatment,
)
from depth_dive.harness import run_dive
from depth_dive.transform import PassageTransformError

_VALID_TEXT = TextPassage(content="Attention is all you need.")
_captured_passage_adapter: TypeAdapter[CapturedPassage] = TypeAdapter(CapturedPassage)


def _request(passage: CapturedPassage) -> HarnessBRequest:
    """Wrap a passage in a request with no treatment hints."""
    return HarnessBRequest(captured_passage=passage)


def _png_bytes() -> bytes:
    """Build a real PNG payload."""
    buffer = io.BytesIO()
    Image.new("RGB", (4, 4), (10, 20, 30)).save(buffer, format="PNG")
    return buffer.getvalue()


def _valid_image() -> ImagePassage:
    return ImagePassage(content=_png_bytes(), media_type="image/png")


def _valid_table() -> TablePassage:
    return TablePassage(rows=[["a", "1"], ["b", "2"]], headers=["l", "v"])


def test_run_dive_returns_hardcoded_animation() -> None:
    """A valid text passage yields a HarnessBResponse with the demo animation."""
    response = run_dive(_request(_VALID_TEXT))
    assert isinstance(response, HarnessBResponse)
    assert isinstance(response.output, InteractiveAnimation)
    assert response.output.output_type == "interactive_animation"
    assert len(response.output.elements) > 0
    assert len(response.output.steps) > 0
    assert response.output.initial_state


def test_run_dive_response_is_not_grounded_without_retrieval() -> None:
    """No retrieval/search runs in the tracer bullet: grounded=False, search off."""
    response = run_dive(_request(_VALID_TEXT))
    assert response.grounded is False
    assert response.external_search_attempted is False
    assert response.external_search_failed is False
    assert response.external_search_note is None
    assert response.cited_passages == []


def test_run_dive_recommends_and_applies_worked_example() -> None:
    """The demo treats the passage as a worked_example; no routing overrides."""
    response = run_dive(_request(_VALID_TEXT))
    assert response.recommended_treatments == [Treatment.WORKED_EXAMPLE]
    assert response.applied_treatments == [Treatment.WORKED_EXAMPLE]
    assert response.routing_note is None


def test_run_dive_output_is_static_across_passages() -> None:
    """The tracer bullet returns the identical hardcoded payload for any text."""
    first = run_dive(_request(_VALID_TEXT))
    second = run_dive(_request(TextPassage(content="A completely different passage.")))
    assert first.model_dump() == second.model_dump()


def test_run_dive_validates_before_building() -> None:
    """A passage violating text bounds is rejected, not silently built."""
    with pytest.raises(PassageTransformError):
        run_dive(_request(TextPassage(content="")))


def test_run_dive_accepts_image_passage() -> None:
    """A valid image passage returns a valid HarnessBResponse."""
    response = run_dive(_request(_valid_image()))
    assert isinstance(response, HarnessBResponse)
    assert response.output.output_type == "interactive_animation"


def test_run_dive_accepts_table_passage() -> None:
    """A valid table passage returns a valid HarnessBResponse."""
    response = run_dive(_request(_valid_table()))
    assert isinstance(response, HarnessBResponse)
    assert response.output.output_type == "interactive_animation"


def test_run_dive_rejects_invalid_image() -> None:
    """An image that fails transform validation is rejected, not silently built."""
    with pytest.raises(PassageTransformError):
        run_dive(_request(ImagePassage(content=b"not an image", media_type="image/png")))


def test_run_dive_accepts_type_checked_union_payload() -> None:
    """The union type parses request-shaped payloads before the harness runs."""
    passage = _captured_passage_adapter.validate_python(
        {"passage_type": "text", "content": "Typed union payload", "chunk_id": None}
    )
    response = run_dive(_request(passage))
    assert response.output.output_type == "interactive_animation"
