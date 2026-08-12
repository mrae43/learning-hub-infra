"""Tests for the Captured Passage discriminated union (ADR-0021)."""

from uuid import uuid4

import pytest
from pydantic import TypeAdapter, ValidationError

from core.types.captured_passage import (
    CapturedPassage,
    CodePassage,
    DiagramPassage,
    ImagePassage,
    TablePassage,
    TextPassage,
)

_PNG = b"\x89PNG\r\n\x1a\n"

# ``CapturedPassage`` is an ``Annotated`` union alias; a ``TypeAdapter`` is the
# runtime handle for validating payloads against it at module level.
_captured_passage_adapter: TypeAdapter[CapturedPassage] = TypeAdapter(CapturedPassage)


def test_text_passage_parses_discriminated_union() -> None:
    """A text passage resolves to TextPassage with carried content."""
    chunk_id = uuid4()
    parsed = _captured_passage_adapter.validate_python(
        {"passage_type": "text", "content": "Attention is all you need.", "chunk_id": str(chunk_id)}
    )
    assert isinstance(parsed, TextPassage)
    assert parsed.content == "Attention is all you need."
    assert parsed.chunk_id == chunk_id
    assert parsed.source is None


def test_text_passage_defaults_unanchored() -> None:
    """A text passage without chunk_id or source parses as unanchored."""
    parsed = _captured_passage_adapter.validate_python({"passage_type": "text", "content": "Hello"})
    assert isinstance(parsed, TextPassage)
    assert parsed.chunk_id is None
    assert parsed.source is None


def test_code_passage_parses_discriminated_union() -> None:
    """A code passage carries content, language, and optional chunk anchor."""
    parsed = _captured_passage_adapter.validate_python(
        {
            "passage_type": "code",
            "content": "def f():\n    return 1",
            "language": "python",
            "source": "https://example.com/snippet",
        }
    )
    assert isinstance(parsed, CodePassage)
    assert parsed.language == "python"
    assert parsed.source == "https://example.com/snippet"


def test_image_passage_parses_bytes_and_media_type() -> None:
    """An image passage carries raw bytes and a validated media type."""
    document_id = uuid4()
    parsed = _captured_passage_adapter.validate_python(
        {
            "passage_type": "image",
            "content": _PNG,
            "media_type": "image/png",
            "caption": "Figure 1",
            "document_id": str(document_id),
            "ordinal": "Figure 1",
        }
    )
    assert isinstance(parsed, ImagePassage)
    assert parsed.content == _PNG
    assert parsed.document_id == document_id
    assert parsed.ordinal == "Figure 1"


def test_image_passage_rejects_unsupported_media_type() -> None:
    """An image passage rejects a media type outside the allowed set."""
    with pytest.raises(ValidationError):
        _captured_passage_adapter.validate_python(
            {"passage_type": "image", "content": _PNG, "media_type": "image/svg+xml"}
        )


def test_diagram_passage_is_distinct_variant() -> None:
    """A diagram passage parses to DiagramPassage with the same carrier shape."""
    parsed = _captured_passage_adapter.validate_python(
        {"passage_type": "diagram", "content": _PNG, "media_type": "image/jpeg"}
    )
    assert isinstance(parsed, DiagramPassage)


def test_table_passage_parses_rows_and_headers() -> None:
    """A table passage carries structured rows plus optional headers."""
    parsed = _captured_passage_adapter.validate_python(
        {
            "passage_type": "table",
            "rows": [["a", "1"], ["b", "2"]],
            "headers": ["letter", "number"],
            "caption": "Table 1",
        }
    )
    assert isinstance(parsed, TablePassage)
    assert parsed.rows == [["a", "1"], ["b", "2"]]
    assert parsed.headers == ["letter", "number"]


def test_table_passage_accepts_header_keyed_row_dicts() -> None:
    """A table passage accepts dict rows keyed by header."""
    parsed = _captured_passage_adapter.validate_python(
        {"passage_type": "table", "rows": [{"letter": "a", "number": "1"}]}
    )
    assert isinstance(parsed, TablePassage)
    assert parsed.rows == [{"letter": "a", "number": "1"}]


def test_missing_discriminator_rejects() -> None:
    """A passage without passage_type fails union validation."""
    with pytest.raises(ValidationError):
        _captured_passage_adapter.validate_python({"content": "no type"})


def test_unknown_discriminator_rejects() -> None:
    """An unknown passage_type fails union validation."""
    with pytest.raises(ValidationError):
        _captured_passage_adapter.validate_python({"passage_type": "audio", "content": "x"})


def test_variant_rejects_extra_fields() -> None:
    """Passage variants forbid undeclared fields (boundary hygiene)."""
    with pytest.raises(ValidationError):
        TextPassage(content="x", surprise=True)  # type: ignore[call-arg]


def test_content_always_carried_not_lazy() -> None:
    """Content is a required field; a passage without it fails validation."""
    with pytest.raises(ValidationError):
        _captured_passage_adapter.validate_python({"passage_type": "text"})
