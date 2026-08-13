"""Tests for the Passage Transform stage (depth-dive spec §5)."""

import base64
import io
from collections.abc import Sequence

import pytest
from PIL import Image

from core.types.captured_passage import (
    CodePassage,
    DiagramPassage,
    ImagePassage,
    TablePassage,
    TableRow,
    TextPassage,
)
from depth_dive.transform import (
    IMAGE_MAX_BYTES,
    IMAGE_MAX_DECODED_BYTES,
    IMAGE_MAX_DIMENSION,
    TABLE_MAX_COLUMNS,
    TABLE_MAX_ROWS,
    TEXT_PASSAGE_MAX_CHARS,
    ImageBlock,
    PassageTransformError,
    TableBlock,
    TextBlock,
    transform_passage,
)


def _png_bytes(width: int = 4, height: int = 4) -> bytes:
    """Build a real PNG payload of the requested dimensions."""
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), (10, 20, 30)).save(buffer, format="PNG")
    return buffer.getvalue()


def _animated_gif_bytes() -> bytes:
    """Build a multi-frame GIF payload."""
    frames = [Image.new("RGB", (3, 3), c) for c in ((255, 0, 0), (0, 255, 0))]
    buffer = io.BytesIO()
    frames[0].save(
        buffer, format="GIF", save_all=True, append_images=frames[1:], duration=100, loop=0
    )
    return buffer.getvalue()


def _table(rows: Sequence[TableRow], headers: list[str] | None = None) -> TablePassage:
    """Build a table passage from structured rows."""
    return TablePassage(rows=list(rows), headers=headers)


# ============================================================
# text
# ============================================================


def test_text_passage_returns_text_block() -> None:
    """A valid text passage normalizes to a ``TextBlock`` with stripped content."""
    result = transform_passage(TextPassage(content="  Attention is all you need.  "))
    assert isinstance(result, TextBlock)
    assert result.text == "Attention is all you need."


def test_text_passage_with_anchor_transforms() -> None:
    """An anchored text passage still transforms; the anchor is ignored here."""
    result = transform_passage(TextPassage(content="Hello world"))
    assert isinstance(result, TextBlock)
    assert result.text == "Hello world"


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
    assert isinstance(result, TextBlock)
    assert len(result.text) == TEXT_PASSAGE_MAX_CHARS


def test_bound_measures_normalized_not_raw_content() -> None:
    """The bound applies to the stripped content, matching the returned value."""
    content = "x" * TEXT_PASSAGE_MAX_CHARS
    result = transform_passage(TextPassage(content=f"  {content}  "))
    assert isinstance(result, TextBlock)
    assert len(result.text) == TEXT_PASSAGE_MAX_CHARS


# ============================================================
# image
# ============================================================


def test_image_passage_returns_base64_image_block() -> None:
    """A valid PNG yields an ``ImageBlock`` with base64 payload and dimensions."""
    png = _png_bytes(8, 6)
    result = transform_passage(ImagePassage(content=png, media_type="image/png"))
    assert isinstance(result, ImageBlock)
    assert result.media_type == "image/png"
    assert result.width == 8
    assert result.height == 6
    assert result.base64 == base64.b64encode(png).decode("ascii")


def test_diagram_passage_returns_image_block() -> None:
    """A diagram uses the same image carrier as an image passage."""
    png = _png_bytes()
    result = transform_passage(DiagramPassage(content=png, media_type="image/png"))
    assert isinstance(result, ImageBlock)
    assert result.width == 4


def test_image_media_type_format_mismatch_is_rejected() -> None:
    """A PNG payload declared as JPEG is rejected as a format mismatch."""
    with pytest.raises(PassageTransformError):
        transform_passage(ImagePassage(content=_png_bytes(), media_type="image/jpeg"))


def test_image_size_bound_is_rejected() -> None:
    """Image bytes exceeding the 5 MB size bound are rejected before decoding."""
    payload = b"_" * (IMAGE_MAX_BYTES + 1)
    with pytest.raises(PassageTransformError):
        transform_passage(ImagePassage(content=payload, media_type="image/png"))


def test_image_dimension_bound_is_accepted() -> None:
    """An image exactly at the maximum dimension is accepted."""
    result = transform_passage(
        ImagePassage(content=_png_bytes(IMAGE_MAX_DIMENSION, 1), media_type="image/png")
    )
    assert isinstance(result, ImageBlock)
    assert result.width == IMAGE_MAX_DIMENSION


def test_image_oversized_dimension_is_rejected() -> None:
    """An image exceeding the per-axis dimension bound is rejected."""
    with pytest.raises(PassageTransformError):
        transform_passage(
            ImagePassage(content=_png_bytes(IMAGE_MAX_DIMENSION + 1, 1), media_type="image/png")
        )


def test_image_decoded_size_bound_is_rejected() -> None:
    """Small-file, high-pixel images exceeding the decoded size bound are rejected."""
    decoded = 1500 * 2000 * 3
    assert decoded > IMAGE_MAX_DECODED_BYTES
    with pytest.raises(PassageTransformError):
        transform_passage(ImagePassage(content=_png_bytes(1500, 2000), media_type="image/png"))


def test_image_decoded_size_bound_is_accepted() -> None:
    """An image under the decoded size bound is accepted."""
    decoded = 800 * 800 * 3
    assert decoded < IMAGE_MAX_DECODED_BYTES
    result = transform_passage(ImagePassage(content=_png_bytes(800, 800), media_type="image/png"))
    assert isinstance(result, ImageBlock)
    assert result.width == 800


def test_non_decodable_image_bytes_are_rejected() -> None:
    """Bytes that are not a valid image are rejected."""
    with pytest.raises(PassageTransformError):
        transform_passage(ImagePassage(content=b"not actually an image", media_type="image/png"))


def test_animated_gif_is_rejected() -> None:
    """An animated GIF violates the non-animated media_type restriction."""
    with pytest.raises(PassageTransformError):
        transform_passage(ImagePassage(content=_animated_gif_bytes(), media_type="image/gif"))


# ============================================================
# table
# ============================================================


def test_table_passage_returns_table_block() -> None:
    """A valid table yields a ``TableBlock`` carrying headers and rows."""
    result = transform_passage(_table(rows=[["a", "1"], ["b", "2"]], headers=["l", "v"]))
    assert isinstance(result, TableBlock)
    assert result.headers == ["l", "v"]
    assert result.rows == [["a", "1"], ["b", "2"]]


def test_table_passage_accepts_dict_rows() -> None:
    """Header-keyed dict rows are accepted as a valid structured form."""
    rows = [{"l": "a", "v": "1"}, {"l": "b", "v": "2"}]
    result = transform_passage(_table(rows=rows))
    assert isinstance(result, TableBlock)
    assert result.rows == rows


def test_empty_table_is_rejected() -> None:
    """A table with no rows and no headers is rejected."""
    with pytest.raises(PassageTransformError):
        transform_passage(_table(rows=[]))


def test_headers_only_table_is_accepted() -> None:
    """A table carrying headers but no rows is a valid capture."""
    result = transform_passage(_table(rows=[], headers=["a", "b"]))
    assert isinstance(result, TableBlock)
    assert result.headers == ["a", "b"]


def test_all_empty_rows_are_rejected() -> None:
    """Rows containing no columns with no headers are rejected as empty."""
    with pytest.raises(PassageTransformError):
        transform_passage(_table(rows=[[""], [""]]))


def test_table_over_row_bound_is_rejected() -> None:
    """A table exceeding the row bound is rejected."""
    rows = [["x"] for _ in range(TABLE_MAX_ROWS + 1)]
    with pytest.raises(PassageTransformError):
        transform_passage(_table(rows=rows))


def test_table_over_column_bound_is_rejected() -> None:
    """A table exceeding the column bound in a row is rejected."""
    row = ["x"] * (TABLE_MAX_COLUMNS + 1)
    with pytest.raises(PassageTransformError):
        transform_passage(_table(rows=[row]))


def test_table_over_column_bound_in_headers_is_rejected() -> None:
    """Headers exceeding the column bound are rejected."""
    headers = ["h"] * (TABLE_MAX_COLUMNS + 1)
    with pytest.raises(PassageTransformError):
        transform_passage(_table(rows=[["x"]], headers=headers))


def test_table_boundary_shape_is_accepted() -> None:
    """A table exactly at the row and column bounds is accepted."""
    rows = [["x"] * TABLE_MAX_COLUMNS for _ in range(TABLE_MAX_ROWS)]
    result = transform_passage(_table(rows=rows))
    assert isinstance(result, TableBlock)
    assert len(result.rows) == TABLE_MAX_ROWS


# ============================================================
# unsupported types
# ============================================================


def test_code_passage_is_rejected_for_tracer_bullet() -> None:
    """Code passages remain unsupported by the MVP tracer bullet."""
    with pytest.raises(PassageTransformError):
        transform_passage(CodePassage(content="def f():\n    pass"))
