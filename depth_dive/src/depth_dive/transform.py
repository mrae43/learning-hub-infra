"""Passage Transform stage (depth-dive spec §5).

Runs before the framing agent: validates the carried content against the
declared ``passage_type``, normalizes, and emits a model-ready content block.
Each passage variant maps to a carrier — plain text for ``text``, a base64
image block for ``image``/``diagram``, and structured rows plus a rendered
image for ``table``.

Requests exceeding declared bounds are rejected (422 via the route), never
silently cropped. Image bytes are size-capped before decoding and validated
for format, dimensions, and animation; tables are validated for row/column
shape. A table whose structured form exceeds the row/column bounds falls back
to its rendered image alone (spec §5) rather than being rejected; only when
the rendered image itself exceeds the image size bounds is the passage
rejected.
"""

import base64
import io
from dataclasses import dataclass
from typing import Annotated, Literal, cast

from PIL import Image, ImageDraw, ImageFont, UnidentifiedImageError
from pydantic import BaseModel, ConfigDict, Field

from core.types.captured_passage import (
    CapturedPassage,
    CodePassage,
    DiagramPassage,
    ImageMediaType,
    ImagePassage,
    TablePassage,
    TableRow,
    TextPassage,
)

TEXT_PASSAGE_MAX_CHARS = 100_000
"""MVP size bound for text passages (spec §5: bounded by model token budget)."""

IMAGE_MAX_BYTES = 5 * 1024 * 1024
"""MVP size bound for image/diagram encoded content bytes (spec §5: 5 MB)."""

IMAGE_MAX_DECODED_BYTES = 5 * 1024 * 1024
"""MVP size bound for the decoded pixel buffer (spec §5: 5 MB decoded)."""

IMAGE_MAX_DIMENSION = 8192
"""MVP per-axis pixel bound for image/diagram content (spec §5: 8192 x 8192)."""

TABLE_MAX_ROWS = 200
"""MVP row bound for table passages (spec §5: 200 rows)."""

TABLE_MAX_COLUMNS = 50
"""MVP column bound for table passages (spec §5: 50 columns)."""

_MEDIA_TO_FORMAT: dict[ImageMediaType, str] = {
    "image/png": "PNG",
    "image/jpeg": "JPEG",
    "image/webp": "WEBP",
    "image/gif": "GIF",
}
"""Allowed media types mapped to their Pillow format identifier."""

_CHANNELS_BY_MODE: dict[str, int] = {
    "1": 1,
    "L": 1,
    "P": 1,
    "LA": 2,
    "RGB": 3,
    "YCbCr": 3,
    "RGBA": 4,
    "CMYK": 4,
    "I": 4,
    "F": 4,
}
"""Pillow image mode to bytes-per-pixel, used for decoded-size accounting."""


class PassageTransformError(Exception):
    """A captured passage violated transform validation or is unsupported.

    Raised for size-bound and content violations. The route layer maps this
    to a 422 response.
    """


class TextBlock(BaseModel):
    """A model-ready plain text content block.

    ``text``/``code`` passages share this carrier; a ``code`` passage sets
    ``language`` to its (unvalidated) language hint (spec §5).
    """

    model_config = ConfigDict(extra="forbid")

    kind: Literal["text"] = "text"
    text: str
    language: str | None = None


class ImageBlock(BaseModel):
    """A model-ready base64-encoded image content block."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["image"] = "image"
    media_type: ImageMediaType
    base64: str
    width: int
    height: int


class TableBlock(BaseModel):
    """A model-ready table content block: structured rows plus a rendered image.

    ``headers``/``rows`` carry the canonical structured form (spec §5's
    "structured text"), and ``image`` carries a faithful render of the same
    table as a PNG. When the structured form exceeds the row/column bounds the
    transform returns a bare ``ImageBlock`` instead (image-only fallback), so a
    ``TableBlock`` always carries all three fields.
    """

    model_config = ConfigDict(extra="forbid")

    kind: Literal["table"] = "table"
    headers: list[str] | None = None
    rows: list[TableRow]
    image: ImageBlock


ModelReadyBlock = Annotated[
    TextBlock | ImageBlock | TableBlock,
    Field(discriminator="kind"),
]
"""The model-ready carrier chosen by the transform for a captured passage."""


@dataclass(frozen=True)
class _ImageMetadata:
    """Format and dimensions read from validated image bytes."""

    width: int
    height: int
    decoded_bytes: int


def transform_passage(passage: CapturedPassage) -> ModelReadyBlock:
    """Validate a captured passage and choose its model-ready carrier.

    Args:
        passage: The captured passage carried in the request.

    Returns:
        A ``TextBlock``, ``ImageBlock``, or ``TableBlock`` matching the
        passage variant.

    Raises:
        PassageTransformError: The passage content violates the declared
            bounds, or the passage type is not supported by this transform.
    """
    if isinstance(passage, TextPassage):
        return TextBlock(text=_transform_text(passage.content))
    if isinstance(passage, CodePassage):
        return TextBlock(text=_transform_text(passage.content), language=passage.language)
    if isinstance(passage, (ImagePassage, DiagramPassage)):
        return _transform_image(passage.content, passage.media_type)
    if isinstance(passage, TablePassage):
        return _transform_table(passage.headers, passage.rows)
    raise PassageTransformError(
        f"passage_type {passage.passage_type!r} is not supported by the MVP tracer bullet"
    )


def _transform_text(content: str) -> str:
    """Validate and normalize a text passage.

    Args:
        content: The raw captured text content.

    Returns:
        The stripped text content.

    Raises:
        PassageTransformError: The content is empty/whitespace-only or
            exceeds ``TEXT_PASSAGE_MAX_CHARS`` characters.
    """
    stripped = content.strip()
    if not stripped:
        raise PassageTransformError("text passage content must be non-empty")
    if len(stripped) > TEXT_PASSAGE_MAX_CHARS:
        raise PassageTransformError(
            f"text passage content exceeds the {TEXT_PASSAGE_MAX_CHARS} character limit"
        )
    return stripped


def _transform_image(content: bytes, media_type: ImageMediaType) -> ImageBlock:
    """Validate image bytes and build the base64 image carrier.

    Args:
        content: The raw captured image bytes.
        media_type: The declared media type; must match the actual image.

    Returns:
        An ``ImageBlock`` carrying the base64 payload and validated
        dimensions.

    Raises:
        PassageTransformError: The image exceeds the byte or dimension bound,
            is animated, or its content does not match ``media_type``.
    """
    if len(content) > IMAGE_MAX_BYTES:
        raise PassageTransformError(f"image exceeds the {IMAGE_MAX_BYTES} byte size limit")
    metadata = _read_image_metadata(content, media_type)
    if metadata.width > IMAGE_MAX_DIMENSION or metadata.height > IMAGE_MAX_DIMENSION:
        raise PassageTransformError(
            f"image exceeds the {IMAGE_MAX_DIMENSION}x{IMAGE_MAX_DIMENSION} dimension limit"
        )
    if metadata.decoded_bytes > IMAGE_MAX_DECODED_BYTES:
        raise PassageTransformError(
            f"image exceeds the {IMAGE_MAX_DECODED_BYTES} decoded byte size limit"
        )
    return ImageBlock(
        media_type=media_type,
        base64=base64.b64encode(content).decode("ascii"),
        width=metadata.width,
        height=metadata.height,
    )


def _read_image_metadata(content: bytes, media_type: ImageMediaType) -> _ImageMetadata:
    """Read validated format, dimensions, and animation state from image bytes.

    Args:
        content: The raw captured image bytes.
        media_type: The declared media type.

    Returns:
        The validated image dimensions.

    Raises:
        PassageTransformError: The declared ``media_type`` is unsupported, the
            bytes are not a decodable image, the content format does not match
            ``media_type``, or the image is an animated GIF.
    """
    expected_format = _MEDIA_TO_FORMAT.get(media_type)
    if expected_format is None:
        raise PassageTransformError(f"unsupported media_type {media_type!r}")
    try:
        with Image.open(io.BytesIO(content)) as image:
            if image.format != expected_format:
                raise PassageTransformError(
                    f"image content format {image.format!r} does not match "
                    f"media_type {media_type!r}"
                )
            width, height = image.size
            if getattr(image, "n_frames", 1) > 1:
                raise PassageTransformError("animated images are not supported")
            decoded_bytes = width * height * _CHANNELS_BY_MODE.get(image.mode, 3)
            image.verify()
    except PassageTransformError:
        raise
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise PassageTransformError(f"image content is not a valid {media_type}") from exc
    return _ImageMetadata(width=width, height=height, decoded_bytes=decoded_bytes)


def _transform_table(headers: list[str] | None, rows: list[TableRow]) -> TableBlock | ImageBlock:
    """Validate table shape and build the structured-rows-plus-image carrier.

    A table within the row/column bounds yields a ``TableBlock`` carrying both
    the structured form and a rendered image. A table whose structured form
    exceeds those bounds falls back to the rendered image alone (an
    ``ImageBlock``), per spec §5's "image-only fallback", rather than being
    rejected. In both cases the rendered image must fit the image size bounds.

    Args:
        headers: Optional header metadata.
        rows: The canonical structured rows.

    Returns:
        A ``TableBlock`` (structured rows + rendered image) when the table is
        within bounds, or an ``ImageBlock`` (rendered image only) when the
        structured form exceeds them.

    Raises:
        PassageTransformError: The table is empty, or the rendered image
            exceeds the image size bounds.
    """
    has_content = bool(headers) or any(any(cell != "" for cell in row) for row in rows)
    if not has_content:
        raise PassageTransformError("table passage must contain at least one column")
    within_bounds = _within_table_bounds(headers, rows)
    image = _render_table_image(headers, rows, check_bounds=not within_bounds)
    if within_bounds:
        return TableBlock(headers=headers, rows=rows, image=image)
    return image


def _within_table_bounds(headers: list[str] | None, rows: list[TableRow]) -> bool:
    """Return whether the structured table form fits the row/column bounds."""
    if headers and len(headers) > TABLE_MAX_COLUMNS:
        return False
    if len(rows) > TABLE_MAX_ROWS:
        return False
    return all(len(row) <= TABLE_MAX_COLUMNS for row in rows)


_TABLE_RENDER_CELL_PAD_X = 2
"""Total horizontal padding (pixels) inside a rendered table cell."""

_TABLE_RENDER_CELL_PAD_Y = 0
"""Total vertical padding (pixels) inside a rendered table cell."""


def _render_table_image(
    headers: list[str] | None, rows: list[TableRow], *, check_bounds: bool
) -> ImageBlock:
    """Render the table to a PNG and return its ``ImageBlock`` carrier.

    Args:
        headers: Optional header metadata.
        rows: The canonical structured rows.
        check_bounds: Whether to reject the render for exceeding the image
            size bounds. Only the image-only fallback enforces these bounds
            (spec §5); an in-bounds table keeps its structured rows as the
            primary carrier and is accepted even if its render is large.

    Returns:
        An ``ImageBlock`` carrying the base64-encoded PNG render and its pixel
        dimensions.

    Raises:
        PassageTransformError: ``check_bounds`` is true and the rendered image
            exceeds the image size bounds (per-axis dimension, decoded bytes,
            or encoded bytes).
    """
    grid = _table_to_grid(headers, rows)
    # ``load_default`` always returns a ``FreeTypeFont`` when FreeType support
    # is present (it is, in the bundled Pillow wheels); the union in the stub
    # covers the no-FreeType fallback, which we never hit.
    font = cast(ImageFont.FreeTypeFont, ImageFont.load_default())
    ascent, descent = font.getmetrics()
    line_height = ascent + descent

    n_cols = max((len(cells) for cells in grid), default=0)
    col_widths = [
        max(
            (_cell_text_width(font, _cell_at(grid, row, col)) for row in range(len(grid))),
            default=0,
        )
        for col in range(n_cols)
    ]

    cell_height = line_height + _TABLE_RENDER_CELL_PAD_Y
    width = sum(col_widths) + _TABLE_RENDER_CELL_PAD_X * n_cols + (n_cols + 1)
    height = cell_height * len(grid) + (len(grid) + 1)
    if check_bounds:
        _check_rendered_bounds(width, height)

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    for row in range(len(grid) + 1):
        y = row * cell_height
        draw.line([(0, y), (width - 1, y)], fill="black")
    x = 0
    text_offsets: list[int] = []
    for _col, col_width in enumerate(col_widths):
        draw.line([(x, 0), (x, height - 1)], fill="black")
        text_offsets.append(x + 1 + _TABLE_RENDER_CELL_PAD_X // 2)
        x += 1 + _TABLE_RENDER_CELL_PAD_X + col_width
    draw.line([(x, 0), (x, height - 1)], fill="black")

    for row, cells in enumerate(grid):
        y = row * cell_height + _TABLE_RENDER_CELL_PAD_Y // 2
        for col, cell in enumerate(cells):
            if col >= n_cols:
                break
            draw.text((text_offsets[col], y), cell, font=font, fill="black")

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    png = buffer.getvalue()
    if check_bounds and len(png) > IMAGE_MAX_BYTES:
        raise PassageTransformError(
            f"rendered table image exceeds the {IMAGE_MAX_BYTES} byte size limit"
        )
    return ImageBlock(
        media_type="image/png",
        base64=base64.b64encode(png).decode("ascii"),
        width=width,
        height=height,
    )


def _table_to_grid(headers: list[str] | None, rows: list[TableRow]) -> list[list[str]]:
    """Normalize the structured table into a rectangular grid of cell strings."""
    if headers is not None:
        return [list(headers), *[_row_cells(row, headers) for row in rows]]
    if rows and isinstance(rows[0], dict):
        columns = _dict_columns(rows)
        return [columns, *[_row_cells(row, columns) for row in rows]]
    return [_row_cells(row, None) for row in rows]


def _row_cells(row: TableRow, columns: list[str] | None) -> list[str]:
    """Flatten one table row to its ordered cell strings."""
    if isinstance(row, list):
        return list(row)
    if columns is None:
        return list(row.values())
    return [row.get(column, "") for column in columns]


def _dict_columns(rows: list[TableRow]) -> list[str]:
    """Return the union of dict-row keys in first-seen order."""
    columns: list[str] = []
    for row in rows:
        if isinstance(row, list):
            continue
        for key in row:
            if key not in columns:
                columns.append(key)
    return columns


def _cell_at(grid: list[list[str]], row: int, col: int) -> str:
    """Return the cell at ``(row, col)``, or an empty string if out of range."""
    cells = grid[row]
    return cells[col] if col < len(cells) else ""


def _cell_text_width(font: ImageFont.FreeTypeFont, text: str) -> int:
    """Measure the rendered width (pixels) of a cell's text."""
    if not text:
        return 0
    left, _, right, _ = font.getbbox(text)
    return int(right - left)


def _check_rendered_bounds(width: int, height: int) -> None:
    """Reject a rendered table that exceeds the image size bounds."""
    if width > IMAGE_MAX_DIMENSION or height > IMAGE_MAX_DIMENSION:
        raise PassageTransformError(
            f"rendered table image exceeds the {IMAGE_MAX_DIMENSION}x"
            f"{IMAGE_MAX_DIMENSION} dimension limit"
        )
    if width * height * _CHANNELS_BY_MODE["RGB"] > IMAGE_MAX_DECODED_BYTES:
        raise PassageTransformError(
            f"rendered table image exceeds the {IMAGE_MAX_DECODED_BYTES} decoded byte size limit"
        )


__all__ = [
    "IMAGE_MAX_BYTES",
    "IMAGE_MAX_DECODED_BYTES",
    "IMAGE_MAX_DIMENSION",
    "TABLE_MAX_COLUMNS",
    "TABLE_MAX_ROWS",
    "TEXT_PASSAGE_MAX_CHARS",
    "ImageBlock",
    "ImageMediaType",
    "ModelReadyBlock",
    "PassageTransformError",
    "TableBlock",
    "TextBlock",
    "transform_passage",
]
