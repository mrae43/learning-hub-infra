"""Passage Transform stage (depth-dive spec §5).

Runs before the framing agent: validates the carried content against the
declared ``passage_type``, normalizes, and emits a model-ready content block.
Each passage variant maps to a carrier — plain text for ``text``, a base64
image block for ``image``/``diagram``, and structured rows for ``table``.

Requests exceeding declared bounds are rejected (422 via the route), never
silently cropped. Image bytes are size-capped before decoding and validated
for format, dimensions, and animation; tables are validated for row/column
shape.
"""

import base64
import io
from dataclasses import dataclass
from typing import Annotated, Literal

from PIL import Image, UnidentifiedImageError
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
    """A model-ready structured table content block."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["table"] = "table"
    headers: list[str] | None = None
    rows: list[TableRow]


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


def _transform_table(headers: list[str] | None, rows: list[TableRow]) -> TableBlock:
    """Validate table shape and build the structured rows carrier.

    Args:
        headers: Optional header metadata.
        rows: The canonical structured rows.

    Returns:
        A ``TableBlock`` carrying the validated headers and rows.

    Raises:
        PassageTransformError: The table is empty or exceeds the row/column
            bounds.
    """
    has_content = bool(headers) or any(any(cell != "" for cell in row) for row in rows)
    if not has_content:
        raise PassageTransformError("table passage must contain at least one column")
    if headers and len(headers) > TABLE_MAX_COLUMNS:
        raise PassageTransformError(f"table exceeds the {TABLE_MAX_COLUMNS} column limit")
    if len(rows) > TABLE_MAX_ROWS:
        raise PassageTransformError(f"table exceeds the {TABLE_MAX_ROWS} row limit")
    for row in rows:
        if len(row) > TABLE_MAX_COLUMNS:
            raise PassageTransformError(f"table exceeds the {TABLE_MAX_COLUMNS} column limit")
    return TableBlock(headers=headers, rows=rows)


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
