"""Captured Passage domain model (ADR-0021).

A Captured Passage is the specific content a user selects or pastes to anchor a
Depth Dive request. One of five passage types — text, image, diagram, table, or
code snippet — modeled as a discriminated union keyed on ``passage_type``.

The content is always carried in the request; the store is never the content
source. Anchors are optional provenance/retrieval context: ``text`` and ``code``
may anchor via ``chunk_id`` (ADR-0014); ``image``, ``diagram``, and ``table``
anchor via ``document_id`` + document-relative ``ordinal``. Unanchored passages
may carry an optional ``source`` URL as provenance only.
"""

from typing import Annotated, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

TableRow = list[str] | dict[str, str]
"""Canonical structured table row: column cells or a header-keyed dict."""


class _PassageBase(BaseModel):
    """Shared ``source`` provenance field for every passage variant."""

    model_config = ConfigDict(extra="forbid")

    source: str | None = None


class TextPassage(_PassageBase):
    """A captured text excerpt.

    ``chunk_id`` optionally anchors the passage to an ingested chunk
    (ADR-0014). Content is always carried in the request.
    """

    passage_type: Literal["text"] = "text"
    content: str
    chunk_id: UUID | None = None


class CodePassage(_PassageBase):
    """A captured code snippet.

    ``language`` is a hint only — unvalidated free text. ``chunk_id``
    optionally anchors the snippet to an ingested chunk (ADR-0014).
    """

    passage_type: Literal["code"] = "code"
    content: str
    language: str | None = None
    chunk_id: UUID | None = None


class _BinaryPassageBase(_PassageBase):
    """Carrier shared by the image and diagram variants."""

    content: bytes
    media_type: Literal["image/png", "image/jpeg", "image/webp", "image/gif"]
    caption: str | None = None
    document_id: UUID | None = None
    ordinal: str | None = None


class ImagePassage(_BinaryPassageBase):
    """A captured image.

    ``document_id`` + ``ordinal`` form the optional parallel anchor,
    distinct from the ``chunk_id`` anchor used by text/code.
    """

    passage_type: Literal["image"] = "image"


class DiagramPassage(_BinaryPassageBase):
    """A captured diagram.

    Same carrier as ``ImagePassage`` but semantically distinct: diagrams may
    feed an internal graph IR for harness routing/assembly. That IR is never
    sent to the model as a graph.
    """

    passage_type: Literal["diagram"] = "diagram"


class TablePassage(_PassageBase):
    """A captured structured table.

    ``rows`` carry the canonical structured form; ``headers`` is optional
    header metadata. ``document_id`` + ``ordinal`` form the optional parallel
    anchor.
    """

    passage_type: Literal["table"] = "table"
    rows: list[TableRow]
    headers: list[str] | None = None
    caption: str | None = None
    document_id: UUID | None = None
    ordinal: str | None = None


CapturedPassage = Annotated[
    TextPassage | CodePassage | ImagePassage | DiagramPassage | TablePassage,
    Field(discriminator="passage_type"),
]
"""Discriminated union keyed on ``passage_type`` (ADR-0021)."""

__all__ = [
    "CapturedPassage",
    "CodePassage",
    "DiagramPassage",
    "ImagePassage",
    "TablePassage",
    "TextPassage",
]
