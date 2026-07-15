"""Chunk metadata Pydantic models.

Schema-level chunk metadata is stored as JSONB; the shape contract is enforced
in Python before persisting (ADR-0014).
"""

from pydantic import BaseModel, ConfigDict


class PaperChunkMetadata(BaseModel):
    """Metadata for a paper chunk."""

    model_config = ConfigDict(extra="forbid")

    section: str
    subsection: str | None
    page: int


class BookChunkMetadata(BaseModel):
    """Metadata for a book chunk."""

    model_config = ConfigDict(extra="forbid")

    chapter: int
    heading: str | None


class DocumentationChunkMetadata(BaseModel):
    """Metadata for a documentation chunk."""

    model_config = ConfigDict(extra="forbid")

    page: str
    section: str | None
