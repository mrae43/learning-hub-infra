"""Shared document-related Pydantic models."""

from datetime import datetime
from enum import StrEnum
from uuid import UUID

from pydantic import BaseModel, ConfigDict


class DocumentType(StrEnum):
    """Known document types, matching the document-type chunkers."""

    PAPER = "paper"
    BOOK = "book"
    DOCUMENTATION = "documentation"


class DocumentStatus(StrEnum):
    """Pipeline phase states for ingestion.

    Order reflects the happy path: validating → chunking → embedding → ready.
    ``failed`` is reachable from any phase.
    """

    VALIDATING = "validating"
    CHUNKING = "chunking"
    EMBEDDING = "embedding"
    READY = "ready"
    FAILED = "failed"


class DocumentStatusResponse(BaseModel):
    """Public representation of a document row.

    Reused across ``POST /ingest`` (202 Accepted) and ``GET /documents/{id}``
    (200 OK). ``from_attributes=True`` lets FastAPI construct it directly from
    a SQLAlchemy row.
    """

    model_config = ConfigDict(from_attributes=True)

    document_id: UUID
    title: str
    document_type: DocumentType
    status: DocumentStatus
    error_message: str | None
    source_filename: str
    created_at: datetime
    updated_at: datetime
