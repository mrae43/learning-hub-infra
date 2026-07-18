"""Pydantic models for the ingestion package."""

from uuid import UUID

from pydantic import BaseModel

from core.types.document import DocumentType


class PendingIngestion(BaseModel):
    """Document-identity fields grouped for the ingestion pipeline.

    This model replaces individual parameters threaded through
    ``run_ingestion()``, ``schedule_ingestion()``, and
    ``_execute_ingestion_task()``. Carries the fields needed to identify
    and process a single document upload.
    """

    document_id: UUID
    title: str
    document_type: DocumentType
    source_filename: str
    file_bytes: bytes
