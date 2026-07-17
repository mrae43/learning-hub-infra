"""FastAPI BackgroundTasks glue for ingestion.

Per ADR-0006, ingestion runs in-process via FastAPI's ``BackgroundTasks``.
This module is the only place that couples the pipeline to the task runner;
``ingestion/pipeline.py`` remains a plain function.
"""

from uuid import UUID

from fastapi import BackgroundTasks

from core.clients.embeddings_client import EmbeddingsClient
from core.config.settings import settings
from core.database.connection import SessionLocal
from core.database.schema import Document
from core.exceptions import IngestionError
from core.types.document import DocumentStatus, DocumentType
from ingestion.pipeline import run_ingestion


def _execute_ingestion_task(
    document_id: UUID,
    document_type: DocumentType,
    source_filename: str,
    file_bytes: bytes,
) -> None:
    """Synchronous wrapper run by ``BackgroundTasks``.

    Opens its own database session so that a pipeline failure can be captured
    and persisted as ``status='failed'`` after rolling back the partial work.
    """
    session = SessionLocal()
    try:
        embeddings_client = EmbeddingsClient(
            api_key=settings.openai_api_key,
            model=settings.embedding_model,
        )
        title = ""
        document = session.get(Document, document_id)
        if document is not None:
            title = document.title

        run_ingestion(
            document_id=document_id,
            title=title,
            document_type=document_type,
            source_filename=source_filename,
            file_bytes=file_bytes,
            session=session,
            embeddings_client=embeddings_client,
            model_name=settings.embedding_model,
        )
        session.commit()
    except Exception as exc:
        session.rollback()
        error_message = str(exc)
        if isinstance(exc, IngestionError) and exc.__cause__ is not None:
            error_message = f"{exc}: {exc.__cause__}"

        # Update the document row in a fresh mini-transaction so the failure
        # reason is retained after the main transaction rolls back.
        try:
            failure_session = SessionLocal()
            failure_document = failure_session.get(Document, document_id)
            if failure_document is not None:
                failure_document.status = DocumentStatus.FAILED
                failure_document.error_message = error_message
                failure_session.commit()
        finally:
            failure_session.close()
    finally:
        session.close()


def schedule_ingestion(
    background_tasks: BackgroundTasks,
    document_id: UUID,
    document_type: DocumentType,
    source_filename: str,
    file_bytes: bytes,
) -> None:
    """Schedule the ingestion pipeline as a background task.

    Args:
        background_tasks: FastAPI's background task container.
        document_id: UUID of the newly created document row.
        document_type: Document type (e.g. ``DocumentType.PAPER``).
        source_filename: Original upload filename.
        file_bytes: Raw uploaded file contents.
    """
    background_tasks.add_task(
        _execute_ingestion_task,
        document_id,
        document_type,
        source_filename,
        file_bytes,
    )


__all__ = ["schedule_ingestion"]
