"""FastAPI BackgroundTasks glue for ingestion.

Per ADR-0006, ingestion runs in-process via FastAPI's ``BackgroundTasks``.
This module is the only place that couples the pipeline to the task runner;
``ingestion/pipeline.py`` remains a plain function.
"""

from fastapi import BackgroundTasks

from core.clients import Embedder
from core.database.connection import SessionLocal
from core.database.schema import Document
from core.exceptions import IngestionError
from core.types.document import DocumentStatus
from ingestion.models import PendingIngestion
from ingestion.pipeline import run_ingestion


def _execute_ingestion_task(
    pending: PendingIngestion,
    embedder: Embedder,
    model_name: str,
) -> None:
    """Synchronous wrapper run by ``BackgroundTasks``.

    Opens its own database session so that a pipeline failure can be captured
    and persisted as ``status='failed'`` after rolling back the partial work.
    """
    session = SessionLocal()
    try:
        title = ""
        document = session.get(Document, pending.document_id)
        if document is not None:
            title = document.title

        resolved = pending.model_copy(update={"title": title})
        run_ingestion(
            pending=resolved,
            session=session,
            embeddings_client=embedder,
            model_name=model_name,
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
            failure_document = failure_session.get(Document, pending.document_id)
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
    pending: PendingIngestion,
    embedder: Embedder,
    model_name: str,
) -> None:
    """Schedule the ingestion pipeline as a background task.

    Args:
        background_tasks: FastAPI's background task container.
        pending: Document-identity fields for the ingestion.
        embedder: Provider used to embed chunk contents.
        model_name: Model name to store alongside each embedding row.
    """
    background_tasks.add_task(
        _execute_ingestion_task,
        pending,
        embedder,
        model_name,
    )


__all__ = ["schedule_ingestion"]
