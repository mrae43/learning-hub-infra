"""Ingestion route: POST /ingest."""

import atexit
import contextlib
import os
import tempfile
from pathlib import Path
from typing import Annotated
from uuid import UUID

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    Form,
    HTTPException,
    Request,
    Response,
    UploadFile,
)
from pydantic import Field

from api.dependencies import get_embedder
from core.clients import Embedder
from core.config.settings import settings
from core.database.connection import db_session
from core.database.schema import Document
from core.types.document import DocumentStatusResponse, DocumentType
from ingestion.models import PendingIngestion
from ingestion.tasks import schedule_ingestion

MAGIC_PDF = b"%PDF"
MAGIC_ZIP = b"PK"

router = APIRouter(tags=["ingestion"])


async def _validate_content_type(file: UploadFile, extension: str) -> None:
    head = await file.read(4)
    await file.seek(0)

    if extension == "pdf" and not head.startswith(MAGIC_PDF):
        raise HTTPException(
            status_code=415,
            detail="File content does not match PDF format",
        )
    if extension == "epub" and not head.startswith(MAGIC_ZIP):
        raise HTTPException(
            status_code=415,
            detail="File content does not match EPUB format",
        )


def _extension(filename: str | None) -> str | None:
    """Return the lower-case extension without the leading dot, or None."""
    if not filename:
        return None
    suffix = Path(filename).suffix.lower().lstrip(".")
    return suffix if suffix else None


# Track temp files handed off to background tasks so the atexit handler can
# clean them up if the process exits before the background task runs.
_pending_temp_files: set[Path] = set()


@atexit.register
def _cleanup_pending_temp_files() -> None:
    for path in _pending_temp_files:
        with contextlib.suppress(OSError):
            path.unlink(missing_ok=True)


async def _save_upload_to_temp(upload: UploadFile, max_bytes: int) -> Path:
    """Write upload to a temp file, raising 413 if the limit is exceeded.

    Returns the ``Path`` to the temp file, which the caller is responsible for
    cleaning up after the background ingestion task completes.
    """
    suffix = Path(upload.filename or "upload").suffix or ".tmp"
    fd, path = tempfile.mkstemp(suffix=suffix)
    total = 0
    try:
        with os.fdopen(fd, "wb") as f:
            while True:
                chunk = await upload.read(8192)
                if not chunk:
                    break
                total += len(chunk)
                if total > max_bytes:
                    raise HTTPException(status_code=413, detail="File exceeds maximum upload size")
                f.write(chunk)
    except Exception:
        os.unlink(path)
        raise
    return Path(path)


@router.post("/ingest", status_code=202, response_model=DocumentStatusResponse)
async def ingest_document(
    request: Request,
    response: Response,
    background_tasks: BackgroundTasks,
    file: UploadFile,
    title: Annotated[str, Form(...), Field(max_length=500)],
    document_type: Annotated[DocumentType, Form(...)],
    embedder: Annotated[Embedder, Depends(get_embedder)],
) -> DocumentStatusResponse:
    """Accept a document for background ingestion.

    Returns 202 Accepted with a ``Location`` header pointing at the status
    endpoint.     Pre-flight checks return 415 for unsupported content/mime type and 413 for
    oversized files; missing fields return 422 via FastAPI defaults.
    """
    extension = _extension(file.filename)
    if extension is None or extension not in settings.allowed_file_extensions:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type: {extension or 'unknown'}",
        )

    await _validate_content_type(file, extension)

    file_path = await _save_upload_to_temp(file, settings.max_upload_bytes)
    source_filename = file.filename or "upload"

    try:
        with db_session() as session:
            document = Document(
                title=title,
                document_type=document_type,
                source_filename=source_filename,
            )
            session.add(document)
            session.flush()
            document_id: UUID = document.document_id
            response_body = DocumentStatusResponse.model_validate(document)
    except Exception as exc:
        os.unlink(file_path)
        raise HTTPException(status_code=500, detail=f"Failed to create document: {exc}") from exc

    schedule_ingestion(
        background_tasks,
        pending=PendingIngestion(
            document_id=document_id,
            title=title,
            document_type=document_type,
            source_filename=source_filename,
            file_path=file_path,
        ),
        embedder=embedder,
        model_name=settings.embedding_model,
    )

    _pending_temp_files.add(file_path)

    location = request.url_for("get_document", document_id=str(document_id))
    response.headers["Location"] = str(location)
    return response_body
