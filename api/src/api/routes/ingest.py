"""Ingestion route: POST /ingest."""

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

from api.dependencies import get_embedder
from core.clients import Embedder
from core.config.settings import settings
from core.database.connection import db_session
from core.database.schema import Document
from core.types.document import DocumentStatusResponse, DocumentType
from ingestion.models import PendingIngestion
from ingestion.tasks import schedule_ingestion

router = APIRouter(tags=["ingestion"])


def _extension(filename: str | None) -> str | None:
    """Return the lower-case extension without the leading dot, or None."""
    if not filename:
        return None
    suffix = Path(filename).suffix.lower().lstrip(".")
    return suffix if suffix else None


async def _read_upload_with_limit(upload: UploadFile, max_bytes: int) -> bytes:
    """Read upload contents, raising 413 if the limit is exceeded."""
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = await upload.read(8192)
        if not chunk:
            break
        total += len(chunk)
        if total > max_bytes:
            raise HTTPException(status_code=413, detail="File exceeds maximum upload size")
        chunks.append(chunk)
    return b"".join(chunks)


@router.post("/ingest", status_code=202, response_model=DocumentStatusResponse)
async def ingest_document(
    request: Request,
    response: Response,
    background_tasks: BackgroundTasks,
    file: UploadFile,
    title: Annotated[str, Form(...)],
    document_type: Annotated[DocumentType, Form(...)],
    embedder: Annotated[Embedder, Depends(get_embedder)],
) -> DocumentStatusResponse:
    """Accept a document for background ingestion.

    Returns 202 Accepted with a ``Location`` header pointing at the status
    endpoint. Pre-flight checks return 413 for oversized files and 415 for
    unsupported extensions; missing fields return 422 via FastAPI defaults.
    """
    extension = _extension(file.filename)
    if extension is None or extension not in settings.allowed_file_extensions:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type: {extension or 'unknown'}",
        )

    file_bytes = await _read_upload_with_limit(file, settings.max_upload_bytes)
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
        raise HTTPException(status_code=500, detail=f"Failed to create document: {exc}") from exc

    schedule_ingestion(
        background_tasks,
        pending=PendingIngestion(
            document_id=document_id,
            title=title,
            document_type=document_type,
            source_filename=source_filename,
            file_bytes=file_bytes,
        ),
        embedder=embedder,
        model_name=settings.embedding_model,
    )

    location = request.url_for("get_document", document_id=str(document_id))
    response.headers["Location"] = str(location)
    return response_body
