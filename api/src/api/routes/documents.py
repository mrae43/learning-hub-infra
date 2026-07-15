"""Document status route: GET /documents/{document_id}."""

from uuid import UUID

from fastapi import APIRouter, HTTPException

from core.database.connection import db_session
from core.database.schema import Document
from core.types.document import DocumentStatusResponse

router = APIRouter(tags=["documents"])


@router.get(
    "/documents/{document_id}",
    response_model=DocumentStatusResponse,
)
def get_document(document_id: UUID) -> DocumentStatusResponse:
    """Return the current status of a document.

    FastAPI validates the path parameter as a UUID and returns 422 for
    malformed values. An unknown ID returns 404.
    """
    with db_session() as session:
        document = session.get(Document, document_id)
        if document is None:
            raise HTTPException(status_code=404, detail="Document not found")
        return DocumentStatusResponse.model_validate(document)
