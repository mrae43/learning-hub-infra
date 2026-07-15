"""Tests for document-related Pydantic models."""

from datetime import UTC, datetime
from uuid import uuid4

from core.types.document import DocumentStatus, DocumentStatusResponse, DocumentType


def test_document_type_enum_values() -> None:
    """DocumentType exposes the expected string values."""
    assert str(DocumentType.PAPER) == "paper"
    assert str(DocumentType.BOOK) == "book"
    assert str(DocumentType.DOCUMENTATION) == "documentation"


def test_document_status_enum_values() -> None:
    """DocumentStatus exposes the expected pipeline phase values."""
    assert str(DocumentStatus.VALIDATING) == "validating"
    assert str(DocumentStatus.CHUNKING) == "chunking"
    assert str(DocumentStatus.EMBEDDING) == "embedding"
    assert str(DocumentStatus.READY) == "ready"
    assert str(DocumentStatus.FAILED) == "failed"


def test_document_status_response_roundtrip() -> None:
    """DocumentStatusResponse serialises and deserialises cleanly."""
    now = datetime.now(UTC)
    doc_id = uuid4()
    resp = DocumentStatusResponse(
        document_id=doc_id,
        title="Test Paper",
        document_type=DocumentType.PAPER,
        status=DocumentStatus.READY,
        error_message=None,
        source_filename="test.pdf",
        created_at=now,
        updated_at=now,
    )
    data = resp.model_dump()
    restored = DocumentStatusResponse.model_validate(data)
    assert restored.document_id == doc_id
    assert restored.title == "Test Paper"
    assert restored.status == DocumentStatus.READY


def test_document_status_response_ignores_unknown_fields() -> None:
    """DocumentStatusResponse silently drops extra fields."""
    now = datetime.now(UTC)
    resp = DocumentStatusResponse.model_validate(
        {
            "document_id": str(uuid4()),
            "title": "Test",
            "document_type": "paper",
            "status": "validating",
            "error_message": None,
            "source_filename": "test.pdf",
            "created_at": now.isoformat(),
            "updated_at": now.isoformat(),
            "unexpected_field": "oops",
        }
    )
    assert resp.title == "Test"
    assert not hasattr(resp, "unexpected_field")


def test_document_status_response_allows_none_error_message() -> None:
    """error_message may be None for successful documents."""
    now = datetime.now(UTC)
    resp = DocumentStatusResponse(
        document_id=uuid4(),
        title="Clean",
        document_type=DocumentType.BOOK,
        status=DocumentStatus.READY,
        error_message=None,
        source_filename="book.pdf",
        created_at=now,
        updated_at=now,
    )
    assert resp.error_message is None
