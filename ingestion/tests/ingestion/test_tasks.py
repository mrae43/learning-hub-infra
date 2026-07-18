"""Tests for the FastAPI background-tasks glue layer."""

from unittest.mock import MagicMock
from uuid import uuid4

from core.clients import InMemoryEmbedder
from core.types.document import DocumentType
from ingestion.models import PendingIngestion
from ingestion.tasks import schedule_ingestion


def test_schedule_ingestion_adds_background_task() -> None:
    """schedule_ingestion registers _execute_ingestion_task on BackgroundTasks."""
    mock_bg = MagicMock()
    doc_id = uuid4()
    embedder = InMemoryEmbedder()
    model_name = "text-embedding-3-small"

    pending = PendingIngestion(
        document_id=doc_id,
        title="Test Document",
        document_type=DocumentType.PAPER,
        source_filename="test.pdf",
        file_bytes=b"fake-pdf-bytes",
    )

    schedule_ingestion(
        background_tasks=mock_bg,
        pending=pending,
        embedder=embedder,
        model_name=model_name,
    )

    mock_bg.add_task.assert_called_once()
    args = mock_bg.add_task.call_args
    # First positional arg is the callable
    assert callable(args[0][0])
    # Second positional arg is the PendingIngestion model
    assert args[0][1] is pending
    # Remaining positional args match infrastructure args
    assert args[0][2] is embedder
    assert args[0][3] == model_name


def test_schedule_ingestion_is_public_api() -> None:
    """schedule_ingestion is exposed in __all__."""
    from ingestion.tasks import __all__

    assert "schedule_ingestion" in __all__
