"""Tests for the FastAPI background-tasks glue layer."""

from unittest.mock import MagicMock
from uuid import uuid4

from ingestion.tasks import schedule_ingestion


def test_schedule_ingestion_adds_background_task() -> None:
    """schedule_ingestion registers _execute_ingestion_task on BackgroundTasks."""
    mock_bg = MagicMock()
    doc_id = uuid4()

    schedule_ingestion(
        background_tasks=mock_bg,
        document_id=doc_id,
        document_type="paper",
        source_filename="test.pdf",
        file_bytes=b"fake-pdf-bytes",
    )

    mock_bg.add_task.assert_called_once()
    args = mock_bg.add_task.call_args
    # First positional arg is the callable
    assert callable(args[0][0])
    # Remaining positional args match the arguments passed through
    assert args[0][1] == doc_id
    assert args[0][2] == "paper"
    assert args[0][3] == "test.pdf"
    assert args[0][4] == b"fake-pdf-bytes"


def test_schedule_ingestion_is_public_api() -> None:
    """schedule_ingestion is exposed in __all__."""
    from ingestion.tasks import __all__

    assert "schedule_ingestion" in __all__
