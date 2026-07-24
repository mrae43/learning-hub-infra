"""Tests for the FastAPI background-tasks glue layer."""

import threading
from unittest.mock import MagicMock, patch
from uuid import uuid4

from core.clients import InMemoryEmbedder
from core.types.document import DocumentType
from ingestion.models import PendingIngestion
from ingestion.tasks import schedule_ingestion

_SHARED_PENDING = PendingIngestion(
    document_id=uuid4(),
    title="Test",
    document_type=DocumentType.PAPER,
    source_filename="test.pdf",
    file_bytes=b"data",
)


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


# ── Concurrency-limiter tests ─────────────────────────────────────────────


def _make_mock_session() -> MagicMock:
    """Build a MagicMock session that behaves like one with a found document."""
    sess = MagicMock()
    doc = MagicMock()
    doc.title = "Test"
    sess.get.return_value = doc
    return sess


def test_ingestion_semaphore_initial_value() -> None:
    """The ingestion semaphore is initialised with count 2."""
    from ingestion.tasks import _INGESTION_SEMAPHORE

    assert isinstance(_INGESTION_SEMAPHORE, threading.Semaphore)
    assert _INGESTION_SEMAPHORE._value == 2


def test_semaphore_released_on_success() -> None:
    """_execute_ingestion_task releases the semaphore after a successful run."""
    from ingestion.tasks import _INGESTION_SEMAPHORE, _execute_ingestion_task

    with (
        patch("ingestion.tasks.get_session", return_value=_make_mock_session()),
        patch("ingestion.tasks.run_ingestion") as mock_run,
    ):
        initial = _INGESTION_SEMAPHORE._value
        _execute_ingestion_task(_SHARED_PENDING, InMemoryEmbedder(), "model")
        assert _INGESTION_SEMAPHORE._value == initial
        mock_run.assert_called_once()


def test_semaphore_released_on_failure() -> None:
    """_execute_ingestion_task releases the semaphore when the pipeline raises."""
    from ingestion.tasks import _INGESTION_SEMAPHORE, _execute_ingestion_task

    with (
        patch("ingestion.tasks.get_session", return_value=_make_mock_session()),
        patch("ingestion.tasks.run_ingestion", side_effect=ValueError("boom")),
        patch("ingestion.tasks.get_session") as mock_get_session,  # for failure handling
    ):
        mock_get_session.return_value = _make_mock_session()
        initial = _INGESTION_SEMAPHORE._value
        _execute_ingestion_task(_SHARED_PENDING, InMemoryEmbedder(), "model")
        assert _INGESTION_SEMAPHORE._value == initial


def test_semaphore_blocks_third_concurrent_call() -> None:
    """When both permits are held, a third acquire(blocking=False) returns False."""
    from ingestion.tasks import _INGESTION_SEMAPHORE

    acquired: list[bool] = []
    for _ in range(2):
        acquired.append(_INGESTION_SEMAPHORE.acquire(blocking=False))
    assert acquired == [True, True]

    blocked = _INGESTION_SEMAPHORE.acquire(blocking=False)
    assert blocked is False

    # Restore permits
    _INGESTION_SEMAPHORE.release()
    _INGESTION_SEMAPHORE.release()
