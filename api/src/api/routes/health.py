"""Health-check route: GET /health."""

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from sqlalchemy import text

from core.database.connection import get_engine

router = APIRouter(tags=["health"])


@router.get("/health", response_model=None)
def health_check() -> dict[str, str] | JSONResponse:
    """Probe the database and return connectivity status.

    Returns ``{"status": "ok"}`` (200) on success, or
    ``{"status": "unhealthy", "detail": "<error>"}`` (503) on failure.
    """
    try:
        with get_engine().connect() as conn:
            conn.execute(text("SELECT 1"))
        return {"status": "ok"}
    except Exception as _exc:
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "detail": "Service unavailable"},
        )
