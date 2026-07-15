"""FastAPI application factory."""

from fastapi import FastAPI

from api.routes.documents import router as documents_router
from api.routes.ingest import router as ingest_router


def create_app() -> FastAPI:
    """Create and configure the Learning Hub API."""
    app = FastAPI(title="Learning Hub", version="0.1.0")
    app.include_router(ingest_router)
    app.include_router(documents_router)
    return app
