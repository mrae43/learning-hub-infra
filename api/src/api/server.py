"""FastAPI application factory."""

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from api.routes.documents import router as documents_router
from api.routes.ingest import router as ingest_router
from api.routes.retrieval_qa import router as query_router
from core.exceptions import UpstreamBadResponse, UpstreamUnavailable


def _retrieval_error_handler(request: Request, exc: Exception, status_code: int) -> JSONResponse:
    """Map a RetrievalError subclass to a 502/503 `{detail: ...}` body.

    Per ADR-0014 the error envelope reuses FastAPI's default ``{"detail": str}``
    shape (via HTTPException) rather than a typed ErrorResponse model.
    """
    _ = request  # unused; handler signature tracks Starlette's contract.
    return JSONResponse(status_code=status_code, content={"detail": str(exc)})


def create_app() -> FastAPI:
    """Create and configure the Learning Hub API."""
    app = FastAPI(title="Learning Hub", version="0.1.0")
    app.include_router(ingest_router)
    app.include_router(documents_router)
    app.include_router(query_router)

    app.add_exception_handler(
        UpstreamBadResponse,
        lambda request, exc: _retrieval_error_handler(request, exc, status_code=502),
    )
    app.add_exception_handler(
        UpstreamUnavailable,
        lambda request, exc: _retrieval_error_handler(request, exc, status_code=503),
    )
    # Catch-all: FastAPI already returns 500 for unexpected errors; this
    # explicit handler normalizes the body to the same {detail: ...} shape
    # used by the upstream handlers above.
    app.add_exception_handler(
        Exception,
        lambda request, exc: (
            _retrieval_error_handler(request, exc, status_code=500)
            if not isinstance(exc, HTTPException)
            else JSONResponse(
                status_code=exc.status_code,
                content={"detail": exc.detail},
            )
        ),
    )
    return app
