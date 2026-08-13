"""Depth Dive route: POST /dive (depth-dive spec §9)."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from api.dependencies import (
    get_completion_provider,
    get_embedder,
    get_web_search_client,
)
from core.clients import CompletionProvider, Embedder
from core.config.settings import settings
from core.database.connection import db_session
from core.types.depth_dive import HarnessBRequest, HarnessBResponse
from core.types.retrieval_config import RetrievalConfig
from depth_dive.harness import run_dive
from depth_dive.transform import PassageTransformError
from depth_dive.web_search.client import WebSearchClient

router = APIRouter(tags=["dive"])


@router.post("/dive", response_model=HarnessBResponse)
def dive(
    body: HarnessBRequest,
    embeddings_client: Annotated[Embedder, Depends(get_embedder)],
    web_search: Annotated[WebSearchClient, Depends(get_web_search_client)],
    completion_provider: Annotated[CompletionProvider, Depends(get_completion_provider)],
) -> HarnessBResponse:
    """Generate a Depth Dive interactive animation for a Captured Passage.

    Runs the framing + assembly pipeline (ADR-0020): the assembly agent
    grounds the passage against the ingested corpus, populates
    ``grounded``/``cited_passages``, runs the retry-once web-search step
    when the brief carries a ``search_intent`` (ADR-0013) — surfacing the
    outcome on ``external_search_*`` — and runs the LLM generation turn that
    produces the ``interactive_animation`` scene graph (ticket #245). Returns
    200 with a ``HarnessBResponse``. Passages that violate the declared
    size/bounds are rejected with 422 (``PassageTransformError``). Upstream
    failures map to 502 / 503 via the ``RetrievalError`` subclass handlers
    registered on the app; malformed request bodies return 422 via FastAPI
    defaults. A failed web search or malformed model output is a valid
    degraded response (note + flags / fallback scene graph), never an error.
    """
    try:
        with db_session() as session:
            return run_dive(
                body,
                session=session,
                embedder=embeddings_client,
                config=RetrievalConfig(
                    model_name=settings.embedding_model,
                    ef_search=settings.hnsw_ef_search,
                    top_k=settings.query_top_k,
                ),
                web_search=web_search,
                completion_provider=completion_provider,
            )
    except PassageTransformError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


__all__ = ["router"]
