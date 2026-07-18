"""Query route: POST /query (ADR-0014)."""

from typing import Annotated

from fastapi import APIRouter, Depends

from api.controllers.qa_controller import run_query
from api.dependencies import get_completion_provider, get_embedder
from core.clients import CompletionProvider, Embedder
from core.config.settings import settings
from core.database.connection import db_session
from core.types.responses import HarnessARequest, HarnessAResponse
from core.types.retrieval_config import RetrievalConfig

router = APIRouter(tags=["query"])


@router.post("/query", response_model=HarnessAResponse)
def query(
    body: HarnessARequest,
    embeddings_client: Annotated[Embedder, Depends(get_embedder)],
    llm_client: Annotated[CompletionProvider, Depends(get_completion_provider)],
) -> HarnessAResponse:
    """Answer a query against the ingested corpus.

    Returns 200 with a ``HarnessAResponse`` on both grounded and not-found
    branches (ADR-0009: the not-found case is a valid response, not an
    exception). Upstream API failures surface as 502 / 503 via the
    ``RetrievalError`` subclass handlers registered in the app; missing
    or empty ``query`` returns 422 via FastAPI defaults.
    """
    with db_session() as session:
        return run_query(
            query=body.query,
            session=session,
            embeddings_client=embeddings_client,
            llm_client=llm_client,
            config=RetrievalConfig(
                model_name=settings.embedding_model,
                ef_search=settings.hnsw_ef_search,
                top_k=settings.query_top_k,
            ),
        )
