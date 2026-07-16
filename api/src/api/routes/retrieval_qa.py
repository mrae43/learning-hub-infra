"""Query route: POST /query (ADR-0014)."""

from fastapi import APIRouter

from api.controllers.qa_controller import run_query
from core.clients.embeddings_client import EmbeddingsClient
from core.clients.llm_client import LLMClient
from core.config.settings import settings
from core.database.connection import db_session
from core.types.responses import HarnessARequest, HarnessAResponse
from core.types.retrieval_config import RetrievalConfig

router = APIRouter(tags=["query"])


@router.post("/query", response_model=HarnessAResponse)
def query(body: HarnessARequest) -> HarnessAResponse:
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
            embeddings_client=EmbeddingsClient(
                api_key=settings.openai_api_key,
                model=settings.embedding_model,
            ),
            llm_client=LLMClient(
                api_key=settings.openai_api_key,
                model=settings.inference_model,
            ),
            config=RetrievalConfig(
                model_name=settings.embedding_model,
                ef_search=settings.hnsw_ef_search,
                top_k=settings.query_top_k,
            ),
        )
