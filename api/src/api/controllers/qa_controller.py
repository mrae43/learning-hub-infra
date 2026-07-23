"""QA controller for Harness A's ``POST /query``.

Orchestrates the retrieve -> prompt-assemble -> inference flow hand-rolled
per ADR-0003:

1. Embed the user's query via the hosted embeddings client (ADR-0004).
2. Retrieve top-k chunks against the pgvector embeddings table, scoped to
   ``status='ready'`` documents (ADR-0014).
3. Assemble the Injected Context (CONTEXT.md) and call the hosted inference
   client (ADR-0001). When retrieval found nothing relevant, build a
   "no passages" prompt so the model emits a natural refusal — the not-found
   case is a valid response (``grounded=False``), not an exception (ADR-0009).

Upstream failures (embeddings or inference) propagate
``UpstreamBadResponse`` / ``UpstreamUnavailable`` for the route layer to map
to 502 / 503 (ADR-0014 § Error contract). DB-level failures propagate
verbatim; the route's catch-all maps them to 500.
"""

from sqlalchemy.orm import Session

from api.prompt import build_messages
from core.clients import CompletionProvider, Embedder
from core.types.responses import HarnessAResponse
from core.types.retrieval_config import RetrievalConfig
from retrieval_qa.retrieval.query import retrieve_relevant_chunks


def run_query(
    *,
    query: str,
    session: Session,
    embeddings_client: Embedder,
    llm_client: CompletionProvider,
    config: RetrievalConfig,
) -> HarnessAResponse:
    """Run the full Harness A query flow and return a ``HarnessAResponse``.

    Args:
        query: The user's bare query string (ADR-0014: no filters).
        session: SQLAlchemy session bound to the documents database. The
            retrieval step issues ``SET LOCAL hnsw.ef_search`` scoped to this
            session's transaction.
        embeddings_client: Provider used to embed the query (1536-dim).
        llm_client: Provider used to generate the answer.
        config: Retrieval configuration (model name, ef_search, top_k).

    Returns:
        A ``HarnessAResponse`` with ``answer`` always populated,
        ``cited_passages`` populated iff relevant chunks were found, and
        ``grounded`` reflecting whether retrieval produced chunks.

    Raises:
        UpstreamBadResponse: The embeddings or inference API returned an
            unexpected response (route maps to 502).
        UpstreamUnavailable: The embeddings or inference API was unreachable
            or timed out (route maps to 503).
    """
    query_vectors = embeddings_client.embed([query])
    query_vector = query_vectors[0]

    chunks = retrieve_relevant_chunks(
        query_vector=query_vector,
        session=session,
        config=config,
        query_text=query,
    )

    messages = build_messages(query, chunks)
    answer = llm_client.chat(messages)

    return HarnessAResponse(
        answer=answer,
        cited_passages=list(chunks),
        grounded=bool(chunks),
    )


__all__ = ["run_query"]
