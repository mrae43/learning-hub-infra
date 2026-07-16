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

from collections.abc import Sequence

from sqlalchemy.orm import Session

from core.clients.embeddings_client import EmbeddingsClient
from core.clients.llm_client import LLMClient
from core.types.chat import ChatMessage
from core.types.responses import CitedPassage, HarnessAResponse
from retrieval_qa.retrieval.query import retrieve_relevant_chunks

_SYSTEM_PROMPT = (
    "You answer the user's question using only the provided passages. "
    "If no passages are provided, or if the answer is not contained in them, "
    "say you cannot answer from the current corpus. Do not invent information."
)


def _build_messages(
    query: str,
    chunks: Sequence[CitedPassage],
) -> list[ChatMessage]:
    """Assemble the chat-completions message list for the inference call.

    Args:
        query: The user's raw query string.
        chunks: Retrieved chunks (objects with a ``text`` attribute). Empty
            for the not-found branch, in which case the user message carries
            the query alone (no fake passages) so the model emits a refusal.

    Returns:
        A two-message list: a system message setting the grounding contract,
        and a user message containing passages (when present) plus the query.
    """
    user_parts: list[str] = []
    if chunks:
        passages = "\n\n".join(f"[{i}] {chunk.text}" for i, chunk in enumerate(chunks, start=1))
        user_parts.append("Passages:\n")
        user_parts.append(passages)
    user_parts.append(f"Question: {query}")
    user_message = "\n\n".join(p for p in user_parts if p)
    return [
        ChatMessage(role="system", content=_SYSTEM_PROMPT),
        ChatMessage(role="user", content=user_message),
    ]


def run_query(
    *,
    query: str,
    session: Session,
    embeddings_client: EmbeddingsClient,
    llm_client: LLMClient,
    model_name: str,
    ef_search: int,
    top_k: int,
) -> HarnessAResponse:
    """Run the full Harness A query flow and return a ``HarnessAResponse``.

    Args:
        query: The user's bare query string (ADR-0014: no filters).
        session: SQLAlchemy session bound to the documents database. The
            retrieval step issues ``SET LOCAL hnsw.ef_search`` scoped to this
            session's transaction.
        embeddings_client: Client used to embed the query (1536-dim).
        llm_client: Client used to generate the answer.
        model_name: Active embedding model name (provenance filter on
            ``embeddings.model_name``).
        ef_search: HNSW query-time candidate-list size.
        top_k: Maximum number of chunks to retrieve.

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
        model_name=model_name,
        ef_search=ef_search,
        top_k=top_k,
    )

    messages = _build_messages(query, chunks)
    answer = llm_client.chat(messages)

    return HarnessAResponse(
        answer=answer,
        cited_passages=list(chunks),
        grounded=bool(chunks),
    )


__all__ = ["run_query"]
