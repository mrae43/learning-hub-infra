"""Harness B entrypoint (ADR-0020).

Runs the Depth Dive pipeline for one ``HarnessBRequest`` and returns a
``HarnessBResponse``. The passage is validated by the Passage Transform stage
(whose model-ready carrier is forwarded to the generation turn), the framing
agent resolves the treatment set and search intent (ticket #242), and the
assembly agent grounds the passage against the ingested corpus via the shared
``core/retrieval/`` primitives (ticket #243), runs the retry-once web-search
step when the brief carries a ``search_intent`` (ticket #244), and runs the
LLM generation turn that produces the ``interactive_animation`` scene graph
(ticket #245).
"""

from sqlalchemy.orm import Session

from core.clients import CompletionProvider, Embedder
from core.types.depth_dive import HarnessBRequest, HarnessBResponse
from core.types.retrieval_config import RetrievalConfig
from depth_dive.assembly.assembly_agent import run_assembly
from depth_dive.framing.framing_agent import run_framing
from depth_dive.transform import transform_passage
from depth_dive.web_search.client import WebSearchClient


def run_dive(
    request: HarnessBRequest,
    *,
    session: Session,
    embedder: Embedder,
    config: RetrievalConfig,
    web_search: WebSearchClient,
    completion_provider: CompletionProvider,
) -> HarnessBResponse:
    """Validate a request and return the dive response.

    Args:
        request: The ``HarnessBRequest`` carried by ``POST /dive``.
        session: SQLAlchemy session bound to the documents database, used by
            the assembly agent's corpus grounding.
        embedder: Provider used to embed the passage for grounding
            (1536-dim).
        config: Retrieval configuration (model name, ef_search, top_k).
        web_search: Web-search provider used by the assembly agent when the
            framing brief carries a ``search_intent`` (ADR-0012, ADR-0013).
        completion_provider: Chat-completions provider used by the assembly
            agent's LLM generation turn (ticket #245).

    Returns:
        A ``HarnessBResponse`` whose ``output`` is the LLM-generated
        ``interactive_animation`` scene graph (or the minimal fallback scene
        graph when model output was malformed), whose treatment fields come
        from the framing agent, whose ``grounded``/``cited_passages`` come
        from the assembly agent's corpus grounding, and whose
        ``external_search_*`` fields come from the web-search step.

    Raises:
        PassageTransformError: The passage violates the declared size/bounds
            or is a type this harness does not yet support.
        UpstreamBadResponse: The embeddings API or the inference API returned
            an unexpected response during grounding/generation (route maps to
            502).
        UpstreamUnavailable: The embeddings API, the database, or the
            inference API was unreachable during grounding/generation (route
            maps to 503).
    """
    carrier = transform_passage(request.captured_passage)
    brief = run_framing(
        request.captured_passage,
        requested_treatments=request.requested_treatments,
        preferred_treatments=request.preferred_treatments,
        requested_output_type=request.requested_output_type,
    )
    assembly = run_assembly(
        request.captured_passage,
        brief,
        carrier,
        session=session,
        embedder=embedder,
        config=config,
        web_search=web_search,
        completion_provider=completion_provider,
    )
    return HarnessBResponse(
        output=assembly.animation,
        recommended_treatments=brief.recommended_treatments,
        applied_treatments=brief.applied_treatments,
        routing_note=brief.routing_note,
        grounded=assembly.grounded,
        external_search_attempted=assembly.external_search_attempted,
        external_search_failed=assembly.external_search_failed,
        external_search_note=assembly.external_search_note,
        cited_passages=assembly.cited_passages,
    )


__all__ = ["run_dive"]
