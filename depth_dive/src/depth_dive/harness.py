"""Harness B entrypoint (ADR-0020).

Runs the Depth Dive pipeline for one ``HarnessBRequest`` and returns a
``HarnessBResponse``. The passage is validated by the Passage Transform stage,
the framing agent resolves the treatment set and search intent (ticket #242),
and the assembly agent grounds the passage against the ingested corpus via the
shared ``core/retrieval/`` primitives (ticket #243). Web search does not
participate yet (ticket #244), so the search flags stay off; the artifact
payload stays the hardcoded demo until LLM generation lands (ticket #245).
"""

from sqlalchemy.orm import Session

from core.clients import Embedder
from core.types.depth_dive import HarnessBRequest, HarnessBResponse
from core.types.retrieval_config import RetrievalConfig
from depth_dive.assembly.assembly_agent import run_assembly
from depth_dive.framing.framing_agent import run_framing
from depth_dive.generation.demo_animation import build_demo_animation
from depth_dive.transform import transform_passage


def run_dive(
    request: HarnessBRequest,
    *,
    session: Session,
    embedder: Embedder,
    config: RetrievalConfig,
) -> HarnessBResponse:
    """Validate a request and return the dive response.

    Args:
        request: The ``HarnessBRequest`` carried by ``POST /dive``.
        session: SQLAlchemy session bound to the documents database, used by
            the assembly agent's corpus grounding.
        embedder: Provider used to embed the passage for grounding
            (1536-dim).
        config: Retrieval configuration (model name, ef_search, top_k).

    Returns:
        A ``HarnessBResponse`` whose ``output`` is the hardcoded demo
        ``interactive_animation`` scene graph, whose treatment fields come
        from the framing agent, and whose ``grounded``/``cited_passages``
        come from the assembly agent's corpus grounding. The search flags
        stay off until the web-search step lands (ticket #244).

    Raises:
        PassageTransformError: The passage violates the declared size/bounds
            or is a type this harness does not yet support.
        UpstreamBadResponse: The embeddings API returned an unexpected
            response during grounding (route maps to 502).
        UpstreamUnavailable: The embeddings API or the database was
            unreachable during grounding (route maps to 503).
    """
    transform_passage(request.captured_passage)
    brief = run_framing(
        request.captured_passage,
        requested_treatments=request.requested_treatments,
        preferred_treatments=request.preferred_treatments,
    )
    grounding = run_assembly(
        request.captured_passage,
        brief,
        session=session,
        embedder=embedder,
        config=config,
    )
    return HarnessBResponse(
        output=build_demo_animation(),
        recommended_treatments=brief.recommended_treatments,
        applied_treatments=brief.applied_treatments,
        routing_note=brief.routing_note,
        grounded=grounding.grounded,
        external_search_attempted=False,
        external_search_failed=False,
        cited_passages=grounding.cited_passages,
    )


__all__ = ["run_dive"]
