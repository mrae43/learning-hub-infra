"""Harness B entrypoint (ADR-0020).

Runs the Depth Dive pipeline for one ``HarnessBRequest`` and returns a
``HarnessBResponse``. The passage is validated by the Passage Transform stage,
then the framing agent resolves the treatment set and search intent (ADR-0020,
ticket #242). The assembly agent is still the hardcoded demo; no retrieval or
web search participates yet, so ``grounded`` is always ``False`` and the search
flags are off.
"""

from core.types.depth_dive import HarnessBRequest, HarnessBResponse
from depth_dive.framing.framing_agent import run_framing
from depth_dive.generation.demo_animation import build_demo_animation
from depth_dive.transform import transform_passage


def run_dive(request: HarnessBRequest) -> HarnessBResponse:
    """Validate a request and return the dive response.

    Args:
        request: The ``HarnessBRequest`` carried by ``POST /dive``.

    Returns:
        A ``HarnessBResponse`` whose ``output`` is the hardcoded demo
        ``interactive_animation`` scene graph, whose treatment fields come from
        the framing agent, and whose search flags stay off for the tracer
        bullet.

    Raises:
        PassageTransformError: The passage violates the declared size/bounds
            or is a type this harness does not yet support.
    """
    transform_passage(request.captured_passage)
    brief = run_framing(
        request.captured_passage,
        requested_treatments=request.requested_treatments,
        preferred_treatments=request.preferred_treatments,
    )
    return HarnessBResponse(
        output=build_demo_animation(),
        recommended_treatments=brief.recommended_treatments,
        applied_treatments=brief.applied_treatments,
        routing_note=brief.routing_note,
        grounded=False,
        external_search_attempted=False,
        external_search_failed=False,
        cited_passages=[],
    )


__all__ = ["run_dive"]
