"""Harness B entrypoint (ADR-0020).

Runs the Depth Dive pipeline for one Captured Passage and returns a
``HarnessBResponse``. The full two-agent pipeline (framing then assembly) is
deferred; the MVP tracer bullet validates the passage via the Passage Transform
stage and returns the hardcoded demo ``interactive_animation``. No retrieval,
web search, or per-learner state participates yet, so ``grounded`` is always
``False`` and the search flags are off.
"""

from core.types.captured_passage import CapturedPassage
from core.types.depth_dive import HarnessBResponse, Treatment
from depth_dive.generation.demo_animation import build_demo_animation
from depth_dive.transform import transform_passage


def run_dive(passage: CapturedPassage) -> HarnessBResponse:
    """Validate a Captured Passage and return the tracer-bullet dive response.

    Args:
        passage: The captured passage carried in the ``HarnessBRequest``.

    Returns:
        A ``HarnessBResponse`` whose ``output`` is the hardcoded demo
        ``interactive_animation`` scene graph.

    Raises:
        PassageTransformError: The passage violates the declared size/bounds
            or is a type this harness does not yet support.
    """
    transform_passage(passage)
    return HarnessBResponse(
        output=build_demo_animation(),
        recommended_treatments=[Treatment.WORKED_EXAMPLE],
        applied_treatments=[Treatment.WORKED_EXAMPLE],
        grounded=False,
        external_search_attempted=False,
        external_search_failed=False,
        cited_passages=[],
    )


__all__ = ["run_dive"]
