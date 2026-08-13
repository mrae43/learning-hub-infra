"""Assembly agent: grounds and builds the dive (ADR-0020, ticket #243).

Consumes the :class:`~depth_dive.framing.framing_agent.FramingBrief` and
grounds the dive against the ingested corpus via the shared ``core/retrieval/``
primitives (ADR-0019): anchored ``text``/``code`` passages fetch the parent
chunk plus semantic neighbors, and every other passage runs the
corpus-similarity gate. The brief's ``search_intent`` drives the web-search
step once it lands (ticket #244); the artifact payload stays the hardcoded
demo until LLM generation lands (ticket #245).
"""

from depth_dive.assembly.assembly_agent import GroundingResult, run_assembly

__all__ = ["GroundingResult", "run_assembly"]
