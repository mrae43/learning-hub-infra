"""Assembly agent: grounds and builds the dive (ADR-0020, tickets #243, #244).

Consumes the :class:`~depth_dive.framing.framing_agent.FramingBrief` and
grounds the dive against the ingested corpus via the shared ``core/retrieval/``
primitives (ADR-0019): anchored ``text``/``code`` passages fetch the parent
chunk plus semantic neighbors, and every other passage runs the
corpus-similarity gate. The brief's ``search_intent`` drives the retry-once
web-search step (ADR-0013, ticket #244); the artifact payload stays the
hardcoded demo until LLM generation lands (ticket #245).
"""

from depth_dive.assembly.assembly_agent import AssemblyResult, run_assembly

__all__ = ["AssemblyResult", "run_assembly"]
