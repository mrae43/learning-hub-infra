"""Framing agent: emits the internal creative brief (ADR-0020, ticket #242).

The framing agent consumes a ``CapturedPassage`` plus optional request hints,
decides whether external grounding (web search) would help this specific dive
(ADR-0012), and resolves the final treatment set through the precedence rules
(explicit ask > preferred > harness recommendation). The brief it emits stays
internal and is not exposed in ``HarnessBResponse``.
"""

from depth_dive.framing.framing_agent import FramingBrief, run_framing

__all__ = ["FramingBrief", "run_framing"]
