"""Depth Dive package (Harness B).

Owns the passage-transform validation stage, the two-agent harness (framing +
assembly; stubbed for the MVP tracer bullet), the similarity-gate policy, and
the generative artifacts. Consumes ``core/retrieval/`` primitives (ADR-0019)
once retrieval participates, and never imports ``retrieval_qa`` (ADR-0011).
"""

from depth_dive.harness import run_dive

__all__ = ["run_dive"]
