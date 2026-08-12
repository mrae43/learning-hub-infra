"""Hardcoded demo artifacts for the MVP tracer bullet.

Replaces the assembly agent (ADR-0020) until real generation lands: every
``POST /dive`` returns the same self-attention ``interactive_animation`` scene
graph, proving the contract, type wiring, and passage-transform validation
end-to-end.
"""

from depth_dive.generation.demo_animation import build_demo_animation

__all__ = ["build_demo_animation"]
