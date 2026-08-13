"""Hardcoded demo artifacts for the MVP tracer bullet.

Stands in for the LLM-driven artifact generation inside the assembly agent
(ticket #245): every ``POST /dive`` returns the same self-attention
``interactive_animation`` scene graph while corpus grounding is already live.
"""

from depth_dive.generation.demo_animation import build_demo_animation

__all__ = ["build_demo_animation"]
