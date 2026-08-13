"""LLM-driven artifact generation for the Depth Dive assembly agent (ticket #245).

The generation turn builds a prompt from the framing brief, the cited corpus
passages, and any web-search results, calls the hosted inference API, and
parses the response into a validated ``interactive_animation`` scene graph.
Malformed model output falls back to a minimal valid scene graph carrying a
note instead of failing the dive.
"""

from depth_dive.generation.fallback_animation import build_fallback_animation
from depth_dive.generation.generation_agent import (
    MALFORMED_OUTPUT_NOTE,
    SYSTEM_PROMPT,
    GenerationResult,
    build_generation_prompt,
    parse_animation,
    run_generation,
)

__all__ = [
    "MALFORMED_OUTPUT_NOTE",
    "SYSTEM_PROMPT",
    "GenerationResult",
    "build_fallback_animation",
    "build_generation_prompt",
    "parse_animation",
    "run_generation",
]
