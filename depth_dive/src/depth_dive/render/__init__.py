"""Deterministic renderer for the interactive-animation scene graph (ADR-0022).

Two pure, side-effect-free stages form the public interface:
``build_scene_graph`` (stage 1, semantic spec -> scene graph) and
``render_html`` (stage 2, scene graph -> self-contained HTML). Stage 1 is a
separate work item (ticket #307); this package currently exposes only the HTML
render stage, a pure library function for the ADR-0023 plugin to invoke.
"""

from depth_dive.render.html_renderer import render_html

__all__ = ["render_html"]
