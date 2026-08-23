"""Render a validated scene graph to self-contained HTML (ADR-0022, ticket #306).

Stage 2 of the deterministic renderer: ``render_html`` turns a validated
``InteractiveAnimation`` scene graph into a single self-contained HTML document
that opens via ``file://`` with no further server calls. All markup, styling,
and the CSS-driven animation (step reveal, prediction reveal, carousel
transition) are code-owned; the model's text is escaped data, never markup.

The ``.scratch/depth-dive/interactive-animation-contract.html`` prototype is
the starting point, with its demo-only fields (top-level ``version`` and
``treatment``) dropped. Animation is CSS-driven: each step is a ``data-step``
state on the stage, element opacity and highlight transition via CSS, and
carousel segments slide in through a keyframe animation. The embedded player
only advances state and applies per-step text/value data — it never
interpolates opacity per frame.

Escaping is held by construction (no sanitizer): model text is HTML-escaped
where it appears in the body, and the scene-graph JSON embedded in a
``<script>`` tag has ``<``/``>``/``&`` escaped to ``\\u003c``/``\\u003e``/
``\\u0026`` so a ``</script>`` sequence can never close the block early.
"""

import html
import json

from core.types.depth_dive import ElementState, InteractiveAnimation, SceneElement, Viewport

_BASE_CSS = """\
:root {
  --depth-bg: #0f172a;
  --depth-panel: #1e293b;
  --depth-panel-2: #334155;
  --depth-text: #f8fafc;
  --depth-muted: #94a3b8;
  --depth-accent: #38bdf8;
  --depth-accent-2: #f472b6;
  --depth-ok: #34d399;
  --depth-warn: #fbbf24;
  --depth-font: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont,
    "Segoe UI", Roboto, sans-serif;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  font-family: var(--depth-font);
  background: var(--depth-bg);
  color: var(--depth-text);
  line-height: 1.5;
}
header { padding: 1.25rem 1.5rem; border-bottom: 1px solid var(--depth-panel-2); }
header h1 { margin: 0 0 0.25rem; font-size: 1.25rem; }
header p { margin: 0; color: var(--depth-muted); }
.layout {
  display: grid;
  grid-template-columns: 1fr 26rem;
  gap: 1rem;
  padding: 1rem;
  max-width: 100rem;
  margin: 0 auto;
}
.panel {
  background: var(--depth-panel);
  border-radius: 0.75rem;
  border: 1px solid var(--depth-panel-2);
  overflow: hidden;
  min-width: 0;
}
.panel h2 {
  margin: 0;
  padding: 0.75rem 1rem;
  font-size: 0.8rem;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  color: var(--depth-muted);
  border-bottom: 1px solid var(--depth-panel-2);
  background: rgba(255, 255, 255, 0.03);
}
.stage { position: relative; }
.scene-wrap { min-height: 20rem; }
svg {
  display: block;
  width: 100%;
  height: auto;
  min-height: 20rem;
}
.scene-element { transition: opacity 0.3s ease; }
.scene-element[data-type="token"] rect {
  fill: var(--depth-panel-2);
  stroke: var(--depth-panel-2);
  stroke-width: 1;
}
.scene-element[data-type="token"] text { fill: var(--depth-text); }
.scene-element[data-type="vector"] rect { fill: var(--depth-muted); }
.scene-element[data-type="score"] rect { fill: var(--depth-accent); }
.controls {
  display: flex;
  gap: 0.5rem;
  padding: 0.75rem 1rem;
  border-top: 1px solid var(--depth-panel-2);
  flex-wrap: wrap;
  align-items: center;
}
button {
  background: var(--depth-panel-2);
  color: var(--depth-text);
  border: 1px solid var(--depth-panel-2);
  border-radius: 0.375rem;
  padding: 0.4rem 0.75rem;
  font-family: inherit;
  font-size: 0.875rem;
  cursor: pointer;
}
button:hover { border-color: var(--depth-accent); }
button.primary {
  background: var(--depth-accent);
  color: #0f172a;
  border-color: var(--depth-accent);
  font-weight: 600;
}
.body { padding: 1rem; display: flex; flex-direction: column; gap: 0.75rem; }
.metric { font-size: 0.8rem; color: var(--depth-muted); }
.metric strong { color: var(--depth-text); }
.step-label { font-size: 0.9375rem; min-height: 3rem; color: var(--depth-text); }
.depth-dive-slide-in .scene-wrap {
  animation: depth-dive-slide 0.45s ease;
}
@keyframes depth-dive-slide {
  from { transform: translateX(24px); opacity: 0; }
  to { transform: none; opacity: 1; }
}
@media (max-width: 900px) { .layout { grid-template-columns: 1fr; } }
@media (max-width: 600px) {
  header { padding: 1rem; }
  .layout { padding: 0.5rem; }
}
"""

_PLAYER_JS = """\
(function () {
  "use strict";
  var scene = document.getElementById("depth-dive-scene");
  var data = JSON.parse(scene.textContent);
  var stage = document.getElementById("depth-dive-stage");
  var label = document.getElementById("depth-dive-step-label");
  var indexEl = document.getElementById("depth-dive-step-index");
  var countEl = document.getElementById("depth-dive-step-count");
  var progressEl = document.getElementById("depth-dive-progress");
  var slideEl = document.getElementById("depth-dive-slide");
  var steps = data.steps;
  var interactions = data.interactions || {};
  var segAttr = stage.getAttribute("data-segments");
  var segmentByIndex = segAttr ? segAttr.split(" ").map(Number) : [];
  var hasSegments = segmentByIndex.length > 0;
  var segmentOf = function (i) { return segmentByIndex[i] || 0; };
  var segmentCount = slideEl ? Number(slideEl.getAttribute("data-segment-count")) : 0;
  var nodes = {};
  var idx = 0;
  var timer = null;
  var totalMs = 0;
  var els = stage.querySelectorAll("[data-el]");
  var i;
  for (i = 0; i < els.length; i++) {
    nodes[els[i].getAttribute("data-el")] = els[i];
  }
  for (i = 0; i < steps.length; i++) {
    totalMs += steps[i].duration_ms || 1000;
  }
  function current() { return steps[idx]; }
  function applyState(elId, state) {
    var node = nodes[elId];
    if (!node) return;
    if (typeof state.text === "string") {
      var text = node.querySelector("[data-role=text]");
      if (text) text.textContent = state.text;
    }
    if (typeof state.value === "number") {
      var value = node.querySelector("[data-role=value]");
      if (value) value.textContent = state.value.toFixed(2);
      var bar = node.querySelector("[data-role=bar]");
      if (bar) {
        var w = Math.max(0, Math.min(1, state.value)) * 120;
        var cx = parseFloat(bar.getAttribute("x")) + parseFloat(bar.getAttribute("width")) / 2;
        bar.setAttribute("width", String(w));
        bar.setAttribute("x", String(cx - w / 2));
      }
    }
  }
  function applyStep() {
    stage.setAttribute("data-step", String(idx));
    label.textContent = current().label;
    indexEl.textContent = String(idx + 1);
    countEl.textContent = String(steps.length);
    var elapsed = 0;
    var i2;
    for (i2 = 0; i2 < idx; i2++) elapsed += steps[i2].duration_ms || 1000;
    progressEl.textContent = (idx === steps.length - 1)
      ? "100%"
      : (totalMs ? Math.round((elapsed / totalMs) * 100) : 0) + "%";
    if (slideEl) {
      slideEl.textContent = hasSegments
        ? "Slide " + (segmentOf(idx) + 1) + " of " + segmentCount
        : "";
    }
    var states = current().element_states;
    for (var id in states) {
      if (Object.prototype.hasOwnProperty.call(states, id)) applyState(id, states[id]);
    }
  }
  function triggerSlide() {
    stage.classList.remove("depth-dive-slide-in");
    void stage.offsetWidth;
    stage.classList.add("depth-dive-slide-in");
  }
  function go(nextIndex) {
    if (nextIndex < 0 || nextIndex > steps.length - 1) return;
    idx = nextIndex;
    applyStep();
    if (hasSegments && idx > 0 && segmentOf(idx) !== segmentOf(idx - 1)) {
      triggerSlide();
    }
  }
  function stop() { if (timer) { clearTimeout(timer); timer = null; } }
  function next() { stop(); go(idx + 1); }
  function prev() { stop(); go(idx - 1); }
  function reset() { stop(); go(0); }
  function play() {
    if (idx >= steps.length - 1) reset();
    if (timer) return;
    var tick = function () {
      if (idx >= steps.length - 1) { stop(); return; }
      go(idx + 1);
      timer = setTimeout(tick, steps[idx].duration_ms || 1000);
    };
    timer = setTimeout(tick, steps[idx].duration_ms || 1000);
  }
  document.getElementById("depth-dive-play").addEventListener("click", play);
  document.getElementById("depth-dive-pause").addEventListener("click", stop);
  document.getElementById("depth-dive-next").addEventListener("click", next);
  document.getElementById("depth-dive-prev").addEventListener("click", prev);
  document.getElementById("depth-dive-reset").addEventListener("click", reset);
  if (interactions.click_to_advance !== false) {
    stage.addEventListener("click", function () { if (timer) stop(); else next(); });
  }
  applyStep();
})();
"""


def render_html(animation: InteractiveAnimation) -> str:
    """Render a validated scene graph to self-contained HTML/CSS/JS.

    All markup, styling, and the CSS-driven animation (step reveal, prediction
    reveal, carousel transition) are code-owned; the model's text appears as
    escaped data, never markup. The output carries no external CSS/JS/font or
    image references, opens via ``file://``, adapts to screen size, and is
    byte-for-byte deterministic for a given scene graph.

    Args:
        animation: A validated ``InteractiveAnimation`` scene graph (spec §7).

    Returns:
        A complete HTML document as a single string.
    """
    title = _escape_text(animation.title)
    concept = _escape_text(animation.concept)
    first_step_label = _escape_text(animation.steps[0].label)
    viewport = animation.viewport
    scene_json = _serialize_scene(animation)
    elements_svg = "\n".join(_render_element(e, viewport) for e in animation.elements)
    segments_attr = (
        f' data-segments="{" ".join(str(i) for i in _segment_indexes(animation))}"'
        if _has_segments(animation)
        else ""
    )
    slide_row = (
        f'<div class="metric"><strong>Slide:</strong> <span id="depth-dive-slide" '
        f'data-segment-count="{_segment_count(animation)}"></span></div>\n'
        if _has_segments(animation)
        else ""
    )
    return "".join(
        [
            "<!DOCTYPE html>\n",
            '<html lang="en">\n',
            "<head>\n",
            '<meta charset="utf-8">\n',
            '<meta name="viewport" content="width=device-width, initial-scale=1">\n',
            f"<title>{title}</title>\n",
            "<style>\n",
            _BASE_CSS,
            "\n",
            _build_step_css(animation),
            "\n</style>\n",
            "</head>\n",
            "<body>\n",
            "<header>\n",
            f"<h1>{title}</h1>\n",
            f"<p>{concept}</p>\n",
            "</header>\n",
            '<main class="layout">\n',
            '<section class="panel">\n',
            "<h2>Animation</h2>\n",
            '<div class="stage" id="depth-dive-stage" data-step="0"' + segments_attr + ">\n",
            '<div class="scene-wrap">\n',
            f'<svg id="depth-dive-scene" viewBox="0 0 {viewport.width} {viewport.height}" '
            f'role="img" aria-label="{title}">\n',
            "<defs>\n",
            '<marker id="depth-dive-arrow" markerWidth="10" markerHeight="10" '
            'refX="8" refY="3" orient="auto" markerUnits="strokeWidth">\n',
            '<path d="M0,0 L0,6 L9,3 z" fill="var(--depth-muted)"></path>\n',
            "</defs>\n",
            elements_svg,
            "\n</svg>\n",
            "</div>\n",
            '<div class="controls">\n',
            '<button id="depth-dive-play" class="primary" type="button">Play</button>\n',
            '<button id="depth-dive-pause" type="button">Pause</button>\n',
            '<button id="depth-dive-prev" type="button">Step back</button>\n',
            '<button id="depth-dive-next" type="button">Step forward</button>\n',
            '<button id="depth-dive-reset" type="button">Reset</button>\n',
            "</div>\n",
            "</div>\n",
            "</section>\n",
            '<aside class="panel">\n',
            "<h2>Narration</h2>\n",
            '<div class="body">\n',
            '<div class="metric"><strong>Current step:</strong> '
            f'<span id="depth-dive-step-index">1</span> / '
            f'<span id="depth-dive-step-count">{len(animation.steps)}</span></div>\n',
            '<div class="metric"><strong>Progress:</strong> '
            '<span id="depth-dive-progress">0%</span></div>\n',
            slide_row,
            f'<div class="step-label" id="depth-dive-step-label">{first_step_label}</div>\n',
            "</div>\n",
            "</aside>\n",
            "</main>\n",
            f'<script type="application/json" id="depth-dive-scene">{scene_json}</script>\n',
            "<script>\n",
            _PLAYER_JS,
            "\n</script>\n",
            "</body>\n",
            "</html>\n",
        ]
    )


def _build_step_css(animation: InteractiveAnimation) -> str:
    """Emit the per-step CSS rules that drive opacity and highlight transitions.

    Each step becomes a ``[data-step=N]`` state on the stage; element opacity
    (and token highlight) for that step is a plain CSS rule, so stepping
    animates via CSS transitions. ``reveal_on_last_step`` elements are pinned
    to ``opacity: 0`` for every step but the last, encoding prediction reveal
    deterministically.
    """
    reveal: set[str] = set(animation.interactions.reveal_on_last_step)
    last_index = len(animation.steps) - 1
    rules: list[str] = []
    for step_index, step in enumerate(animation.steps):
        for element in animation.elements:
            state = _effective_state(
                animation.initial_state.get(element.id),
                step.element_states.get(element.id),
            )
            opacity = 1.0 if state.opacity is None else state.opacity
            if element.id in reveal and step_index < last_index:
                opacity = 0.0
            highlight = state.highlight
            if highlight is None:
                highlight = element.highlight
            base = (
                f'#depth-dive-stage[data-step="{step_index}"] [data-el={_css_string(element.id)}]'
            )
            rules.append(f"{base} {{ opacity: {_fmt(opacity)}; }}")
            if highlight is True:
                rules.append(
                    f"{base} rect, {base} line {{ fill: rgba(56,189,248,0.2); "
                    "stroke: var(--depth-accent); stroke-width: 2; }"
                )
    return "\n".join(rules)


def _render_element(element: SceneElement, viewport: Viewport) -> str:
    """Render one scene element as an SVG group; markup is code-owned.

    ``x``/``y`` are optional layout hints; an element without a position is
    centered in the viewport. ``arrow`` elements have no endpoint fields in the
    contract, so a code-owned horizontal arrow is drawn centered on ``x``/``y``
    with a length derived from ``value`` (the accepted ADR-0022 tradeoff: a
    template can only draw what its SVG is parameterized to draw).
    """
    x = viewport.width / 2 if element.x is None else element.x
    y = viewport.height / 2 if element.y is None else element.y
    if element.type == "text":
        style = element.style
        node = _svg_element(
            "text",
            {
                "x": _fmt(x),
                "y": _fmt(y),
                "data-role": "text",
                "text-anchor": style.textAnchor if style and style.textAnchor else "start",
                "font-size": _fmt(style.fontSize, 14) if style and style.fontSize else "14",
                "font-weight": style.fontWeight if style and style.fontWeight else "normal",
                "fill": style.fill if style and style.fill else "var(--depth-text)",
            },
            _escape_text(element.text or ""),
        )
    elif element.type == "token":
        box = _svg_element(
            "rect",
            {
                "x": _fmt(x - 36),
                "y": _fmt(y - 24),
                "width": "72",
                "height": "36",
                "rx": "8",
            },
        )
        label = _svg_element(
            "text",
            {
                "x": _fmt(x),
                "y": _fmt(y - 2),
                "data-role": "text",
                "text-anchor": "middle",
                "font-size": "15",
            },
            _escape_text(element.text or ""),
        )
        node = f"{box}\n  {label}"
    elif element.type == "vector":
        bar = _svg_element(
            "rect",
            {
                "x": _fmt(x - 40),
                "y": _fmt(y - 8),
                "width": "80",
                "height": "16",
                "rx": "4",
                "data-role": "bar",
                "fill": element.color or "var(--depth-muted)",
            },
        )
        label = _svg_element(
            "text",
            {
                "x": _fmt(x),
                "y": _fmt(y + 24),
                "data-role": "label",
                "text-anchor": "middle",
                "font-size": "11",
                "fill": "var(--depth-muted)",
            },
            _escape_text(element.label or ""),
        )
        node = f"{bar}\n  {label}"
    elif element.type == "score":
        value = 0.0 if element.value is None else element.value
        width = max(0.0, min(1.0, value)) * 120
        value_node = _svg_element(
            "text",
            {
                "x": _fmt(x),
                "y": _fmt(y),
                "data-role": "value",
                "text-anchor": "middle",
                "font-size": "13",
                "font-weight": "bold",
            },
            f"{value:.2f}",
        )
        bar = _svg_element(
            "rect",
            {
                "x": _fmt(x - width / 2),
                "y": _fmt(y + 8),
                "width": _fmt(width),
                "height": "6",
                "rx": "3",
                "data-role": "bar",
            },
        )
        node = f"{value_node}\n  {bar}"
    elif element.type == "arrow":
        length = 120 if element.value is None else max(40.0, min(200.0, element.value * 200))
        node = _svg_element(
            "line",
            {
                "x1": _fmt(x - length / 2),
                "y1": _fmt(y),
                "x2": _fmt(x + length / 2),
                "y2": _fmt(y),
                "data-role": "bar",
                "stroke": element.color or "var(--depth-muted)",
                "stroke-width": "2",
                "marker-end": "url(#depth-dive-arrow)",
            },
        )
    else:
        node = ""
    return (
        f'<g data-el="{_escape_attr(element.id)}" class="scene-element" '
        f'data-type="{element.type}">\n  {node}\n</g>'
    )


def _effective_state(base: ElementState | None, update: ElementState | None) -> ElementState:
    """Merge a step's sparse state over ``initial_state`` into an effective state."""
    return ElementState(
        opacity=_coalesce(update.opacity if update else None, base.opacity if base else None),
        highlight=_coalesce(update.highlight if update else None, base.highlight if base else None),
        value=_coalesce(update.value if update else None, base.value if base else None),
        text=_coalesce(update.text if update else None, base.text if base else None),
    )


def _coalesce[T](update: T | None, base: T | None) -> T | None:
    """Return ``update`` when set, otherwise ``base`` (sparse-state merge)."""
    return base if update is None else update


def _has_segments(animation: InteractiveAnimation) -> bool:
    """True when the animation carries segmented-carousel structure."""
    return bool(animation.interactions.segments) or any(
        step.segment is not None for step in animation.steps
    )


def _segment_indexes(animation: InteractiveAnimation) -> list[int]:
    """Compute the zero-based segment index of every step.

    ``InteractionHints.segments`` (the step ids that start a new slide) is the
    authoritative boundary list when present; otherwise ``AnimationStep.segment``
    supplies each step's index directly. The first step always starts segment
    zero, even when it is not named in ``segments``.
    """
    boundaries = list(animation.interactions.segments)
    if boundaries:
        indexes: list[int] = []
        current = -1
        for index, step in enumerate(animation.steps):
            if index == 0 or step.id in boundaries:
                current += 1
            indexes.append(current)
        return indexes
    return [0 if step.segment is None else step.segment for step in animation.steps]


def _segment_count(animation: InteractiveAnimation) -> int:
    """Return the number of carousel slides in the animation (0 when not one)."""
    if not _has_segments(animation):
        return 0
    return max(_segment_indexes(animation)) + 1


def _serialize_scene(animation: InteractiveAnimation) -> str:
    """Serialize the scene graph as JSON safe to embed inside a ``<script>`` tag.

    ``<``/``>``/``&`` are escaped to unicode escapes so model text can never
    close the script block (e.g. a literal ``</script>`` becomes
    ``\\u003c/script\\u003e``).
    """
    payload = json.dumps(
        animation.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return payload.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")


def _svg_element(tag: str, attrs: dict[str, str], content: str | None = None) -> str:
    """Emit an SVG element with escaped attribute values and optional content."""
    attr_text = _svg_attrs(attrs)
    if content is None:
        return f"<{tag} {attr_text}></{tag}>"
    return f"<{tag} {attr_text}>{content}</{tag}>"


def _svg_attrs(attrs: dict[str, str]) -> str:
    """Join a dict of SVG attributes into an escaped attribute string."""
    return " ".join(f'{key}="{_escape_attr(value)}"' for key, value in attrs.items())


def _fmt(value: int | float | None, default: int | float = 0) -> str:
    """Format a numeric layout hint as a compact, deterministic string."""
    if value is None:
        value = default
    if isinstance(value, int):
        return str(value)
    number = float(value)
    if number.is_integer():
        return str(int(number))
    return repr(number)


def _css_string(value: str) -> str:
    """Quote a value for use inside a CSS attribute selector."""
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _escape_text(value: str) -> str:
    """Escape model-provided text so it renders as data, never markup."""
    return html.escape(value, quote=True)


def _escape_attr(value: str) -> str:
    """Escape a value for safe use inside a double-quoted HTML attribute."""
    return html.escape(value, quote=True)


__all__ = ["render_html"]
