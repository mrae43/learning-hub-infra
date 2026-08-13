"""Harness B request/response and interactive-animation models (depth-dive spec §7-§9).

``HarnessBRequest`` and ``HarnessBResponse`` are the ``POST /dive`` contract.
``InteractiveAnimation`` is the MVP output: a declarative, stateless scene graph
the client renders without further server calls (dual coding).
"""

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from core.types.captured_passage import CapturedPassage
from core.types.responses import CitedPassage


class Treatment(StrEnum):
    """MVP pedagogical patterns layered onto an interactive animation (spec §3).

    These are the treatments the harness can actually produce. The deferred
    treatments live in :class:`DeferredTreatment` so a request can name them
    and be routed (with a ``routing_note``) rather than rejected.
    """

    WORKED_EXAMPLE = "worked_example"
    PREDICTION_REVEAL = "prediction_reveal"
    SEGMENTED_CAROUSEL = "segmented_carousel"


class DeferredTreatment(StrEnum):
    """Eval-gated treatments that are deferred from the MVP type set (spec §3).

    Modeled separately from :class:`Treatment` so clients can still name them
    in a request hint; the framing agent drops them with a ``routing_note`` and
    they never appear in the applied/recommended response fields.
    """

    ANALOGY_MAPPING = "analogy_mapping"
    ELABORATIVE_PROMPT = "elaborative_prompt"
    INTERACTIVE_CONCEPT_MAP = "interactive_concept_map"


TreatmentHint = Treatment | DeferredTreatment
"""Any treatment name a request may carry, MVP supported or deferred."""


class Viewport(BaseModel):
    """Nominal scene dimensions.

    Screen-size independence is the client's job; ``width``/``height`` are the
    design-time coordinate space (prototype: 800 x 520).
    """

    model_config = ConfigDict(extra="forbid")

    width: int = Field(gt=0)
    height: int = Field(gt=0)


class ElementStyle(BaseModel):
    """Optional presentation hints for a scene element."""

    model_config = ConfigDict(extra="forbid")

    fontSize: float | None = None
    fontWeight: str | None = None
    textAnchor: str | None = None
    fill: str | None = None


class SceneElement(BaseModel):
    """A persistent scene primitive with a stable ID.

    Type-specific payload lives in the optional fields (``text`` for
    ``text``/``token``, ``label`` + ``color`` for ``vector``, ``value`` for
    ``score``, ``style`` for text rendering). The exact primitive set is
    finalized during implementation (spec §7).
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    type: Literal["text", "token", "vector", "score", "arrow", "group"]
    x: float
    y: float
    text: str | None = None
    label: str | None = None
    color: str | None = None
    value: float | None = None
    highlight: bool = False
    style: ElementStyle | None = None


class ElementState(BaseModel):
    """A sparse per-element mutation applied by a step.

    Fields are all optional so a step may update only what changes
    (opacity, highlight, value, text).
    """

    model_config = ConfigDict(extra="forbid")

    opacity: float | None = None
    highlight: bool | None = None
    value: float | None = None
    text: str | None = None


class AnimationStep(BaseModel):
    """One ordered state in the animation, mutating ``element_states`` by ID."""

    model_config = ConfigDict(extra="forbid")

    id: str
    label: str
    duration_ms: int | None = Field(default=None, gt=0)
    segment: int | None = None
    element_states: dict[str, ElementState]


class InteractionHints(BaseModel):
    """Optional interaction affordances the client may expose."""

    model_config = ConfigDict(extra="forbid")

    click_to_advance: bool = True
    reveal_on_last_step: list[str] = Field(default_factory=list)
    segments: list[str] = Field(default_factory=list)


class InteractiveAnimation(BaseModel):
    """The ``interactive_animation`` output contract (spec §7).

    A declarative, stateless scene graph: persistent ``elements`` plus ordered
    ``steps`` that mutate ``element_states`` by element ID, seeded from
    ``initial_state``. The entire artifact ships in the response.
    """

    model_config = ConfigDict(extra="forbid")

    output_type: Literal["interactive_animation"] = "interactive_animation"
    title: str
    concept: str
    viewport: Viewport
    elements: list[SceneElement] = Field(min_length=1)
    steps: list[AnimationStep] = Field(min_length=1)
    initial_state: dict[str, ElementState]
    interactions: InteractionHints = InteractionHints()


class HarnessBRequest(BaseModel):
    """Request body for ``POST /dive`` (spec §9)."""

    model_config = ConfigDict(extra="forbid")

    captured_passage: CapturedPassage
    requested_output_type: Literal["interactive_animation"] | None = None
    requested_treatments: list[TreatmentHint] | None = None
    preferred_treatments: list[TreatmentHint] | None = None
    client_passage_id: str | None = None


class HarnessBResponse(BaseModel):
    """Response body for ``POST /dive`` (spec §9)."""

    model_config = ConfigDict(extra="forbid")

    output: InteractiveAnimation
    recommended_treatments: list[Treatment]
    applied_treatments: list[Treatment]
    routing_note: str | None = None
    grounded: bool
    external_search_attempted: bool
    external_search_failed: bool
    external_search_note: str | None = None
    cited_passages: list[CitedPassage]


__all__ = [
    "AnimationStep",
    "DeferredTreatment",
    "ElementState",
    "ElementStyle",
    "HarnessBRequest",
    "HarnessBResponse",
    "InteractionHints",
    "InteractiveAnimation",
    "SceneElement",
    "Treatment",
    "TreatmentHint",
    "Viewport",
]
