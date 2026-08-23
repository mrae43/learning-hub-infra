"""Bounded semantic-spec model the generation turn emits (ADR-0022, ticket #305).

The model emits a semantic spec instead of a full scene graph: content and
relations only, with no geometry (no coordinates, viewport, or element states).
The spec is a harness-internal artifact, like :class:`FramingBrief` — it is
never serialized onto ``HarnessBResponse``; the deterministic renderer consumes
it to build the ``InteractiveAnimation`` scene graph.

Boundedness is the point: narration, token, and relation counts are capped,
token ids must be unique and non-empty, relation weights stay within
``[0, 1]``, and every relation endpoint must reference a declared token id.
A payload violating any rule is malformed — :func:`parse_semantic_spec`
returns ``None`` so the generation turn falls back gracefully rather than
producing a broken scene graph.
"""

import json
import re
from collections.abc import Mapping

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

# Boundedness caps (schema decision; the rules, not the concrete numbers, are
# the point). Keep narration small (one entry per animation step), tokens and
# relations bounded so the renderer's layout stays legible.
NARRATION_MAX_ENTRIES = 12
"""Maximum number of narration entries (one per animation step)."""

NARRATION_ENTRY_MAX_CHARS = 200
"""Maximum characters per narration entry."""

TOKENS_MAX_COUNT = 20
"""Maximum number of tokens the visual may center on."""

RELATIONS_MAX_COUNT = 40
"""Maximum number of weighted relations between tokens."""


class SemanticSpecToken(BaseModel):
    """A concept token/vector the visual centers on (id + label)."""

    model_config = ConfigDict(extra="forbid")

    id: str
    label: str


class SemanticSpecRelation(BaseModel):
    """A weighted directed relation between two declared tokens."""

    model_config = ConfigDict(extra="forbid")

    from_: str = Field(alias="from")
    to: str
    weight: float = Field(ge=0.0, le=1.0)


class SemanticSpec(BaseModel):
    """The bounded semantic spec the generation turn's model emits.

    Carries content and relations only — ``title``, ``concept``, an ordered
    ``narration`` list, ``tokens`` (id + label), and weighted directed
    ``relations`` (from / to / weight) — and no coordinates, viewport, or
    element states. Boundedness rules are enforced at validation time; any
    violation makes the spec malformed.
    """

    model_config = ConfigDict(extra="forbid")

    title: str
    concept: str
    narration: list[str] = Field(max_length=NARRATION_MAX_ENTRIES)
    tokens: list[SemanticSpecToken] = Field(max_length=TOKENS_MAX_COUNT)
    relations: list[SemanticSpecRelation] = Field(max_length=RELATIONS_MAX_COUNT)

    @field_validator("narration")
    @classmethod
    def _narration_entries_are_within_the_character_cap(cls, narration: list[str]) -> list[str]:
        """Reject a narration entry longer than the per-entry cap."""
        for entry in narration:
            if len(entry) > NARRATION_ENTRY_MAX_CHARS:
                raise ValueError(f"narration entry exceeds {NARRATION_ENTRY_MAX_CHARS} characters")
        return narration

    @field_validator("tokens")
    @classmethod
    def _token_ids_are_unique_and_non_empty(
        cls, tokens: list[SemanticSpecToken]
    ) -> list[SemanticSpecToken]:
        """Reject empty or duplicated token ids."""
        seen: set[str] = set()
        for token in tokens:
            if not token.id:
                raise ValueError("token id must be non-empty")
            if token.id in seen:
                raise ValueError(f"duplicate token id {token.id!r}")
            seen.add(token.id)
        return tokens

    @model_validator(mode="after")
    def _relation_endpoints_reference_declared_tokens(self) -> "SemanticSpec":
        """Reject a relation endpoint that is not a declared token id."""
        declared = {token.id for token in self.tokens}
        for relation in self.relations:
            if relation.from_ not in declared or relation.to not in declared:
                raise ValueError(
                    "relation endpoint references an undeclared token id "
                    f"({relation.from_!r} -> {relation.to!r})"
                )
        return self


def parse_semantic_spec(raw: str | Mapping[str, object]) -> SemanticSpec | None:
    """Parse and validate a model response into a bounded semantic spec.

    Tolerates a raw JSON object, one wrapped in markdown code fences, or an
    already-decoded mapping. Returns ``None`` when the payload is not valid
    JSON, does not satisfy the ``SemanticSpec`` contract, or violates a
    boundedness rule (oversized narration, duplicate or empty token id,
    out-of-range weight, or an undeclared relation endpoint) — so malformed
    specs fall back gracefully rather than producing broken scene graphs.

    Args:
        raw: The model's message content.

    Returns:
        The validated ``SemanticSpec``, or ``None`` for malformed output.
    """
    try:
        data = raw if isinstance(raw, Mapping) else _extract_json(raw)
        return SemanticSpec.model_validate(data)
    except (json.JSONDecodeError, ValueError, TypeError, ValidationError):
        return None


def _extract_json(raw: str) -> object:
    """Parse the JSON object in ``raw``, tolerating markdown code fences.

    Raises:
        json.JSONDecodeError: The response contains no parseable JSON object.
    """
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match is None:
            raise
        return json.loads(match.group(0))


__all__ = [
    "NARRATION_ENTRY_MAX_CHARS",
    "NARRATION_MAX_ENTRIES",
    "RELATIONS_MAX_COUNT",
    "TOKENS_MAX_COUNT",
    "SemanticSpec",
    "SemanticSpecRelation",
    "SemanticSpecToken",
    "parse_semantic_spec",
]
