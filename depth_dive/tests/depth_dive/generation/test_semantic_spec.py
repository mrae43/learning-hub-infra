"""Tests for the bounded semantic spec model and parser (ADR-0022, ticket #305).

The semantic spec is the harness-internal artifact the generation turn's model
emits instead of a full scene graph: content and relations only, no geometry.
These tests cover the spec's shape, its boundedness rules, and the pure
parse/validate function that tolerates markdown fences and treats every
boundedness violation as malformed.
"""

import json

import pytest
from pydantic import ValidationError

from depth_dive.generation.semantic_spec import (
    NARRATION_ENTRY_MAX_CHARS,
    NARRATION_MAX_ENTRIES,
    RELATIONS_MAX_COUNT,
    TOKENS_MAX_COUNT,
    SemanticSpec,
    SemanticSpecRelation,
    SemanticSpecToken,
    parse_semantic_spec,
)


def _valid_spec() -> dict[str, object]:
    """A semantic-spec payload that satisfies every boundedness rule."""
    return {
        "title": "Self-attention",
        "concept": "attention",
        "narration": [
            "Three input words arrive.",
            "For every word, build query, key, and value vectors.",
            "Each query attends to every key, weighted by similarity.",
        ],
        "tokens": [
            {"id": "tok-river", "label": "river"},
            {"id": "tok-bank", "label": "bank"},
            {"id": "tok-money", "label": "money"},
        ],
        "relations": [
            {"from": "tok-bank", "to": "tok-money", "weight": 0.7},
            {"from": "tok-river", "to": "tok-bank", "weight": 0.3},
        ],
    }


# ============================================================
# Model shape (ticket #305 acceptance criterion 1)
# ============================================================


def test_spec_parses_from_a_valid_payload() -> None:
    """A valid payload builds a SemanticSpec carrying every field."""
    spec = SemanticSpec.model_validate(_valid_spec())
    assert spec.title == "Self-attention"
    assert spec.concept == "attention"
    assert spec.narration == [
        "Three input words arrive.",
        "For every word, build query, key, and value vectors.",
        "Each query attends to every key, weighted by similarity.",
    ]
    assert spec.tokens[0].id == "tok-river"
    assert spec.tokens[0].label == "river"
    assert spec.relations[0].from_ == "tok-bank"
    assert spec.relations[0].to == "tok-money"
    assert spec.relations[0].weight == 0.7


def test_spec_rejects_geometry_fields() -> None:
    """The spec carries no coordinates, viewport, or element states."""
    payload = _valid_spec()
    payload["viewport"] = {"width": 800, "height": 520}
    with pytest.raises(ValidationError):
        SemanticSpec.model_validate(payload)


def test_spec_rejects_elements_and_element_states() -> None:
    """Scene-graph-only fields are not part of the semantic spec."""
    for field_name, value in (
        ("elements", [{"id": "e1", "type": "token", "text": "bank"}]),
        ("initial_state", {"tok-bank": {"opacity": 0}}),
        ("output_type", "interactive_animation"),
    ):
        payload = _valid_spec()
        payload[field_name] = value
        with pytest.raises(ValidationError):
            SemanticSpec.model_validate(payload)


def test_spec_accepts_empty_tokens_and_relations() -> None:
    """An image/diagram/table passage carries no tokens or relations (ADR-0022)."""
    payload = _valid_spec()
    payload["tokens"] = []
    payload["relations"] = []
    spec = SemanticSpec.model_validate(payload)
    assert spec.tokens == []
    assert spec.relations == []


def test_token_has_id_and_label_only() -> None:
    """A token carries exactly its id and label; no geometry creeps in."""
    token = SemanticSpecToken.model_validate({"id": "tok-bank", "label": "bank"})
    assert token.id == "tok-bank"
    assert token.label == "bank"


def test_token_rejects_coordinates() -> None:
    """A token carrying x/y coordinates violates the no-geometry rule."""
    with pytest.raises(ValidationError):
        SemanticSpecToken.model_validate({"id": "tok-bank", "label": "bank", "x": 100, "y": 200})


def test_relation_has_from_to_and_weight_only() -> None:
    """A relation carries from/to/weight; the from key is the JSON wire name."""
    relation = SemanticSpecRelation.model_validate(
        {"from": "tok-bank", "to": "tok-money", "weight": 0.7}
    )
    assert relation.from_ == "tok-bank"
    assert relation.to == "tok-money"
    assert relation.weight == 0.7


# ============================================================
# Boundedness rules (ticket #305 acceptance criterion 2)
# ============================================================


def _bounded_violation(payload: dict[str, object]) -> None:
    """A payload violating any boundedness rule must fail model validation."""
    with pytest.raises(ValidationError):
        SemanticSpec.model_validate(payload)


def test_narration_entry_count_is_capped() -> None:
    """More narration entries than the cap violate the boundedness rule."""
    payload = _valid_spec()
    payload["narration"] = [f"entry {i}" for i in range(NARRATION_MAX_ENTRIES + 1)]
    _bounded_violation(payload)


def test_narration_entry_character_cap_is_enforced() -> None:
    """A narration entry longer than the cap violates the boundedness rule."""
    payload = _valid_spec()
    payload["narration"] = ["x" * (NARRATION_ENTRY_MAX_CHARS + 1)]
    _bounded_violation(payload)


def test_narration_entry_at_the_character_cap_is_accepted() -> None:
    """An entry exactly at the character cap is within bounds."""
    payload = _valid_spec()
    payload["narration"] = ["x" * NARRATION_ENTRY_MAX_CHARS]
    spec = SemanticSpec.model_validate(payload)
    assert len(spec.narration[0]) == NARRATION_ENTRY_MAX_CHARS


def test_token_count_is_capped() -> None:
    """More tokens than the cap violate the boundedness rule."""
    payload = _valid_spec()
    payload["tokens"] = [
        {"id": f"tok-{i}", "label": f"token {i}"} for i in range(TOKENS_MAX_COUNT + 1)
    ]
    _bounded_violation(payload)


def test_token_ids_are_unique() -> None:
    """Duplicate token ids violate the boundedness rule."""
    payload = _valid_spec()
    payload["tokens"] = [
        {"id": "tok-bank", "label": "bank"},
        {"id": "tok-bank", "label": "the other bank"},
    ]
    _bounded_violation(payload)


def test_token_ids_are_non_empty() -> None:
    """An empty token id violates the boundedness rule."""
    payload = _valid_spec()
    payload["tokens"] = [
        {"id": "", "label": "unnamed"},
        {"id": "tok-bank", "label": "bank"},
    ]
    _bounded_violation(payload)


def test_relation_count_is_capped() -> None:
    """More relations than the cap violate the boundedness rule."""
    payload = _valid_spec()
    payload["relations"] = [
        {"from": "tok-bank", "to": "tok-money", "weight": 0.5}
        for _ in range(RELATIONS_MAX_COUNT + 1)
    ]
    _bounded_violation(payload)


@pytest.mark.parametrize("weight", [-0.1, 1.5, 2.0, -1.0])
def test_relation_weight_is_bounded_to_unit_interval(weight: float) -> None:
    """A weight outside [0, 1] violates the boundedness rule."""
    payload = _valid_spec()
    payload["relations"] = [{"from": "tok-bank", "to": "tok-money", "weight": weight}]
    _bounded_violation(payload)


@pytest.mark.parametrize("weight", [0.0, 0.5, 1.0])
def test_relation_weight_at_the_boundaries_is_accepted(weight: float) -> None:
    """Weights at 0 and 1 are within bounds."""
    payload = _valid_spec()
    payload["relations"] = [{"from": "tok-bank", "to": "tok-money", "weight": weight}]
    spec = SemanticSpec.model_validate(payload)
    assert spec.relations[0].weight == weight


def test_relation_from_must_reference_a_declared_token() -> None:
    """A relation whose from id is undeclared violates the boundedness rule."""
    payload = _valid_spec()
    payload["relations"] = [{"from": "tok-ghost", "to": "tok-bank", "weight": 0.5}]
    _bounded_violation(payload)


def test_relation_to_must_reference_a_declared_token() -> None:
    """A relation whose to id is undeclared violates the boundedness rule."""
    payload = _valid_spec()
    payload["relations"] = [{"from": "tok-bank", "to": "tok-ghost", "weight": 0.5}]
    _bounded_violation(payload)


# ============================================================
# parse_semantic_spec (ticket #305 acceptance criterion 3)
# ============================================================


def test_parse_returns_the_validated_spec() -> None:
    """A valid payload parses into a SemanticSpec."""
    spec = parse_semantic_spec(_valid_spec())
    assert spec is not None
    assert spec.title == "Self-attention"
    assert len(spec.tokens) == 3
    assert len(spec.relations) == 2


def test_parse_tolerates_a_json_string() -> None:
    """The parser accepts the raw JSON string form."""
    spec = parse_semantic_spec(json.dumps(_valid_spec()))
    assert spec is not None
    assert spec.concept == "attention"


def test_parse_tolerates_markdown_code_fence() -> None:
    """A payload wrapped in ```json fences still parses."""
    raw = f"```json\n{json.dumps(_valid_spec())}\n```"
    spec = parse_semantic_spec(raw)
    assert spec is not None
    assert spec.title == "Self-attention"


def test_parse_returns_none_for_oversized_narration() -> None:
    """An oversized narration is malformed, so parsing yields None."""
    payload = _valid_spec()
    payload["narration"] = [f"entry {i}" for i in range(NARRATION_MAX_ENTRIES + 1)]
    assert parse_semantic_spec(json.dumps(payload)) is None


def test_parse_returns_none_for_duplicate_token_id() -> None:
    """A duplicate token id is malformed, so parsing yields None."""
    payload = _valid_spec()
    payload["tokens"] = [
        {"id": "tok-bank", "label": "bank"},
        {"id": "tok-bank", "label": "the other bank"},
    ]
    assert parse_semantic_spec(json.dumps(payload)) is None


def test_parse_returns_none_for_empty_token_id() -> None:
    """An empty token id is malformed, so parsing yields None."""
    payload = _valid_spec()
    payload["tokens"] = [
        {"id": "", "label": "unnamed"},
        {"id": "tok-bank", "label": "bank"},
    ]
    assert parse_semantic_spec(json.dumps(payload)) is None


def test_parse_returns_none_for_oversized_narration_entry() -> None:
    """A narration entry over the character cap is malformed, so parsing yields None."""
    payload = _valid_spec()
    payload["narration"] = ["x" * (NARRATION_ENTRY_MAX_CHARS + 1)]
    assert parse_semantic_spec(json.dumps(payload)) is None


def test_parse_returns_none_for_out_of_range_weight() -> None:
    """An out-of-range weight is malformed, so parsing yields None."""
    payload = _valid_spec()
    payload["relations"] = [{"from": "tok-bank", "to": "tok-money", "weight": 3.0}]
    assert parse_semantic_spec(json.dumps(payload)) is None


def test_parse_returns_none_for_undeclared_relation_endpoint() -> None:
    """An undeclared relation endpoint is malformed, so parsing yields None."""
    payload = _valid_spec()
    payload["relations"] = [{"from": "tok-ghost", "to": "tok-bank", "weight": 0.5}]
    assert parse_semantic_spec(json.dumps(payload)) is None


def test_parse_returns_none_for_invalid_json() -> None:
    """Non-JSON model output fails to parse."""
    assert parse_semantic_spec("not json at all") is None
    assert parse_semantic_spec("") is None


def test_parse_returns_none_for_schema_violation() -> None:
    """Valid JSON missing a required field fails validation."""
    assert parse_semantic_spec('{"title": "only a title"}') is None
    assert parse_semantic_spec("[]") is None


def test_parse_returns_none_for_scene_graph_payload() -> None:
    """A full scene graph (with geometry) is not a semantic spec."""
    payload = {
        "output_type": "interactive_animation",
        "title": "Self-attention",
        "concept": "attention",
        "viewport": {"width": 800, "height": 520},
        "elements": [],
        "steps": [],
        "initial_state": {},
    }
    assert parse_semantic_spec(json.dumps(payload)) is None
