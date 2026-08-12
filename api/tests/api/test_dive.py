"""Tests for the POST /dive route (depth-dive spec §9).

The route needs no database session, so the ``mock_client`` fixture (real app,
mocked upstream providers) exercises the full request -> transform -> response
path without a Postgres instance.
"""

from typing import Any

from fastapi.testclient import TestClient

from depth_dive.transform import TEXT_PASSAGE_MAX_CHARS

_VALID_BODY = {
    "captured_passage": {
        "passage_type": "text",
        "content": "Attention is all you need.",
    }
}


def _faulty_body(*, content: str) -> dict[str, Any]:
    return {"captured_passage": {"passage_type": "text", "content": content}}


# ============================================================
# 422 — request validation (FastAPI / Pydantic defaults)
# ============================================================


def test_dive_missing_body_returns_422(mock_client: TestClient) -> None:
    """POST /dive without a body returns 422 by FastAPI default."""
    response = mock_client.post("/dive")
    assert response.status_code == 422


def test_dive_missing_captured_passage_returns_422(mock_client: TestClient) -> None:
    """A body without captured_passage returns 422."""
    response = mock_client.post("/dive", json={})
    assert response.status_code == 422


def test_dive_unknown_passage_type_returns_422(mock_client: TestClient) -> None:
    """An unrecognized passage_type fails union validation with 422."""
    response = mock_client.post("/dive", json={"captured_passage": {"passage_type": "audio"}})
    assert response.status_code == 422


def test_dive_invalid_chunk_id_returns_422(mock_client: TestClient) -> None:
    """A malformed chunk_id anchor is rejected with 422."""
    body = {
        "captured_passage": {
            "passage_type": "text",
            "content": "text",
            "chunk_id": "not-a-uuid",
        }
    }
    response = mock_client.post("/dive", json=body)
    assert response.status_code == 422


# ============================================================
# 422 — passage transform validation (size/bounds)
# ============================================================


def test_dive_empty_text_returns_422(mock_client: TestClient) -> None:
    """Empty text content violates the transform bounds and returns 422."""
    response = mock_client.post("/dive", json=_faulty_body(content=""))
    assert response.status_code == 422
    assert "detail" in response.json()


def test_dive_oversized_text_returns_422(mock_client: TestClient) -> None:
    """Text exceeding the declared size bound is rejected with 422, not cropped."""
    response = mock_client.post(
        "/dive", json=_faulty_body(content="x" * (TEXT_PASSAGE_MAX_CHARS + 1))
    )
    assert response.status_code == 422
    assert "detail" in response.json()


def test_dive_non_text_passage_returns_422(mock_client: TestClient) -> None:
    """A non-text passage is unsupported by the MVP tracer bullet and returns 422."""
    body = {
        "captured_passage": {
            "passage_type": "code",
            "content": "def f():\n    return 1",
            "language": "python",
        }
    }
    response = mock_client.post("/dive", json=body)
    assert response.status_code == 422
    assert "detail" in response.json()


# ============================================================
# 200 — valid text passage returns the hardcoded animation
# ============================================================


def test_dive_returns_interactive_animation_scene_graph(mock_client: TestClient) -> None:
    """A valid text passage returns 200 with elements, steps, and initial_state."""
    response = mock_client.post("/dive", json=_VALID_BODY)
    assert response.status_code == 200
    body = response.json()
    output = body["output"]
    assert output["output_type"] == "interactive_animation"
    assert output["elements"]
    assert output["steps"]
    assert output["initial_state"]
    assert isinstance(output["title"], str)
    assert output["viewport"]["width"] > 0
    assert output["viewport"]["height"] > 0


def test_dive_response_has_exact_field_set(mock_client: TestClient) -> None:
    """The 200 body exposes exactly the HarnessBResponse spec §9 fields."""
    response = mock_client.post("/dive", json=_VALID_BODY)
    assert response.status_code == 200
    assert set(response.json().keys()) == {
        "output",
        "recommended_treatments",
        "applied_treatments",
        "routing_note",
        "grounded",
        "external_search_attempted",
        "external_search_failed",
        "external_search_note",
        "cited_passages",
    }


def test_dive_tracer_bullet_is_not_grounded(mock_client: TestClient) -> None:
    """No retrieval runs in the tracer bullet: grounded=False, search flags off."""
    response = mock_client.post("/dive", json=_VALID_BODY)
    assert response.status_code == 200
    body = response.json()
    assert body["grounded"] is False
    assert body["external_search_attempted"] is False
    assert body["external_search_failed"] is False
    assert body["external_search_note"] is None
    assert body["cited_passages"] == []


def test_dive_applies_worked_example_treatment(mock_client: TestClient) -> None:
    """The demo payload recommends and applies the worked_example treatment."""
    response = mock_client.post("/dive", json=_VALID_BODY)
    assert response.status_code == 200
    body = response.json()
    assert body["recommended_treatments"] == ["worked_example"]
    assert body["applied_treatments"] == ["worked_example"]
    assert body["routing_note"] is None


def test_dive_steps_reference_declared_elements(mock_client: TestClient) -> None:
    """Every step's element_states key resolves to a declared scene element."""
    response = mock_client.post("/dive", json=_VALID_BODY)
    assert response.status_code == 200
    output = response.json()["output"]
    element_ids = {element["id"] for element in output["elements"]}
    for step in output["steps"]:
        for element_id in step["element_states"]:
            assert element_id in element_ids
