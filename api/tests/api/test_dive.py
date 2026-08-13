"""Tests for the POST /dive route (depth-dive spec §9).

The route needs no database session, so the ``mock_client`` fixture (real app,
mocked upstream providers) exercises the full request -> transform -> response
path without a Postgres instance.
"""

import base64
import io
from typing import Any
from unittest.mock import MagicMock
from uuid import uuid4

from fastapi.testclient import TestClient
from PIL import Image
from sqlalchemy.orm import Session

from api.dependencies import get_embedder
from api.tests.conftest import set_dependency_override
from core.database.schema import Chunk
from core.exceptions import UpstreamBadResponse, UpstreamUnavailable
from core.types.responses import CitedPassage
from depth_dive.transform import IMAGE_MAX_BYTES, TABLE_MAX_ROWS, TEXT_PASSAGE_MAX_CHARS

_VALID_BODY = {
    "captured_passage": {
        "passage_type": "text",
        "content": "Attention is all you need.",
    }
}


def _faulty_body(*, content: str) -> dict[str, Any]:
    return {"captured_passage": {"passage_type": "text", "content": content}}


def _png(w: int = 4, h: int = 4) -> str:
    """Return base64-encoded PNG bytes of the requested dimensions."""
    buffer = io.BytesIO()
    Image.new("RGB", (w, h), (10, 20, 30)).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _image_body(*, png: str, media_type: str = "image/png") -> dict[str, Any]:
    return {
        "captured_passage": {
            "passage_type": "image",
            "content": png,
            "media_type": media_type,
        }
    }


def _table_body(*, rows: list[list[str]], headers: list[str] | None = None) -> dict[str, Any]:
    body: dict[str, Any] = {
        "captured_passage": {
            "passage_type": "table",
            "rows": rows,
        }
    }
    if headers is not None:
        body["captured_passage"]["headers"] = headers
    return body


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


def test_dive_code_passage_returns_422(mock_client: TestClient) -> None:
    """A code passage is unsupported by the MVP tracer bullet and returns 422."""
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


def test_dive_returns_interactive_animation_scene_graph(
    mock_client: TestClient, patched_dive_grounding: Any
) -> None:
    """A valid text passage returns 200 with elements, steps, and initial_state."""
    patched_dive_grounding()
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


def test_dive_response_has_exact_field_set(
    mock_client: TestClient, patched_dive_grounding: Any
) -> None:
    """The 200 body exposes exactly the HarnessBResponse spec §9 fields."""
    patched_dive_grounding()
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


def test_dive_tracer_bullet_is_not_grounded(
    mock_client: TestClient, patched_dive_grounding: Any
) -> None:
    """An ungrounded assembly result leaves the search flags off and cites nothing."""
    patched_dive_grounding()
    response = mock_client.post("/dive", json=_VALID_BODY)
    assert response.status_code == 200
    body = response.json()
    assert body["grounded"] is False
    assert body["external_search_attempted"] is False
    assert body["external_search_failed"] is False
    assert body["external_search_note"] is None
    assert body["cited_passages"] == []


def test_dive_grounded_response_populates_citations(
    mock_client: TestClient, patched_dive_grounding: Any
) -> None:
    """A grounded assembly result populates grounded and cited_passages."""
    passage_id = uuid4()
    patched_dive_grounding(
        grounded=True, cited_passages=[CitedPassage(chunk_id=passage_id, text="corpus chunk")]
    )
    response = mock_client.post("/dive", json=_VALID_BODY)
    assert response.status_code == 200
    body = response.json()
    assert body["grounded"] is True
    assert len(body["cited_passages"]) == 1
    passage = body["cited_passages"][0]
    assert set(passage.keys()) == {"chunk_id", "text"}
    assert passage["chunk_id"] == str(passage_id)
    assert passage["text"] == "corpus chunk"


# ============================================================
# 200 — external_search_* flag wiring (web-search step, ticket #244)
# ============================================================


def test_dive_successful_search_reports_attempted_only(
    mock_client: TestClient, patched_dive_grounding: Any
) -> None:
    """A successful web search leaves attempted=True with no failure or note."""
    patched_dive_grounding(external_search_attempted=True)
    response = mock_client.post("/dive", json=_VALID_BODY)
    assert response.status_code == 200
    body = response.json()
    assert body["external_search_attempted"] is True
    assert body["external_search_failed"] is False
    assert body["external_search_note"] is None


def test_dive_failed_search_surfaces_flag_and_note(
    mock_client: TestClient, patched_dive_grounding: Any
) -> None:
    """A double-failed search surfaces the failed flag and the user-facing note."""
    patched_dive_grounding(
        external_search_attempted=True,
        external_search_failed=True,
        external_search_note="External grounding was sought but could not be "
        "retrieved; this dive reflects the ingested corpus only.",
    )
    response = mock_client.post("/dive", json=_VALID_BODY)
    assert response.status_code == 200
    body = response.json()
    assert body["external_search_attempted"] is True
    assert body["external_search_failed"] is True
    assert body["external_search_note"] is not None
    assert "external" in body["external_search_note"].lower()


def test_dive_applies_worked_example_treatment(
    mock_client: TestClient, patched_dive_grounding: Any
) -> None:
    """The demo payload recommends and applies the worked_example treatment."""
    patched_dive_grounding()
    response = mock_client.post("/dive", json=_VALID_BODY)
    assert response.status_code == 200
    body = response.json()
    assert body["recommended_treatments"] == ["worked_example"]
    assert body["applied_treatments"] == ["worked_example"]
    assert body["routing_note"] is None


def test_dive_steps_reference_declared_elements(
    mock_client: TestClient, patched_dive_grounding: Any
) -> None:
    """Every step's element_states key resolves to a declared scene element."""
    patched_dive_grounding()
    response = mock_client.post("/dive", json=_VALID_BODY)
    assert response.status_code == 200
    output = response.json()["output"]
    element_ids = {element["id"] for element in output["elements"]}
    for step in output["steps"]:
        for element_id in step["element_states"]:
            assert element_id in element_ids


def test_dive_explicit_request_override_routes_treatment(
    mock_client: TestClient, patched_dive_grounding: Any
) -> None:
    """An explicit requested treatment wins over the recommendation with a note."""
    patched_dive_grounding()
    body = {**_VALID_BODY, "requested_treatments": ["segmented_carousel"]}
    response = mock_client.post("/dive", json=body)
    assert response.status_code == 200
    data = response.json()
    assert data["recommended_treatments"] == ["worked_example"]
    assert data["applied_treatments"] == ["segmented_carousel"]
    assert data["routing_note"] is not None


def test_dive_deferred_treatment_is_accepted_and_routed_not_422(
    mock_client: TestClient, patched_dive_grounding: Any
) -> None:
    """A deferred requested treatment is accepted, dropped, and noted (not a 422)."""
    patched_dive_grounding()
    body = {**_VALID_BODY, "requested_treatments": ["analogy_mapping"]}
    response = mock_client.post("/dive", json=body)
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["applied_treatments"] == ["worked_example"]
    assert data["routing_note"] is not None
    assert "analogy_mapping" in data["routing_note"]


# ============================================================
# 200 + 422 — image, diagram, and table passages
# ============================================================


def test_dive_accepts_image_passage(mock_client: TestClient, patched_dive_grounding: Any) -> None:
    """A valid image passage returns 200 with the interactive animation."""
    patched_dive_grounding()
    response = mock_client.post("/dive", json=_image_body(png=_png()))
    assert response.status_code == 200, response.text
    assert response.json()["output"]["output_type"] == "interactive_animation"


def test_dive_accepts_diagram_passage(mock_client: TestClient, patched_dive_grounding: Any) -> None:
    """A valid diagram passage uses the same carrier and returns 200."""
    patched_dive_grounding()
    body = {
        "captured_passage": {
            "passage_type": "diagram",
            "content": _png(),
            "media_type": "image/png",
        }
    }
    response = mock_client.post("/dive", json=body)
    assert response.status_code == 200


def test_dive_accepts_table_passage(mock_client: TestClient, patched_dive_grounding: Any) -> None:
    """A valid table passage returns 200 with the interactive animation."""
    patched_dive_grounding()
    response = mock_client.post("/dive", json=_table_body(rows=[["a", "1"], ["b", "2"]]))
    assert response.status_code == 200
    assert response.json()["output"]["output_type"] == "interactive_animation"


def test_dive_image_oversized_returns_422(mock_client: TestClient) -> None:
    """Image bytes exceeding the 5 MB size bound return 422 with a clear message."""
    oversized = base64.b64encode(b"_" * (IMAGE_MAX_BYTES + 1)).decode("ascii")
    response = mock_client.post("/dive", json=_image_body(png=oversized))
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert "size limit" in detail


def test_dive_image_invalid_payload_returns_422(mock_client: TestClient) -> None:
    """Non-image bytes declared as image content return 422 with a clear message."""
    bogus = base64.b64encode(b"not actually an image").decode("ascii")
    response = mock_client.post("/dive", json=_image_body(png=bogus))
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert "not a valid image" in detail


def test_dive_image_unsupported_media_type_returns_422(mock_client: TestClient) -> None:
    """A media_type outside the allowed set is rejected with 422."""
    png = _png()
    body = {
        "captured_passage": {
            "passage_type": "image",
            "content": png,
            "media_type": "image/bmp",
        }
    }
    response = mock_client.post("/dive", json=body)
    assert response.status_code == 422


def test_dive_table_over_row_bound_returns_422(mock_client: TestClient) -> None:
    """A table exceeding the row bound returns 422 with a clear message."""
    rows = [["x"] for _ in range(TABLE_MAX_ROWS + 1)]
    response = mock_client.post("/dive", json=_table_body(rows=rows))
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert "row limit" in detail


# ============================================================
# 502 / 503 — upstream error mapping
# ============================================================


def test_dive_embeddings_bad_response_returns_502(mock_client: TestClient) -> None:
    """A mocked embeddings provider returning a bad response maps to 502."""

    def _bad_embedder() -> Any:
        embedder = MagicMock()
        embedder.embed.side_effect = UpstreamBadResponse("bad upstream response")
        return embedder

    set_dependency_override(mock_client, get_embedder, _bad_embedder)
    response = mock_client.post("/dive", json=_VALID_BODY)

    assert response.status_code == 502
    assert "detail" in response.json()


def test_dive_embeddings_unavailable_returns_503(mock_client: TestClient) -> None:
    """A mocked embeddings provider that's unreachable/timeout maps to 503."""

    def _unavailable_embedder() -> Any:
        embedder = MagicMock()
        embedder.embed.side_effect = UpstreamUnavailable("timeout")
        return embedder

    set_dependency_override(mock_client, get_embedder, _unavailable_embedder)
    response = mock_client.post("/dive", json=_VALID_BODY)

    assert response.status_code == 503
    assert "detail" in response.json()


# ============================================================
# 200 — end-to-end against a real test DB (skips without Postgres)
# ============================================================


def test_dive_end_to_end_anchored_text_is_grounded(
    client: TestClient,
    test_session: Session,
    ingest_a_paper: Any,
) -> None:
    """An anchored text passage against a ready corpus is grounded with citations."""
    ingest_a_paper("RAG Paper")
    anchor = test_session.query(Chunk).first()
    assert anchor is not None
    parent_id = anchor.parent_chunk_id or anchor.chunk_id

    body = {
        "captured_passage": {
            "passage_type": "text",
            "content": "Attention is all you need.",
            "chunk_id": str(anchor.chunk_id),
        }
    }
    response = client.post("/dive", json=body)

    assert response.status_code == 200
    data = response.json()
    assert data["grounded"] is True
    assert len(data["cited_passages"]) >= 1
    assert data["cited_passages"][0]["chunk_id"] == str(parent_id)
    assert data["cited_passages"][0]["text"]


def test_dive_end_to_end_unanchored_text_is_grounded(
    client: TestClient,
    ingest_a_paper: Any,
) -> None:
    """An unanchored passage matching a ready corpus passes the similarity gate."""
    ingest_a_paper("RAG Paper")
    body = {
        "captured_passage": {
            "passage_type": "text",
            "content": "Attention is all you need.",
        }
    }
    response = client.post("/dive", json=body)

    assert response.status_code == 200
    data = response.json()
    assert data["grounded"] is True
    assert len(data["cited_passages"]) >= 1


def test_dive_end_to_end_empty_corpus_unanchored_is_not_grounded(client: TestClient) -> None:
    """An unanchored passage against an empty corpus is ungrounded, not an error."""
    body = {
        "captured_passage": {
            "passage_type": "text",
            "content": "Attention is all you need.",
        }
    }
    response = client.post("/dive", json=body)

    assert response.status_code == 200
    data = response.json()
    assert data["grounded"] is False
    assert data["cited_passages"] == []


def test_dive_end_to_end_empty_corpus_anchored_is_not_grounded(client: TestClient) -> None:
    """An anchor that no ready corpus can resolve is ungrounded, not an error."""
    body = {
        "captured_passage": {
            "passage_type": "text",
            "content": "Attention is all you need.",
            "chunk_id": str(uuid4()),
        }
    }
    response = client.post("/dive", json=body)

    assert response.status_code == 200
    data = response.json()
    assert data["grounded"] is False
    assert data["cited_passages"] == []
