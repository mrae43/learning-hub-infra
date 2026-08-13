"""Tests for the Depth Dive assembly agent (ADR-0020, ticket #243).

The ``core/retrieval/`` primitives are mocked at the assembly module boundary
(their database behaviour is covered by ``core``'s own retrieval tests) and the
embedder is a stub, per coding-standards.md: hosted API calls stay out of unit
tests.
"""

from collections.abc import Sequence
from unittest.mock import MagicMock
from uuid import UUID, uuid4

import pytest
from sqlalchemy.orm import Session

from core.clients import Embedder
from core.exceptions import UpstreamUnavailable
from core.types.captured_passage import (
    CapturedPassage,
    CodePassage,
    ImagePassage,
    TablePassage,
    TextPassage,
)
from core.types.responses import CitedPassage, ScoredChunk
from core.types.retrieval_config import RetrievalConfig
from depth_dive.assembly import assembly_agent
from depth_dive.assembly.assembly_agent import (
    SIMILARITY_GATE_THRESHOLD,
    GroundingResult,
    run_assembly,
)
from depth_dive.framing.framing_agent import run_framing

_CONFIG = RetrievalConfig(model_name="text-embedding-3-small", ef_search=40, top_k=5)


class _StubEmbedder:
    """A stub embedder that returns one fixed vector and records its inputs."""

    def __init__(self, vector: list[float] | None = None) -> None:
        self.vector = vector if vector is not None else [0.5] * 1536
        self.embedded: list[str] = []

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        self.embedded.extend(texts)
        return [list(self.vector) for _ in texts]


def _neighbor(chunk_id: UUID, text: str, score: float) -> ScoredChunk:
    return ScoredChunk(chunk_id=chunk_id, text=text, score=score)


def _run(passage: CapturedPassage, *, session: Session, embedder: Embedder) -> GroundingResult:
    return run_assembly(
        passage,
        run_framing(passage),
        session=session,
        embedder=embedder,
        config=_CONFIG,
    )


def test_anchored_text_grounds_with_parent_and_neighbors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An anchored text passage grounds with the parent chunk plus neighbors."""
    anchor_id = uuid4()
    parent_id = uuid4()
    neighbor_id = uuid4()
    fetch_parent = MagicMock(
        return_value=CitedPassage(chunk_id=parent_id, text="the enclosing parent chunk")
    )
    search_neighbors = MagicMock(return_value=[_neighbor(neighbor_id, "a semantic neighbor", 0.9)])
    monkeypatch.setattr(assembly_agent, "fetch_parent_chunk", fetch_parent)
    monkeypatch.setattr(assembly_agent, "search_dense_neighbors", search_neighbors)

    passage = TextPassage(content="Attention is all you need.", chunk_id=anchor_id)
    session = MagicMock()
    result = _run(passage, session=session, embedder=_StubEmbedder())

    assert result.grounded is True
    assert [p.chunk_id for p in result.cited_passages] == [parent_id, neighbor_id]
    assert result.cited_passages[0].text == "the enclosing parent chunk"
    fetch_parent.assert_called_once_with(session=session, chunk_id=anchor_id)


def test_anchored_code_grounds_with_parent_and_neighbors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An anchored code passage uses the same parent + neighbors grounding."""
    anchor_id = uuid4()
    parent_id = uuid4()
    monkeypatch.setattr(
        assembly_agent,
        "fetch_parent_chunk",
        MagicMock(return_value=CitedPassage(chunk_id=parent_id, text="parent")),
    )
    monkeypatch.setattr(assembly_agent, "search_dense_neighbors", MagicMock(return_value=[]))

    passage = CodePassage(content="def f():\n    return 1", language="python", chunk_id=anchor_id)
    result = _run(passage, session=MagicMock(), embedder=_StubEmbedder())

    assert result.grounded is True
    assert [p.chunk_id for p in result.cited_passages] == [parent_id]


def test_anchored_passage_with_unresolvable_anchor_is_not_grounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An anchor the corpus cannot resolve grounds nothing, without embedding."""
    monkeypatch.setattr(assembly_agent, "fetch_parent_chunk", MagicMock(return_value=None))
    search_neighbors = MagicMock()
    monkeypatch.setattr(assembly_agent, "search_dense_neighbors", search_neighbors)

    passage = TextPassage(content="Attention is all you need.", chunk_id=uuid4())
    embedder = _StubEmbedder()
    result = _run(passage, session=MagicMock(), embedder=embedder)

    assert result.grounded is False
    assert result.cited_passages == []
    search_neighbors.assert_not_called()
    assert embedder.embedded == []


# ============================================================
# Unanchored passages: the corpus-similarity gate
# ============================================================


def test_unanchored_text_passes_gate_when_corpus_is_similar(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A top neighbor at/above the gate threshold grounds the passage."""
    close_id = uuid4()
    monkeypatch.setattr(assembly_agent, "fetch_parent_chunk", MagicMock())
    monkeypatch.setattr(
        assembly_agent,
        "search_dense_neighbors",
        MagicMock(return_value=[_neighbor(close_id, "a near-duplicate chunk", 0.9)]),
    )

    passage = TextPassage(content="Attention is all you need.")
    result = _run(passage, session=MagicMock(), embedder=_StubEmbedder())

    assert result.grounded is True
    assert [p.chunk_id for p in result.cited_passages] == [close_id]
    assert result.cited_passages[0].text == "a near-duplicate chunk"


def test_unanchored_text_fails_gate_when_corpus_is_dissimilar(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A top neighbor below the gate threshold leaves the passage ungrounded."""
    monkeypatch.setattr(assembly_agent, "fetch_parent_chunk", MagicMock())
    monkeypatch.setattr(
        assembly_agent,
        "search_dense_neighbors",
        MagicMock(
            return_value=[
                _neighbor(uuid4(), "unrelated chunk", SIMILARITY_GATE_THRESHOLD - 0.01),
            ]
        ),
    )

    passage = TextPassage(content="Attention is all you need.")
    result = _run(passage, session=MagicMock(), embedder=_StubEmbedder())

    assert result.grounded is False
    assert result.cited_passages == []


def test_unanchored_gate_boundary_score_passes(monkeypatch: pytest.MonkeyPatch) -> None:
    """A top neighbor exactly at the threshold counts as grounded."""
    monkeypatch.setattr(assembly_agent, "fetch_parent_chunk", MagicMock())
    monkeypatch.setattr(
        assembly_agent,
        "search_dense_neighbors",
        MagicMock(return_value=[_neighbor(uuid4(), "boundary chunk", SIMILARITY_GATE_THRESHOLD)]),
    )

    passage = TextPassage(content="Attention is all you need.")
    result = _run(passage, session=MagicMock(), embedder=_StubEmbedder())

    assert result.grounded is True


def test_unanchored_gate_cites_only_neighbors_above_threshold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Gate-passing citations keep only the neighbors that clear the threshold."""
    close_id = uuid4()
    monkeypatch.setattr(assembly_agent, "fetch_parent_chunk", MagicMock())
    monkeypatch.setattr(
        assembly_agent,
        "search_dense_neighbors",
        MagicMock(
            return_value=[
                _neighbor(close_id, "close chunk", 0.9),
                _neighbor(uuid4(), "weak chunk", SIMILARITY_GATE_THRESHOLD - 0.01),
            ]
        ),
    )

    passage = TextPassage(content="Attention is all you need.")
    result = _run(passage, session=MagicMock(), embedder=_StubEmbedder())

    assert result.grounded is True
    assert [p.chunk_id for p in result.cited_passages] == [close_id]


def test_unanchored_empty_corpus_fails_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    """No neighbors at all (empty corpus) is a failed gate, not an error."""
    monkeypatch.setattr(assembly_agent, "fetch_parent_chunk", MagicMock())
    search_neighbors = MagicMock(return_value=[])
    monkeypatch.setattr(assembly_agent, "search_dense_neighbors", search_neighbors)

    passage = TextPassage(content="Attention is all you need.")
    embedder = _StubEmbedder()
    result = _run(passage, session=MagicMock(), embedder=embedder)

    assert result.grounded is False
    assert result.cited_passages == []
    assert embedder.embedded == ["Attention is all you need."]
    search_neighbors.assert_called_once()
    assert search_neighbors.call_args.kwargs["query_vector"] == embedder.vector


# ============================================================
# Non-text passages: embeddable representation for the gate
# ============================================================


def test_unanchored_image_gates_on_caption(monkeypatch: pytest.MonkeyPatch) -> None:
    """An unanchored image embeds its caption for the similarity gate."""
    close_id = uuid4()
    monkeypatch.setattr(assembly_agent, "fetch_parent_chunk", MagicMock())
    monkeypatch.setattr(
        assembly_agent,
        "search_dense_neighbors",
        MagicMock(return_value=[_neighbor(close_id, "caption-matching chunk", 0.9)]),
    )

    passage = ImagePassage(content=b"\x00", media_type="image/png", caption="An attention plot")
    embedder = _StubEmbedder()
    result = _run(passage, session=MagicMock(), embedder=embedder)

    assert result.grounded is True
    assert embedder.embedded == ["An attention plot"]


def test_unanchored_image_without_caption_cannot_be_grounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No caption means nothing embeddable: ungrounded without any call."""
    search_neighbors = MagicMock()
    monkeypatch.setattr(assembly_agent, "fetch_parent_chunk", MagicMock())
    monkeypatch.setattr(assembly_agent, "search_dense_neighbors", search_neighbors)

    passage = ImagePassage(content=b"\x00", media_type="image/png")
    embedder = _StubEmbedder()
    result = _run(passage, session=MagicMock(), embedder=embedder)

    assert result.grounded is False
    assert result.cited_passages == []
    assert embedder.embedded == []
    search_neighbors.assert_not_called()


def test_unanchored_table_gates_on_serialized_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    """An unanchored table embeds a deterministic serialization of its rows."""
    monkeypatch.setattr(assembly_agent, "fetch_parent_chunk", MagicMock())
    monkeypatch.setattr(
        assembly_agent,
        "search_dense_neighbors",
        MagicMock(return_value=[_neighbor(uuid4(), "table chunk", 0.9)]),
    )

    passage = TablePassage(rows=[["a", "1"], ["b", "2"]], headers=["l", "v"], caption="Scores")
    embedder = _StubEmbedder()
    result = _run(passage, session=MagicMock(), embedder=embedder)

    assert result.grounded is True
    assert embedder.embedded == ["Scores\nl | v\na | 1\nb | 2"]


def test_unanchored_dict_table_rows_serialize_keyed_cells(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Header-keyed dict rows serialize as ``key: value`` cells."""
    monkeypatch.setattr(assembly_agent, "fetch_parent_chunk", MagicMock())
    monkeypatch.setattr(
        assembly_agent,
        "search_dense_neighbors",
        MagicMock(return_value=[_neighbor(uuid4(), "table chunk", 0.9)]),
    )

    passage = TablePassage(rows=[{"l": "a", "v": "1"}])
    embedder = _StubEmbedder()
    _run(passage, session=MagicMock(), embedder=embedder)

    assert embedder.embedded == ["l: a | v: 1"]


def test_anchored_image_falls_back_to_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    """A document_id anchor cannot resolve yet (ADR-0019), so the gate runs."""
    close_id = uuid4()
    fetch_parent = MagicMock()
    monkeypatch.setattr(assembly_agent, "fetch_parent_chunk", fetch_parent)
    monkeypatch.setattr(
        assembly_agent,
        "search_dense_neighbors",
        MagicMock(return_value=[_neighbor(close_id, "caption-matching chunk", 0.9)]),
    )

    passage = ImagePassage(
        content=b"\x00",
        media_type="image/png",
        caption="An attention plot",
        document_id=uuid4(),
        ordinal="Figure 3",
    )
    result = _run(passage, session=MagicMock(), embedder=_StubEmbedder())

    assert result.grounded is True
    assert [p.chunk_id for p in result.cited_passages] == [close_id]
    fetch_parent.assert_not_called()


# ============================================================
# Anchored citation hygiene and upstream failures
# ============================================================


def test_anchored_citations_skip_anchor_chunk_and_duplicates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The anchor chunk itself and duplicate ids never appear in citations."""
    anchor_id = uuid4()
    parent_id = uuid4()
    neighbor_id = uuid4()
    monkeypatch.setattr(
        assembly_agent,
        "fetch_parent_chunk",
        MagicMock(return_value=CitedPassage(chunk_id=parent_id, text="parent")),
    )
    monkeypatch.setattr(
        assembly_agent,
        "search_dense_neighbors",
        MagicMock(
            return_value=[
                _neighbor(anchor_id, "the anchor chunk itself", 0.99),
                _neighbor(neighbor_id, "a semantic neighbor", 0.9),
                _neighbor(parent_id, "the parent again", 0.8),
                _neighbor(neighbor_id, "the neighbor again", 0.7),
            ]
        ),
    )

    passage = TextPassage(content="Attention is all you need.", chunk_id=anchor_id)
    result = _run(passage, session=MagicMock(), embedder=_StubEmbedder())

    assert result.grounded is True
    assert [p.chunk_id for p in result.cited_passages] == [parent_id, neighbor_id]


def test_assembly_propagates_embeddings_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    """An unreachable embeddings API surfaces as UpstreamUnavailable (503)."""
    monkeypatch.setattr(
        assembly_agent,
        "fetch_parent_chunk",
        MagicMock(return_value=CitedPassage(chunk_id=uuid4(), text="parent")),
    )

    class _FailingEmbedder:
        def embed(self, texts: Sequence[str]) -> list[list[float]]:
            raise UpstreamUnavailable("embeddings API unreachable")

    passage = TextPassage(content="Attention is all you need.", chunk_id=uuid4())
    with pytest.raises(UpstreamUnavailable):
        _run(passage, session=MagicMock(), embedder=_FailingEmbedder())


def test_assembly_propagates_database_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    """An unreachable database surfaces as UpstreamUnavailable (503)."""
    monkeypatch.setattr(
        assembly_agent,
        "fetch_parent_chunk",
        MagicMock(side_effect=UpstreamUnavailable("database unreachable")),
    )

    passage = TextPassage(content="Attention is all you need.", chunk_id=uuid4())
    with pytest.raises(UpstreamUnavailable):
        _run(passage, session=MagicMock(), embedder=_StubEmbedder())
