"""Tests for the Depth Dive assembly agent (ADR-0020, ticket #243, #244).

The ``core/retrieval/`` primitives are mocked at the assembly module boundary
(their database behaviour is covered by ``core``'s own retrieval tests) and the
embedder is a stub; the web-search provider is the deterministic
``StubWebSearchClient``. Hosted API calls stay out of unit tests per
coding-standards.md.
"""

from collections.abc import Sequence
from unittest.mock import MagicMock
from uuid import UUID, uuid4

import pytest
from sqlalchemy.orm import Session

from core.clients import CompletionProvider, Embedder, MockCompletionProvider
from core.exceptions import UpstreamUnavailable
from core.types.captured_passage import (
    CapturedPassage,
    CodePassage,
    DiagramPassage,
    ImagePassage,
    TablePassage,
    TextPassage,
)
from core.types.responses import CitedPassage, ScoredChunk
from core.types.retrieval_config import RetrievalConfig
from depth_dive.assembly import assembly_agent
from depth_dive.assembly.assembly_agent import (
    SIMILARITY_GATE_THRESHOLD,
    AssemblyResult,
    run_assembly,
)
from depth_dive.framing.framing_agent import run_framing
from depth_dive.generation.fallback_animation import build_fallback_animation
from depth_dive.generation.generation_agent import GenerationResult
from depth_dive.web_search.client import StubWebSearchClient, WebSearchResult
from depth_dive.web_search.wrapper import FALLBACK_NOTE

_CONFIG = RetrievalConfig(model_name="text-embedding-3-small", ef_search=40, top_k=5)


@pytest.fixture(autouse=True)
def _stub_generation(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Patch the generation turn so no LLM call runs in these tests.

    Grounding and search policy are the focus of this suite; the generation
    turn's own behaviour is covered by ``test_generation_agent.py``. The stub
    records the call so wiring tests can assert its inputs.
    """
    generation = MagicMock(
        return_value=GenerationResult(animation=build_fallback_animation("test"))
    )
    monkeypatch.setattr(assembly_agent, "run_generation", generation)
    return generation


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


def _run(
    passage: CapturedPassage,
    *,
    session: Session,
    embedder: Embedder,
    web_search: StubWebSearchClient | None = None,
    completion_provider: CompletionProvider | None = None,
) -> AssemblyResult:
    if web_search is None:
        web_search = StubWebSearchClient()
    return run_assembly(
        passage,
        run_framing(passage),
        session=session,
        embedder=embedder,
        config=_CONFIG,
        web_search=web_search,
        completion_provider=completion_provider or MockCompletionProvider(),
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


def _doc_chunk(chunk_id: UUID, text: str) -> CitedPassage:
    return CitedPassage(chunk_id=chunk_id, text=text)


# ============================================================
# Anchored non-text passages: document-relative context (ticket #253)
# ============================================================


def test_anchored_image_grounds_with_document_context_and_neighbors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A valid document anchor grounds on the document's text plus neighbors."""
    document_id = uuid4()
    doc_chunk_id = uuid4()
    neighbor_id = uuid4()
    fetch_document = MagicMock(
        return_value=[_doc_chunk(doc_chunk_id, "the section surrounding the figure")]
    )
    search_neighbors = MagicMock(return_value=[_neighbor(neighbor_id, "a semantic neighbor", 0.9)])
    monkeypatch.setattr(assembly_agent, "fetch_document_chunks", fetch_document)
    monkeypatch.setattr(assembly_agent, "search_dense_neighbors", search_neighbors)

    passage = ImagePassage(
        content=b"\x00",
        media_type="image/png",
        caption="An attention plot",
        document_id=document_id,
        ordinal="Figure 3",
    )
    session = MagicMock()
    embedder = _StubEmbedder()
    result = _run(passage, session=session, embedder=embedder)

    assert result.grounded is True
    assert [p.chunk_id for p in result.cited_passages] == [doc_chunk_id, neighbor_id]
    assert result.cited_passages[0].text == "the section surrounding the figure"
    assert embedder.embedded == ["An attention plot"]
    fetch_document.assert_called_once_with(
        session=session, document_id=document_id, limit=_CONFIG.top_k
    )


def test_anchored_diagram_grounds_with_document_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A diagram uses the same document-relative grounding path as an image."""
    document_id = uuid4()
    doc_chunk_id = uuid4()
    monkeypatch.setattr(
        assembly_agent,
        "fetch_document_chunks",
        MagicMock(return_value=[_doc_chunk(doc_chunk_id, "a diagram's section")]),
    )
    monkeypatch.setattr(assembly_agent, "search_dense_neighbors", MagicMock(return_value=[]))

    passage = DiagramPassage(
        content=b"\x00",
        media_type="image/png",
        caption="The attention mechanism",
        document_id=document_id,
        ordinal="Figure 2",
    )
    result = _run(passage, session=MagicMock(), embedder=_StubEmbedder())

    assert result.grounded is True
    assert [p.chunk_id for p in result.cited_passages] == [doc_chunk_id]


def test_anchored_table_grounds_with_document_context_and_serialized_neighbors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An anchored table embeds its serialized form for the neighbor search."""
    doc_chunk_id = uuid4()
    neighbor_id = uuid4()
    monkeypatch.setattr(
        assembly_agent,
        "fetch_document_chunks",
        MagicMock(return_value=[_doc_chunk(doc_chunk_id, "table context")]),
    )
    monkeypatch.setattr(
        assembly_agent,
        "search_dense_neighbors",
        MagicMock(return_value=[_neighbor(neighbor_id, "related table chunk", 0.9)]),
    )

    passage = TablePassage(
        rows=[["a", "1"]],
        headers=["l", "v"],
        caption="Scores",
        document_id=uuid4(),
        ordinal="Table 1",
    )
    embedder = _StubEmbedder()
    result = _run(passage, session=MagicMock(), embedder=embedder)

    assert result.grounded is True
    assert [p.chunk_id for p in result.cited_passages] == [doc_chunk_id, neighbor_id]
    assert embedder.embedded == ["Scores\nl | v\na | 1"]


def test_anchored_non_text_unresolvable_document_is_not_grounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A document_id the corpus cannot resolve grounds nothing, without embedding."""
    monkeypatch.setattr(assembly_agent, "fetch_document_chunks", MagicMock(return_value=None))
    search_neighbors = MagicMock()
    monkeypatch.setattr(assembly_agent, "search_dense_neighbors", search_neighbors)

    passage = ImagePassage(
        content=b"\x00",
        media_type="image/png",
        caption="An attention plot",
        document_id=uuid4(),
        ordinal="Figure 3",
    )
    embedder = _StubEmbedder()
    result = _run(passage, session=MagicMock(), embedder=embedder)

    assert result.grounded is False
    assert result.cited_passages == []
    assert embedder.embedded == []
    search_neighbors.assert_not_called()


def test_anchored_non_text_without_caption_still_grounds_on_document(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A captionless image still grounds on its valid document anchor alone."""
    doc_chunk_id = uuid4()
    monkeypatch.setattr(
        assembly_agent,
        "fetch_document_chunks",
        MagicMock(return_value=[_doc_chunk(doc_chunk_id, "the figure's section")]),
    )
    search_neighbors = MagicMock()
    monkeypatch.setattr(assembly_agent, "search_dense_neighbors", search_neighbors)

    passage = ImagePassage(
        content=b"\x00",
        media_type="image/png",
        document_id=uuid4(),
        ordinal="Figure 4",
    )
    embedder = _StubEmbedder()
    result = _run(passage, session=MagicMock(), embedder=embedder)

    assert result.grounded is True
    assert [p.chunk_id for p in result.cited_passages] == [doc_chunk_id]
    assert embedder.embedded == []
    search_neighbors.assert_not_called()


def test_anchored_non_text_deduplicates_document_chunks_from_neighbors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A document chunk that also surfaces as a neighbor is cited once."""
    doc_chunk_id = uuid4()
    other_id = uuid4()
    monkeypatch.setattr(
        assembly_agent,
        "fetch_document_chunks",
        MagicMock(return_value=[_doc_chunk(doc_chunk_id, "shared section")]),
    )
    monkeypatch.setattr(
        assembly_agent,
        "search_dense_neighbors",
        MagicMock(
            return_value=[
                _neighbor(doc_chunk_id, "the same section as a neighbor", 0.99),
                _neighbor(other_id, "a distinct neighbor", 0.9),
            ]
        ),
    )

    passage = ImagePassage(
        content=b"\x00",
        media_type="image/png",
        caption="An attention plot",
        document_id=uuid4(),
        ordinal="Figure 5",
    )
    result = _run(passage, session=MagicMock(), embedder=_StubEmbedder())

    assert result.grounded is True
    assert [p.chunk_id for p in result.cited_passages] == [doc_chunk_id, other_id]


def test_anchored_non_text_empty_document_context_is_not_grounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A ready-but-empty document anchors nothing citable, so grounded is False."""
    monkeypatch.setattr(assembly_agent, "fetch_document_chunks", MagicMock(return_value=[]))
    search_neighbors = MagicMock()
    monkeypatch.setattr(assembly_agent, "search_dense_neighbors", search_neighbors)

    passage = ImagePassage(
        content=b"\x00",
        media_type="image/png",
        document_id=uuid4(),
        ordinal="Figure 6",
    )
    embedder = _StubEmbedder()
    result = _run(passage, session=MagicMock(), embedder=embedder)

    assert result.grounded is False
    assert result.cited_passages == []
    assert embedder.embedded == []
    search_neighbors.assert_not_called()


# ============================================================
# Web search: triggered by the brief's search_intent (ticket #244)
# ============================================================


def _passing_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch retrieval so an unanchored text passage clears the similarity gate."""
    monkeypatch.setattr(assembly_agent, "fetch_parent_chunk", MagicMock())
    monkeypatch.setattr(
        assembly_agent,
        "search_dense_neighbors",
        MagicMock(return_value=[_neighbor(uuid4(), "similar chunk", 0.9)]),
    )


def test_assembly_runs_web_search_when_brief_carries_search_intent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unanchored passage with a search intent triggers the web-search step."""
    _passing_gate(monkeypatch)
    results = [
        WebSearchResult(title="Paper", url="https://example.com/paper", snippet="short quote")
    ]
    client = StubWebSearchClient(results=results)

    passage = TextPassage(content="Attention is all you need.")
    result = _run(passage, session=MagicMock(), embedder=_StubEmbedder(), web_search=client)

    assert result.external_search_attempted is True
    assert result.external_search_failed is False
    assert result.external_search_note is None
    assert result.external_search_results == results
    assert client.calls == ["Attention is all you need."]


def test_assembly_skips_web_search_without_search_intent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An anchored passage has no search intent, so the search client is never called."""
    monkeypatch.setattr(
        assembly_agent,
        "fetch_parent_chunk",
        MagicMock(return_value=CitedPassage(chunk_id=uuid4(), text="parent")),
    )
    monkeypatch.setattr(assembly_agent, "search_dense_neighbors", MagicMock(return_value=[]))
    client = StubWebSearchClient()

    passage = TextPassage(content="Attention is all you need.", chunk_id=uuid4())
    result = _run(passage, session=MagicMock(), embedder=_StubEmbedder(), web_search=client)

    assert result.external_search_attempted is False
    assert result.external_search_failed is False
    assert result.external_search_note is None
    assert result.external_search_results == []
    assert client.calls == []


def test_assembly_search_double_failure_sets_failed_and_note(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two failed attempts set the failed flag and carry the user-facing note."""
    _passing_gate(monkeypatch)
    client = StubWebSearchClient(results=[], failures=2)

    passage = TextPassage(content="Attention is all you need.")
    result = _run(passage, session=MagicMock(), embedder=_StubEmbedder(), web_search=client)

    assert result.external_search_attempted is True
    assert result.external_search_failed is True
    assert result.external_search_note == FALLBACK_NOTE
    assert result.external_search_results == []
    assert len(client.calls) == 2


def test_assembly_search_recovers_on_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    """A transient failure is retried exactly once and the search succeeds."""
    _passing_gate(monkeypatch)
    results = [
        WebSearchResult(title="Recovered", url="https://example.com/recovered", snippet="span")
    ]
    client = StubWebSearchClient(results=results, failures=1)

    passage = TextPassage(content="Attention is all you need.")
    result = _run(passage, session=MagicMock(), embedder=_StubEmbedder(), web_search=client)

    assert result.external_search_attempted is True
    assert result.external_search_failed is False
    assert result.external_search_results == results
    assert len(client.calls) == 2


def test_assembly_search_failure_still_reports_grounding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed search does not degrade the corpus-grounding outcome."""
    _passing_gate(monkeypatch)
    close_id = uuid4()
    monkeypatch.setattr(
        assembly_agent,
        "search_dense_neighbors",
        MagicMock(return_value=[_neighbor(close_id, "similar chunk", 0.9)]),
    )
    client = StubWebSearchClient(results=[], failures=2)

    passage = TextPassage(content="Attention is all you need.")
    result = _run(passage, session=MagicMock(), embedder=_StubEmbedder(), web_search=client)

    assert result.grounded is True
    assert [p.chunk_id for p in result.cited_passages] == [close_id]
    assert result.external_search_failed is True


# ============================================================
# Generation turn wiring (ticket #245)
# ============================================================


def test_assembly_passes_citations_and_provider_to_generation(
    monkeypatch: pytest.MonkeyPatch, _stub_generation: MagicMock
) -> None:
    """The generation turn receives the brief, the cited passages, and the provider."""
    parent_id = uuid4()
    monkeypatch.setattr(
        assembly_agent,
        "fetch_parent_chunk",
        MagicMock(return_value=CitedPassage(chunk_id=parent_id, text="parent")),
    )
    monkeypatch.setattr(assembly_agent, "search_dense_neighbors", MagicMock(return_value=[]))
    provider = MockCompletionProvider()
    passage = TextPassage(content="Attention is all you need.", chunk_id=uuid4())
    result = _run(
        passage,
        session=MagicMock(),
        embedder=_StubEmbedder(),
        completion_provider=provider,
    )

    brief = run_framing(passage)
    _stub_generation.assert_called_once()
    args, kwargs = _stub_generation.call_args
    assert args[0] == brief
    assert args[1] == [CitedPassage(chunk_id=parent_id, text="parent")]
    assert args[2] == []
    assert kwargs["completion_provider"] is provider
    assert result.animation == _stub_generation.return_value.animation


def test_assembly_passes_search_results_to_generation(
    monkeypatch: pytest.MonkeyPatch, _stub_generation: MagicMock
) -> None:
    """Successful web-search material reaches the generation turn."""
    _passing_gate(monkeypatch)
    results = [WebSearchResult(title="Paper", url="https://example.com/paper", snippet="quote")]
    client = StubWebSearchClient(results=results)

    passage = TextPassage(content="Attention is all you need.")
    _run(passage, session=MagicMock(), embedder=_StubEmbedder(), web_search=client)

    _stub_generation.assert_called_once()
    assert _stub_generation.call_args.args[2] == results


def test_assembly_skips_generation_search_results_when_none(
    monkeypatch: pytest.MonkeyPatch, _stub_generation: MagicMock
) -> None:
    """An anchored passage generates with an empty search-results list."""
    monkeypatch.setattr(
        assembly_agent,
        "fetch_parent_chunk",
        MagicMock(return_value=CitedPassage(chunk_id=uuid4(), text="parent")),
    )
    monkeypatch.setattr(assembly_agent, "search_dense_neighbors", MagicMock(return_value=[]))

    passage = TextPassage(content="Attention is all you need.", chunk_id=uuid4())
    _run(passage, session=MagicMock(), embedder=_StubEmbedder())

    _stub_generation.assert_called_once()
    assert _stub_generation.call_args.args[2] == []


def test_assembly_surfaces_generated_animation(
    monkeypatch: pytest.MonkeyPatch, _stub_generation: MagicMock
) -> None:
    """The assembly result carries the scene graph produced by the generation turn."""
    _passing_gate(monkeypatch)
    result = _run(
        TextPassage(content="Attention is all you need."),
        session=MagicMock(),
        embedder=_StubEmbedder(),
    )
    assert result.animation == _stub_generation.return_value.animation


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
