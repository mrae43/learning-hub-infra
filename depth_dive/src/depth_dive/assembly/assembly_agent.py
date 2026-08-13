"""Assembly agent implementation (ADR-0020, tickets #243, #244).

Consumes the framing brief and grounds the dive against the ingested corpus
through the shared ``core/retrieval/`` primitives (ADR-0019):

- An anchored ``text``/``code`` passage (``chunk_id``) fetches the enclosing
  Parent Chunk plus up to K semantic neighbors from the global corpus.
- Every other passage runs the corpus-similarity gate on the passage's
  embeddable representation; the gate's threshold and grounded/unverified
  judgement are Depth Dive harness policy (ADR-0019, ADR-0020).

The brief's ``search_intent`` drives the web-search step (ticket #244): when
the framing agent judged external grounding would help (ADR-0012), the
assembly agent calls the retry-once web-search wrapper (ADR-0013) and surfaces
the outcome on :class:`AssemblyResult`. The final turn is the LLM generation
step (ticket #245), which builds a scene graph from the brief, the cited
passages, and any search results.
"""

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.orm import Session

from core.clients import CompletionProvider, Embedder
from core.retrieval import fetch_parent_chunk, search_dense_neighbors
from core.types.captured_passage import (
    CapturedPassage,
    CodePassage,
    DiagramPassage,
    ImagePassage,
    TablePassage,
    TableRow,
    TextPassage,
)
from core.types.depth_dive import InteractiveAnimation
from core.types.responses import CitedPassage
from core.types.retrieval_config import RetrievalConfig
from depth_dive.framing.framing_agent import FramingBrief
from depth_dive.generation.generation_agent import run_generation
from depth_dive.web_search.client import WebSearchClient, WebSearchResult
from depth_dive.web_search.wrapper import SearchOutcome, run_web_search

SIMILARITY_GATE_THRESHOLD: float = 0.5
"""Cosine-similarity floor for the unanchored-passage corpus gate.

Depth Dive harness policy (ADR-0019, ADR-0020): an unanchored passage counts
as grounded only when its closest corpus chunk is at least this similar — a
close paraphrase or the passage itself under different chunk boundaries.
Below the floor the passage is unverified: the dive still proceeds, but
``grounded`` stays False. The floor is an MVP starting point, to be tuned by
retrieval evaluation.
"""


class GroundingResult(BaseModel):
    """The corpus-grounding outcome for one dive.

    ``grounded`` follows the Harness A semantics (same field on
    ``HarnessBResponse``): True when the ingested corpus vouches for the dive.
    ``cited_passages`` is empty whenever ``grounded`` is False. Internal to
    the assembly agent — :class:`AssemblyResult` composes it with the
    web-search outcome.
    """

    model_config = ConfigDict(extra="forbid")

    grounded: bool
    cited_passages: list[CitedPassage]


class AssemblyResult(BaseModel):
    """The assembly agent's outcome for one dive.

    ``grounded`` and ``cited_passages`` follow the Harness A semantics (same
    fields on ``HarnessBResponse``): True when the ingested corpus vouches for
    the dive, and ``cited_passages`` empty whenever ``grounded`` is False.
    ``external_search_*`` carry the web-search outcome (ADR-0013), and
    ``external_search_results`` the surfaced external material for the
    generation turn. ``animation`` is the generated ``interactive_animation``
    scene graph — the model's output, or the minimal fallback scene graph when
    that output was malformed (ticket #245).
    """

    model_config = ConfigDict(extra="forbid")

    grounded: bool
    cited_passages: list[CitedPassage]
    external_search_attempted: bool = False
    external_search_failed: bool = False
    external_search_note: str | None = None
    external_search_results: list[WebSearchResult] = Field(default_factory=list)
    animation: InteractiveAnimation


def run_assembly(
    passage: CapturedPassage,
    brief: FramingBrief,
    *,
    session: Session,
    embedder: Embedder,
    config: RetrievalConfig,
    web_search: WebSearchClient,
    completion_provider: CompletionProvider,
) -> AssemblyResult:
    """Ground one captured passage, run the web-search step, and generate.

    Args:
        passage: The captured passage carried in the request.
        brief: The framing agent's creative brief for this dive. Its
            ``search_intent`` drives the web-search step (ADR-0012, ADR-0013);
            corpus grounding is keyed on the passage's anchor.
        session: SQLAlchemy session bound to the documents database.
        embedder: Provider used to embed the passage content (1536-dim).
        config: Retrieval configuration (model name, ef_search, top_k).
        web_search: Provider for the web-search step; called only when the
            brief carries a ``search_intent``.
        completion_provider: The chat-completions provider used by the LLM
            generation turn (ticket #245).

    Returns:
        An ``AssemblyResult`` carrying the grounded flag, the cited passages,
        the web-search outcome, and the generated scene graph. An unresolvable
        anchor or a failed similarity gate is a valid ungrounded result, a
        failed search is a valid degraded outcome (with a note), and malformed
        model output falls back to a minimal valid scene graph — never
        exceptions.

    Raises:
        UpstreamBadResponse: The embeddings API or the inference API returned
            an unexpected response (route maps to 502).
        UpstreamUnavailable: The embeddings API, the database, or the
            inference API was unreachable (route maps to 503).
    """
    if isinstance(passage, (TextPassage, CodePassage)) and passage.chunk_id is not None:
        grounding = _ground_anchored(
            passage, passage.chunk_id, session=session, embedder=embedder, config=config
        )
    else:
        grounding = _ground_via_gate(passage, session=session, embedder=embedder, config=config)
    outcome = _search_outcome(brief, web_search)
    generation = run_generation(
        brief,
        grounding.cited_passages,
        outcome.results,
        completion_provider=completion_provider,
    )
    return AssemblyResult(
        grounded=grounding.grounded,
        cited_passages=grounding.cited_passages,
        external_search_attempted=outcome.attempted,
        external_search_failed=outcome.failed,
        external_search_note=outcome.note,
        external_search_results=outcome.results,
        animation=generation.animation,
    )


def _search_outcome(brief: FramingBrief, web_search: WebSearchClient) -> SearchOutcome:
    """Run the web-search step when the brief carries a search intent.

    Called only when the framing agent judged external grounding would help
    (ADR-0012) — the brief's ``search_intent`` is present. Otherwise the step
    is a no-op and no search call is made.
    """
    if not brief.search_intent:
        return SearchOutcome(attempted=False)
    return run_web_search(brief.search_intent, web_search)


def _ground_anchored(
    passage: TextPassage | CodePassage,
    chunk_id: UUID,
    *,
    session: Session,
    embedder: Embedder,
    config: RetrievalConfig,
) -> GroundingResult:
    """Ground an anchored text/code passage: parent chunk + K neighbors.

    The parent chunk leads the citations; neighbors follow in cosine-closeness
    order. The anchor chunk itself is skipped (its content is inside the
    parent) and duplicate ids are cited once.
    """
    parent = fetch_parent_chunk(session=session, chunk_id=chunk_id)
    if parent is None:
        return GroundingResult(grounded=False, cited_passages=[])
    query_vector = embedder.embed([passage.content])[0]
    neighbors = search_dense_neighbors(session=session, query_vector=query_vector, config=config)
    cited = [parent]
    seen = {parent.chunk_id, chunk_id}
    for neighbor in neighbors:
        if neighbor.chunk_id in seen:
            continue
        seen.add(neighbor.chunk_id)
        cited.append(CitedPassage(chunk_id=neighbor.chunk_id, text=neighbor.text))
    return GroundingResult(grounded=True, cited_passages=cited)


def _ground_via_gate(
    passage: CapturedPassage,
    *,
    session: Session,
    embedder: Embedder,
    config: RetrievalConfig,
) -> GroundingResult:
    """Ground an unanchored passage through the corpus-similarity gate.

    The closest corpus chunk decides the gate: at or above
    ``SIMILARITY_GATE_THRESHOLD`` the passage is grounded and the passing
    neighbors are cited; below it (or with nothing to embed, or an empty
    corpus) the passage is unverified and ungrounded.
    """
    embeddable = _embeddable_text(passage)
    if embeddable is None:
        return GroundingResult(grounded=False, cited_passages=[])
    query_vector = embedder.embed([embeddable])[0]
    neighbors = search_dense_neighbors(session=session, query_vector=query_vector, config=config)
    if not neighbors or neighbors[0].score < SIMILARITY_GATE_THRESHOLD:
        return GroundingResult(grounded=False, cited_passages=[])
    cited = [
        CitedPassage(chunk_id=n.chunk_id, text=n.text)
        for n in neighbors
        if n.score >= SIMILARITY_GATE_THRESHOLD
    ]
    return GroundingResult(grounded=True, cited_passages=cited)


def _embeddable_text(passage: CapturedPassage) -> str | None:
    """Derive the passage's embeddable text representation, if it has one.

    Text/code embed their content; image/diagram embed their caption (the
    corpus-side text-serializable form per the storage contract, depth-dive
    spec §10); tables embed a deterministic serialization of headers and rows.
    ``None`` means there is nothing to embed, so the gate cannot verify the
    passage.
    """
    if isinstance(passage, (TextPassage, CodePassage)):
        return passage.content
    if isinstance(passage, (ImagePassage, DiagramPassage)):
        caption = (passage.caption or "").strip()
        return caption or None
    if isinstance(passage, TablePassage):
        return _serialize_table(passage)
    return None


def _serialize_table(passage: TablePassage) -> str | None:
    """Serialize a table to a deterministic embeddable text form.

    Caption (when present), then headers, then one line per row; cells joined
    with ``" | "`` and dict rows rendered as ``key: value`` cells. Returns
    ``None`` when the table carries no caption, headers, or rows.
    """
    lines: list[str] = []
    caption = (passage.caption or "").strip()
    if caption:
        lines.append(caption)
    if passage.headers:
        lines.append(" | ".join(passage.headers))
    for row in passage.rows:
        lines.append(_serialize_row(row))
    return "\n".join(lines) if lines else None


def _serialize_row(row: TableRow) -> str:
    if isinstance(row, dict):
        return " | ".join(f"{key}: {value}" for key, value in row.items())
    return " | ".join(row)


__all__ = ["SIMILARITY_GATE_THRESHOLD", "AssemblyResult", "run_assembly"]
