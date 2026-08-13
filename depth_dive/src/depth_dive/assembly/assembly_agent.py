"""Assembly agent implementation (ADR-0020, ticket #243).

Consumes the framing brief and grounds the dive against the ingested corpus
through the shared ``core/retrieval/`` primitives (ADR-0019):

- An anchored ``text``/``code`` passage (``chunk_id``) fetches the enclosing
  Parent Chunk plus up to K semantic neighbors from the global corpus.
- Every other passage runs the corpus-similarity gate on the passage's
  embeddable representation; the gate's threshold and grounded/unverified
  judgement are Depth Dive harness policy (ADR-0019, ADR-0020).

The brief's ``search_intent`` is consumed by the web-search step once it
lands (ticket #244); the artifact payload stays the hardcoded demo until LLM
generation lands (ticket #245).
"""

from uuid import UUID

from pydantic import BaseModel, ConfigDict
from sqlalchemy.orm import Session

from core.clients import Embedder
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
from core.types.responses import CitedPassage
from core.types.retrieval_config import RetrievalConfig
from depth_dive.framing.framing_agent import FramingBrief

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
    """The corpus-grounding outcome of the assembly agent.

    ``grounded`` follows the Harness A semantics (same field on
    ``HarnessBResponse``): True when the ingested corpus vouches for the
    dive. ``cited_passages`` is empty whenever ``grounded`` is False.
    """

    model_config = ConfigDict(extra="forbid")

    grounded: bool
    cited_passages: list[CitedPassage]


def run_assembly(
    passage: CapturedPassage,
    brief: FramingBrief,
    *,
    session: Session,
    embedder: Embedder,
    config: RetrievalConfig,
) -> GroundingResult:
    """Ground one captured passage against the ingested corpus.

    Args:
        passage: The captured passage carried in the request.
        brief: The framing agent's creative brief for this dive. Its
            ``search_intent`` drives the web-search step (ticket #244);
            corpus grounding in this slice is keyed on the passage's anchor.
        session: SQLAlchemy session bound to the documents database.
        embedder: Provider used to embed the passage content (1536-dim).
        config: Retrieval configuration (model name, ef_search, top_k).

    Returns:
        A ``GroundingResult`` carrying the grounded flag and the cited
        passages. An unresolvable anchor or a failed similarity gate is a
        valid ungrounded result, never an exception.

    Raises:
        UpstreamBadResponse: The embeddings API returned an unexpected
            response (route maps to 502).
        UpstreamUnavailable: The embeddings API or the database was
            unreachable (route maps to 503).
    """
    if isinstance(passage, (TextPassage, CodePassage)) and passage.chunk_id is not None:
        return _ground_anchored(
            passage, passage.chunk_id, session=session, embedder=embedder, config=config
        )
    return _ground_via_gate(passage, session=session, embedder=embedder, config=config)


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


__all__ = ["SIMILARITY_GATE_THRESHOLD", "GroundingResult", "run_assembly"]
