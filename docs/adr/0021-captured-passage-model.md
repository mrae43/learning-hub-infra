# 0021 — Captured Passage Model

## Status

Accepted

## Context

`ADR-0014` named `chunk_id` the canonical Captured Passage anchor and deferred the full Captured Passage shape to a future Harness B design session. That session (see map #215) had to account for non-text passages — images, diagrams, tables — that do not fit the text-chunk model. The store is text-only today: chunkers hard-fail on non-text and `chunks.content` is `TEXT`. The redefinition therefore needed a passage model that treats non-text as first-class, keeps the request self-contained, and remains compatible with the existing `chunk_id` anchor for text and code.

## Decision

`CapturedPassage` is a single Pydantic discriminated union keyed on `passage_type` with five variants: **text, image, diagram, table, code**.

### Content location

The content is **carried in the request for all five types**. The store is never the content source for a captured passage; the anchor is provenance/retrieval context, not content.

### Anchoring

| Variant | Anchor | Notes |
|---|---|---|
| `text` | `chunk_id: UUID \| None` | `ADR-0014`'s canonical anchor survives. |
| `code` | `chunk_id: UUID \| None` | Reuses existing text-chunk anchor. |
| `image` | `document_id: UUID \| None`, `ordinal: str \| None` | Parallel anchor; no non-text chunks exist today. |
| `diagram` | `document_id: UUID \| None`, `ordinal: str \| None` | Parallel anchor. |
| `table` | `document_id: UUID \| None`, `ordinal: str \| None` | Parallel anchor. |

Anchor is optional on every variant. Unanchored passages may carry an optional `source` URL as provenance only; it is not authoritative.

### Per-variant shape

- **TextPassage:** `content: str`, optional `chunk_id`, optional `source`.
- **CodePassage:** `content: str`, `language: str \| None`, optional `chunk_id`, optional `source`.
- **ImagePassage:** `content: bytes`, `media_type`, optional `caption`, optional `document_id`/`ordinal`, optional `source`.
- **DiagramPassage:** same carrier as `ImagePassage`; may feed an internal graph IR for harness routing/assembly, but that IR is never sent to the model as a graph.
- **TablePassage:** structured `rows` (+ optional `headers`), optional `caption`, optional `document_id`/`ordinal`, optional `source`.

### Identity

A Captured Passage is identified by **content + anchor**. It is ephemeral: no server-assigned id, never persisted.

### Detection

Detection is explicit via the `passage_type` discriminator. The harness validates that carried content matches the claimed type; content sniffing is reserved for validation failures, not primary routing.

## Consequences

- `CapturedPassage` lives in `core/types/captured_passage.py`; `HarnessBRequest` references it from `core/types/depth_dive.py`.
- The storage contract for non-text passages (image/diagram/table) is a requirement handed to the future ingestion-redesign map: store must provide retrievable references and text-serializable/embeddable representations under `document_id` + ordinal identity.
- Unanchored passages of any type trigger a corpus-similarity gate before output proceeds; the gate result surfaces in `HarnessBResponse.grounded`.
- The rejection of "text representations of non-text" (e.g., markdown tables, captions-as-images) is now encoded in the boundary type: tables carry structured rows, images/diagrams carry bytes, and no variant reduces to plain text.
