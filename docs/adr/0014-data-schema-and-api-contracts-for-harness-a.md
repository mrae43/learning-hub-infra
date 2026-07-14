# 0014 — Data Schema and API Contracts for Harness A

## Status
Accepted

## Context
ADR-0002 chose pgvector for MVP storage; ADR-0004 chose hosted embeddings but left the specific model (and therefore the embedding dimensionality) abstract; ADR-0006 chose FastAPI `BackgroundTasks` for ingestion with no synchronous status visibility; the original (now-superseded) `next-task.md` sketched a 3-field `HarnessAResponse` (formalized in ADR-0009). None of these ADRs pinned the concrete schema (tables, columns, indexes, distance metric), the Pydantic I/O models, or the HTTP endpoint contracts (paths, status codes, error mapping). This ADR records the interlocked decisions made in a single `/grill-with-docs` session that walked the design tree from embeddings model → vector index → storage shape → schema → request models → response models → endpoint contracts → error mapping.

These three layers (schema, models, contracts) are recorded in one ADR because they are interdependent: the API contract reflects the DB schema, and the Pydantic models are the bridge between them (per the brief's "design in that order: schema first").

## Decision

### Embeddings model and dimensionality

MVP uses **OpenAI `text-embedding-3-small`** producing **1536-dim** vectors. ADR-0004 listed Google `text-embedding-004` or OpenAI as the candidate pair; OpenAI is chosen here because ADR-0001 already commits to a hosted inference API from the same vendor (Claude/OpenAI), so pairing embeddings with the same vendor means one client SDK, one auth/billing account, and one config knob to wire.

**Explicit MVP-era constraint:** all embedding models used during MVP share the 1536-dim vector column. A dimension change is a *breaking migration* (new table or new column) when it happens — not a silent polymorphic switch. `model_name` is provenance ("which model produced this vector"), not a dimensional selector.

### pgvector index and distance metric

- **Distance metric: cosine.** `text-embedding-3-small` vectors are L2-normalized to unit length (per OpenAI's embeddings guide), so cosine similarity is recommended and ranking-equivalent to dot-product / L2 on these vectors. The query-side uses pgvector's `<=>` (cosine distance) operator, matched by the HNSW index's `vector_cosine_ops` opclass — the metric is consistent between index and query.
- **Index type: HNSW from day one**, declared in the same Alembic migration as the table (not retrofitted later). Defaults `m=16`, `ef_construction=64` per pgvector's recommendations. HNSW over IVFFlat because the MVP corpus is small (hundreds to low-thousands of chunks) and IVFFlat's training-step / `lists`-tuning buys nothing at that scale; HNSW has no training step and better recall at small/medium scale.
- **Query-time knob: `hnsw.ef_search`** lives in `core/config/settings.py` as `settings.hnsw_ef_search: int = 40`. Retrieval issues `SET LOCAL hnsw.ef_search = :val` inside the retrieval transaction. Default 40 (pgvector's default); ADR-0007's recall@k eval gives the empirical signal to raise it. Operator-controlled via env var, invisible to API clients — consistent with the principle that infra-tuning knobs live in config, not the API contract. Wiring the knob now costs one setting + one SET LOCAL; doing it later (when eval proves recall needs it) is a code change, not a contract change.

### Documents table

```
documents
── document_id    UUID PK, default app-side uuid7
── title          TEXT NOT NULL
── document_type  ENUM('paper','book','documentation') NOT NULL
── source_filename TEXT NOT NULL
── status         ENUM('validating','chunking','embedding','ready','failed')
                  NOT NULL DEFAULT 'validating'
── error_message  TEXT NULL, CHECK (error_message IS NULL OR status = 'failed')
── created_at     TIMESTAMPTZ NOT NULL DEFAULT now()
── updated_at     TIMESTAMPTZ NOT NULL DEFAULT now()  onupdate=func.now()
```

- **`document_type` is a Postgres `ENUM`** — closed known set per CONTEXT.md's "Document-Type Chunker" definition; DB catches bad values at write time. Not free TEXT because the set is genuinely closed (paper/book/documentation, three chunkers in `retrieval_qa/chunking/`).
- **`status` is a five-state enum in pipeline-phase declaration order:** `validating → chunking → embedding → ready`, with `failed` reachable from any phase. `validating` is the entry state (pending was rejected as ambiguous alongside the granular phase states). Each named state maps to one actual code path in `ingestion/pipeline.py`; the granular five-state enables a useful status-check endpoint ("still chunking" vs "now embedding") rather than a minimal "processing" umbrella that would have to be disambiguated later.
- **`error_message` populated only when `status='failed'`**, enforced by a CHECK constraint. The not-found case in Retrieval QA (`grounded=False`) is *not* an exception (ADR-0009); this column records genuine pipeline failures (parse errors, embeddings API failures, disk write errors), not "no good answer found."
- **Atomic per-document rollback with `document_id` retention.** A failed `documents` row retains its `document_id` and `error_message`; partial chunks written before the failure are rolled back via the transaction. The user retries the whole document, not partial chunks.
- **Re-ingestion = new `document_id`.** Re-uploading a document creates a new `documents` row; the old row is untouched, and old chunks/embeddings from the prior ingestion live in the corpus alongside the fresh ones. Duplicate chunks are an accepted MVP tradeoff (not an oversight) — supersedence happens at the *document* level (deleting the old document retires its chunks), not the chunk level. The alternative — a `superseded`/`superseded_by` column on chunks — was rejected as over-engineering for an MVP where the user can delete old documents directly.
- **`updated_at` uses `onupdate=func.now()`** so every `UPDATE` (status transitions, error_message sets) auto-bumps the timestamp server-side. The status-check endpoint gets a free "last activity" timestamp without pipeline code remembering to set it.
- **UUID strategy: app-side UUIDv7** via a small library (`uuid_utils.uuid7()`), stored as a plain `UUID` column. Chosen over DB-side `uuidv7()` (PG 18+) because no Postgres version is pinned anywhere in the project (no compose file, no infra ADR), and app-side generation keeps the schema portable across PG 13–18. Upgrade to PG 18 native `uuidv7()` later is a one-line default swap with no data migration (column type stays `UUID`). Chosen over `uuid.uuid4` because UUIDv7's time-ordered primary keys give B-tree natural insertion-order clustering — directly serving retrieval's "recent chunks surface often" pattern and enabling range scans over recent chunks without an extra `created_at` index.

### Chunks table

```
chunks
── chunk_id        UUID PK, default app-side uuid7
── document_id     UUID FK → documents(document_id) ON DELETE CASCADE
── position        INTEGER NOT NULL, CHECK (position >= 0)
── content         TEXT NOT NULL
── token_count     INTEGER NOT NULL, CHECK (token_count > 0)
── type_metadata   JSONB NOT NULL DEFAULT '{}'::jsonb
── created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
UNIQUE (document_id, position)
```

- **Hybrid metadata pattern:** typed common columns (`position`, `token_count`, `content`) plus a single `type_metadata: JSONB` column for fields that vary by document type. Avoids both the ever-growing-nullable-columns table (one fat table with `section_name`, `chapter_title`, `page_number` all nullable) and the fully-opaque JSON blob (which loses queryability on the common fields). Pattern recommended by the brief's "General extensibility pattern" section.
- **`position INTEGER` = zero-indexed within-document order**, scoped by `UNIQUE (document_id, position)` so no two chunks of the same document share a position. Within-document (not corpus-wide) because cross-document "neighbor" chunks aren't a concept — fetching adjacent chunks is a single-document operation (`WHERE document_id = ? AND position BETWEEN ?-1 AND ?+1`).
- **`token_count INTEGER` with `CHECK (token_count > 0)`** — lower bound catches chunker bugs (empty/whitespace-only chunks); no upper bound couples the schema to context-window assumptions (which change with inference model swaps).
- **Chunks are immutable once written.** There is **no `updated_at` column** on chunks. This is a deliberate exception to AGENTS.md's "Add created_at/updated_at timestamps … to every table from day one" rule — recorded explicitly here so a future contributor (or AI session) doesn't "fix" it by re-adding `updated_at` thinking it's a bug. Rationale: under this ADR's lock that re-chunking = new `document_id` (not chunk mutation), chunks never change in place; `updated_at` would always equal `created_at` and the "row changes over time" signal would be a lie. `created_at` stays for "when was this chunk ingested" observability.
- **No `status` column on chunks.** Chunk visibility is gated by their parent document's status — retrieval queries `JOIN documents ON status='ready'`. Chunks lifecycle is tied to document lifecycle, not a per-row flag.
- **`ON DELETE CASCADE` on `document_id` FK** — deleting a document removes its chunks and (via the embeddings table's cascade) their embeddings in one transaction. Single ownership chain: document owns chunks owns embeddings.

### Chunk metadata shape contract (Pydantic registry)

Schema-level chunk metadata is JSONB; the *shape contract* is enforced in Python via a Pydantic registry in `core/types/chunk.py`:

- `PaperChunkMetadata{section: str, subsection: str | None, page: int}`
- `BookChunkMetadata{chapter: int, heading: str | None}`
- `DocumentationChunkMetadata{page: str, section: str | None}`

All three with `model_config = ConfigDict(extra="forbid")` — a chunker bug surfacing an unknown key fails at the application write boundary, before persisting unvalidated JSONB. The chunker implementations (in `retrieval_qa/chunking/`, per `ai-system-tree.md`) and the retrieval code (in `retrieval_qa/retrieval/`) both import from `core/types/`; both depend on `core`, never on each other (ADR-0011 preserved).

Rejected alternatives:
- **Pure JSONB with no Python contract** (Option A) — leaves the contract implicit in two places' heads; a chunker bug that writes `{page: "seven"}` (page as string) lands in the DB silently.
- **Separate child tables per document type** (`paper_chunks`, `book_chunks`, etc., Option C) — fully relational and queryable, but adds 3 tables and union-queries for retrieval's actual access pattern (which never filters by `type_metadata.section = '…'`). The migration to Option C later, if a real query need emerges, is mechanical (`ADD COLUMN` per type + backfill from JSONB + `DROP type_metadata`) with no data loss — well-trodden path, not a one-way door.

### Embeddings table

```
embeddings
── chunk_id     UUID FK → chunks(chunk_id) ON DELETE CASCADE
── model_name   TEXT NOT NULL
── embedding    Vector(1536) NOT NULL
── created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
PRIMARY KEY (chunk_id, model_name)
-- HNSW index on embeddings.embedding using vector_cosine_ops (m=16, ef_construction=64)
```

- **Separate table, not an `embedding` column on `chunks`.** Separating the vector from the chunk lets multiple models' vectors coexist for one chunk (keyed by `(chunk_id, model_name)`), so a model swap (ADR-0004 defines this as a deferred-but-real graduation trigger) can compare old vs new recall before retiring old vectors. Removing the ~6KB `Vector(1536)` from the chunks table also keeps chunk-metadata queries (position, section, token_count) from scanning past wide vector columns.
- **Composite natural PK `(chunk_id, model_name)`**, no surrogate `embedding_id` UUID. Embeddings are never externally referenced (no Captured Passage points to an embedding; ADR-0007's eval set points to chunks, not vectors). The natural pair *is* the identity; a surrogate UUID would be a column for nothing.
- **`model_name TEXT`, not an enum.** The set of models isn't genuinely closed — ADR-0004 explicitly lists local `sentence-transformers` (BGE, e5 families) as a deferred graduation path, and those model IDs aren't known today. A Postgres ENUM would require a migration for every model addition; free TEXT with the application layer as the single writer (only `settings.embedding_model` is ever written) gives the same typo safety without the friction. The dangerous case (wrong-dim vectors) is already caught by pgvector's `Vector(1536)` type enforcement at write time; the wrong-same-dim-model case is exactly the provenance question TEXT is meant to answer.
- **No `updated_at` column** on embeddings. Symmetric with the chunks-table reasoning: embeddings are write-once (a model swap writes new rows under the new `model_name`; re-chunking = new `document_id` cascades to delete old chunks and their embeddings). `created_at` stays for "when was this vector written" observability — useful during a model swap to see which rows belong to which generation.
- **HNSW index with `vector_cosine_ops`** declared in the same migration as the table; `m=16`, `ef_construction=64` defaults per the index-type decision above. Consistent with the query-side `<=>` operator (cosine distance). The brief's "Keep the distance metric decision consistent between the index definition and the query-side `.cosine_distance()` call" is enforced by opclass selection.

### Query request model

```python
class HarnessARequest(BaseModel):
    query: str
```

- **Bare string, no filters.** CONTEXT.md defines "Cross-Reference (implicit)" as the default retrieval mode across the *global* ingested corpus — filtering to one document is not in Retrieval QA's product mental model at MVP (no UI flow produces a `document_id` to filter on; "currently reading a document" is a Captured Passage / Depth Dive construct, per CONTEXT.md). Adding optional `document_id`/`document_type` filters "because it's cheap" is the API-YAGNI trap — clients start depending on them, the contract freezes, and we support a filter nobody uses. The conversation is: add a v2 request shape *if/when* a real filter use case emerges (e.g. "ask questions about the paper I'm currently annotating"); adding optional fields later is backward-compatible.
- **`top_k` is server-side config, not a client knob.** Symmetric with the `ef_search` decision: `top_k` is a retrieval-tuning parameter evaluated against ADR-0007's recall@k signal, not a client-controlled contract surface. Exposing it would freeze an infra-internal knob into the API and force every future vector-store migration (Qdrant per ADR-0002) to preserve it.

### Response model — full `HarnessAResponse`

Final shape per ADR-0009 (this ADR records the schema-level contracts it depends on):

```python
class CitedPassage(BaseModel):
    chunk_id: UUID
    text: str  # full chunk content; client decides how much to render

class HarnessAResponse(BaseModel):
    answer: str                  # always populated; model refusal text when grounded=False
    cited_passages: list[CitedPassage]  # empty when grounded=False
    grounded: bool
```

- **`cited_passages` is a list of nested `CitedPassage` objects,** not `list[UUID]` or `list[str]`. The chunk content is already loaded into retrieval's memory for prompt construction, so `text` costs no extra DB call. Richer nested type is backward-compatible to extend with optional fields; a bare `list[str]` is a frozen contract the moment any client ships against it.
- **`text` carries the full chunk content,** not a truncated preview. The API contract carries the data; presentation (truncation for UI display) is the client's problem. Naming the field `text` (not `preview_text`) is honest about what it carries; the brief's "preview_text" naming was shorthand for "the text you'd preview," but a field named `preview_text` that ships full content would be a naming-content mismatch.
- **`answer: str` is always populated, never nullable.** Both the grounded branch (`answer` = grounded response, `cited_passages` populated) and the not-found branch (`answer` = model-generated refusal, `cited_passages=[]`, `grounded=False`) keep the field populated. No `answer: str | None`; the response shape is uniform across both branches. Internal not-found reasons (empty corpus, threshold miss) are written to server logs, not exposed in the response.
- **No observability fields** (`latency_ms`, `model_used`) on the response body. Per the same logic as `ef_search`: server-side internals don't belong in the client contract. A v2 response shape adding an optional `metadata: ResponseMetadata` block is non-breaking if/when observability needs a UI surface.

### Ingestion request contract

- **Endpoint:** `POST /ingest`
- **Request shape:** multipart/form-data with `file: UploadFile`, `title: str = Form(...)`, `document_type: DocumentType = Form(...)`. `source_filename` derived from `UploadFile.filename` server-side — no separate metadata field, single source of truth.
- **Response:** **202 Accepted** with `Location: /documents/{document_id}` header and `DocumentStatusResponse` body (full documents row). 202 (not 201) is RFC 9110 §15.3.5's explicit async contract code — "request accepted for processing, but processing not yet complete." 201 would imply "you can now GET this resource in its final state," which ingestion isn't (it's still chunking/embedding).
- **Pre-flight contract for size/type:** 413 (file too large) and 415 (unsupported file type) are wired as *contract shapes* now — FastAPI checks file extension against a settings allowlist and file size against a settings max *at upload time*. The actual allowlist values and max size are deferred to the ingestion-pipeline session; the *contract* (these codes can be returned) belongs to the API surface this session owns. Deeper content validation (PDF parseable? EPUB valid?) stays in the async `validating` phase.

### Status-check endpoint

- **Endpoint:** `GET /documents/{document_id}` with `document_id: UUID` path param (FastAPI validates UUID format, 422 on malformed)
- **Response:** 200 OK + `DocumentStatusResponse` (same model as the 202 ingestion response — `from_attributes=True` makes the route one `SELECT` + `model_validate`). Path param named `document_id` (the domain identity), not `uuid` (the type) — matches the field name across `DocumentStatusResponse.document_id`, the `documents.document_id` column, and the 202's `Location` header URL.
- **Exists at MVP** (not deferred until ADR-0006's graduation trigger fires). The contract cost is one trivial endpoint + the existing model; the visibility quality (granular phase ETA, per-chunk progress) is what's worth deferring. Wiring the contract now means ADR-0006's "first real need for visibility" trigger fires on *use*, not on *adding the endpoint*.

```python
class DocumentStatusResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    document_id: UUID
    title: str
    document_type: DocumentType
    status: DocumentStatus
    error_message: str | None
    source_filename: str
    created_at: datetime
    updated_at: datetime
```

`DocumentStatusResponse` is reused across both endpoints (202 upload + 200 status) — one type, one source of truth; clients parse both responses identically.

- **No chunk listing endpoint in MVP.** `GET /documents/{document_id}/chunks` (Option C) was rejected — no UI surfaces chunks, and ADR-0007's eval harness uses direct DB access, not the user-facing API. Coupling the user-facing status contract to an internal-harness need is bad layering.

### Query endpoint

- **Endpoint:** `POST /query`
- **Request body:** `HarnessARequest{ query: str }` (per the model above)
- **Response:** 200 OK + `HarnessAResponse` (per the model above)

### Error contract mapping

| Endpoint / Failure mode | Status | Body | Named exception |
|---|---|---|---|
| Any endpoint / request validation | 422 | FastAPI default `{detail: ...}` | (Pydantic raises, FastAPI handles) |
| `POST /ingest` / file too large | 413 | FastAPI default | (route raises `HTTPException`) |
| `POST /ingest` / unsupported file type | 415 | FastAPI default | (route raises `HTTPException`) |
| `POST /ingest` / sync-side failure (disk write, docs insert) | 500 | FastAPI default | `IngestionError` |
| `POST /ingest` / async pipeline failure | (no HTTP response) | — | `IngestionError` caught in BackgroundTask, persists `status='failed'` + `error_message` |
| `POST /query` / upstream API returned bad response (embeddings, inference) | 502 | FastAPI default | `RetrievalError` subclass |
| `POST /query` / upstream unreachable (DB down, upstream API timeout) | 503 | FastAPI default | `RetrievalError` subclass |
| `POST /query` / empty corpus or no relevant chunks | 200 | `HarnessAResponse(grounded=False, ...)` | (not an exception, per ADR-0009) |
| `POST /query` / unexpected internal error | 500 | FastAPI default | (catch-all) |
| `GET /documents/{document_id}` / unknown ID | 404 | FastAPI default | (route raises `HTTPException`) |
| `GET /documents/{document_id}` / DB failure | 500 | FastAPI default | (catch-all) |

- **Named exceptions:** `IngestionError` (raised in ingestion pipeline; sync occurrences → 500, async occurrences → caught in BackgroundTask → `documents.status='failed'` + `error_message`, no HTTP response), `RetrievalError` (raised in retrieval/query path; FastAPI exception handler → 502/503). 
- **`GroundingFailure`** (previously listed in `coding-standards.md` as an example named exception) is **removed** — it contradicts the rule immediately following it ("the not-found case is a valid response, not an exception"). The `coding-standards.md` example is updated to list only `IngestionError` and `RetrievalError`.
- **502 vs 503 split** for `RetrievalError` subclasses: 502 for "upstream returned bad response" vs 503 for "upstream unreachable / timeout" — both subclass `RetrievalError`, distinguished by error type so log inspection can tell them apart. RFC 9110 makes this distinction; preserving it costs one subclass.
- **Error response body reuses FastAPI's default `{"detail": str}`** via `HTTPException(detail=msg)`. A typed `ErrorResponse` Pydantic model would add a wrapper that exists only to carry one string field; `coding-standards.md`'s "Pydantic v2 models are the boundary type for all I/O" targets *resource* I/O (documents, chunks, responses), not the error envelope, which is already standardized by FastAPI. If a future need emerges (machine-readable `code` field, multi-language `detail`), swapping the exception handler to return a typed `ErrorResponse` is non-breaking for clients that read only the `detail` string.

### Captured Passage anchor

CONTEXT.md defines a Captured Passage as the user-selected excerpt anchoring a Depth Dive (Harness B) request — explicitly out of scope for this session. The schema ADR records: **`chunk_id` is the canonical Captured Passage anchor** for the future Harness B session; a `captured_passages` table is **deferred** to that session (no stub, no columns guessed at now). Future Harness B reads "the chunk's UUIDv7 PK is your anchor; FK to `chunks.chunk_id`; design your own table shape around it." This defers cleanly because the schema work this ADR records — stable chunk identity — is exactly the precondition Harness B needs to anchor Captured Passages.

## Considered Options

Documented inline above per decision (embeddings model, index type, UUID strategy, hybrid vs. JSONB-only metadata, separate vs. inline embeddings, bare vs. nested `cited_passages`, single vs. two-step ingestion contract, status endpoint presence, status enum granularity, etc.). The decision for each was driven by the locked rationale above; the rejected alternatives were rejected for the specific reasons given, not on taste.

## Consequences

- **MVP-era dimension is a hard constraint.** All embedding models used in the MVP era are 1536-dim; a dimension change (e.g. swap to a 768-dim model like Google `text-embedding-004`) is a *breaking schema migration* (new column or new table), not a polymorphic switch hidden by `Vector(n)` indirection. This is deliberately not abstracted so a future contributor doesn't silently break the corpus by adding a `dimensions` column "for flexibility."
- **Chunks-immutable exception to AGENTS.md's "every table has `updated_at`" rule is explicit.** Future contributors (or AI sessions) reading AGENTS.md and noticing chunks has no `updated_at` should find this ADR as the documented reason, not reverse the decision to comply with the general rule.
- **Model swap is the primary graduation scenario for the separate embeddings table and `model_name` provenance.** ADR-0004 names local sentence-transformers as the deferred swap; ADR-0014 records that the *swap path* is: write new embeddings under the new `model_name`, change `settings.embedding_model`, compare recall@k (ADR-0007) under old vs new, retire old embeddings by `DELETE FROM embeddings WHERE model_name = :old`. The dimension constraint means this swap path is dialect-in-MVP (1536-dim models only); a cross-dimension swap is the breaking migration above.
- **Re-ingestion creates duplicate corpus content, by design.** Re-uploading a document does not retire the old document's chunks. The user deletes the old document explicitly when they want to retire its chunks. This is a known MVP tradeoff (recorded here, not in CONTEXT.md — it's implementation behavior, not a domain term).
- **The API contract for filters and observability is intentionally minimal.** `HarnessARequest` is a bare string, `HarnessAResponse` is three fields. Adding to either later is backward-compatible *if* the additions are optional fields or nested optional blocks. Going the other direction (deprecating fields) would be breaking, so the bias is toward shipping minimal and growing under real client pressure — the same needs-driven principle established by ADR-0001, ADR-0002, ADR-0004, ADR-0006 applied to the API contract.
- **Operator config (`core/config/settings.py`) carries all infra-internal knobs:** active embedding model, HNSW `ef_search`, file size limit, file type allowlist. None surface in the API contract. This makes the API stable across infra-internal swaps (model change, vector store migration, retry tuning) without breaking clients — the contract is decoupled from infra choices, consistent with ADR-0005's "extractable module" goal extended to the API layer.
- **`import-linter` boundaries** (ADR-0011) are not affected by this ADR's schema decisions — chunker / retrieval code (in `retrieval_qa/`) imports from `core/types/` and `core/database/`, not from each other or from `api/`. The Pydantic metadata registry in `core/types/chunk.py` is the shared boundary type both sides depend on.