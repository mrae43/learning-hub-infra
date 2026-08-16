# CONTEXT.md
> Glossary of domain terms for the Learning Hub project. No implementation details — terms and their meaning only.

## Retrieval QA
The closed-book question-answering system. Scope is strictly bounded to the model plus the user's ingested documents — no web search, no external tools. Output format is plain text. This is Retrieval QA's entire job; it does not produce artifacts, code samples, or cite external sources beyond the ingested corpus.

## Depth Dive
The synthesis harness that consumes the shared retrieval layer (ADR-0019) and produces a self-contained, interactive explanation of a single Captured Passage. The MVP output is one **interactive_animation** scene graph — a declarative, stateless payload the client renders without further server calls — pairing explanatory text with a visual (dual coding). Treatments such as `worked_example`, `prediction_reveal`, and `segmented_carousel` are layered onto the animation. Unlike Retrieval QA, Depth Dive is permitted to extend beyond the ingested corpus via agentic web search (ADR-0012, ADR-0013). Stateless; no quizzing, no scheduling, no per-learner state in this repo. Ships after Retrieval QA is proven, since it depends on shared retrieval quality.

## Document-Type Chunker
A structure-aware chunking strategy specific to each ingested document type, rather than one fixed-size splitter applied uniformly. Papers chunk along section/subsection boundaries, books along chapter/heading boundaries, documentation along page/API-entry boundaries. Chosen as a separation-of-concerns principle for the hand-rolled pipeline (see ADR-0003) — each document type owns its own chunking logic rather than sharing a generic splitter.

## Chunk
The atomic retrievable unit of an ingested document, produced by a Document-Type Chunker. Has stable identity, an ordered position within its source document, and content; once embedded, it becomes retrievable as Injected Context and citable in a Retrieval QA response. Distinct from a "Captured Passage" — a captured passage is user-selected and grounds Depth Dive; a chunk is system-produced and grounds Retrieval QA.

## Ingested Corpus
The full set of user-uploaded papers, books, and documentation, stored in a single **global** vector database (not siloed per document or per session). Retrieval QA retrieves across the entire ingested corpus by default — cross-document QA (e.g. "compare how paper X and paper Y define attention") is in scope, not a special mode.

## Injected Context
The specific chunks retrieved for a given query and placed into the model's prompt at inference time. A per-query subset of the ingested corpus — not the whole corpus, and never anything from outside it (no web search) for Retrieval QA.

## Captured Passage
The specific content a user selects or pastes to anchor a Depth Dive request. One of five passage types — **text, image, diagram, table, or code snippet** — modeled as a discriminated union keyed on `passage_type`. The content is always carried in the request; the store is never the content source. Corpus anchoring is **optional**: `text` and `code` may anchor via `chunk_id` (ADR-0014); `image`, `diagram`, and `table` anchor via `document_id` + document-relative ordinal. An unanchored passage may carry an optional `source` URL as provenance only. The harness similarity-checks unanchored passages against the ingested corpus before producing output. Distinct from "injected context" — a captured passage is user-selected and explicit, not retrieved; distinct from a "chunk" — a chunk is system-produced and grounds Retrieval QA, a captured passage grounds Depth Dive.

## Cross-Reference (implicit)
A property of Retrieval QA's global-corpus retrieval, not a separate feature: because all ingested documents share one vector database, a query naturally surfaces injected context from any related document, regardless of which document the user is "currently" reading. In MVP by default — no extra machinery required beyond global retrieval.

## Concept Linking (post-MVP)
The explicit, query-independent capability of proactively surfacing relationships between documents (e.g. "this passage relates to a paper you uploaded last week") without the user asking. Requires its own architecture (entity/concept extraction, a relation layer over the vector store, a UI surface) and its own evaluation criteria. Deferred until Retrieval QA and B are proven.

## Retrieval Practice / Spaced Repetition (post-MVP)
Testing-effect and scheduling-based learning mechanisms (quizzes on captured passages, scheduled resurfacing of concepts) considered part of the "neuroscience-backed" goal but explicitly out of MVP. Requires durable per-concept state (long-term memory), unlike dual coding which is stateless per-request.

## Parent Chunk
The enclosing structural unit (section, chapter, API page) produced by a Document-Type Chunker. Not embedded directly. Contains one or more Child Chunks. At retrieval time, the parent replaces matched child chunks before being handed to the LLM for generation.

## Child Chunk
A fixed-size (~256 tokens, 10% overlap) sliding-window split of a Parent Chunk, the chunk-size/overlap combination chosen by the ADR-0017 tuning harness. Embedded and indexed for retrieval (both dense pgvector and sparse tsvector). A child points to its parent via `parent_chunk_id`. Only children are matched by the query; the parent is what reaches generation.

## Hybrid Search
A retrieval strategy combining dense (embedding-based pgvector cosine search) and sparse (PostgreSQL `tsvector` full-text search) passes, fused via Reciprocal Rank Fusion. Recovers exact-match queries (function names, API endpoints, error codes, symbols) that pure dense retrieval misses.

## Reranking
A second-pass relevance filter that scores the top-K candidates from hybrid search using a cross-encoder (Cohere Rerank for prototyping, planned swap to local BGE-reranker). Keeps the top-5 for generation. Implemented after hybrid search per the RAG reference guide's priority order.

## Query Decomposition (post-MVP)
The technique of splitting a complex multi-hop question into simpler sub-queries, retrieving for each, then synthesizing the results. Deferred until parent-child chunking, hybrid search, and reranking are implemented and evaluated.

## Evaluation

**Content-Signature Labeling**:
A ground-truth labeling strategy for retrieval evaluation where expected passages are identified by a distinctive substring of their text (or a SHA-256 of that substring). Invariant to chunk boundaries — unlike position ranges or page numbers — so the same labeled queries work across any chunk-size configuration without re-labeling.

**Eval Query Taxonomy**:
Four-stratum classification of eval queries:
- *Concept lookup* (dense-friendly) — single-fact lookup ("What is attention?")
- *Exact-match / keyword* (sparse-friendly) — API names, error codes, CLI flags
- *Context-dependent* (parent-child matters) — requires the enclosing section to answer precisely
- *Multi-hop / reasoning* (decomposition prep) — relates two or more concepts across different parts of the corpus

## Mock Upstream
The stand-in that plays the role of the hosted embeddings, inference, and web-search APIs during volume load runs, so a load test exercises the deployed service without spending real API budget or hitting rate limits. Used only for volume runs; budgeted smoke runs hit the real APIs unchanged.

## Edge Gate
The self-hosted reverse proxy placed in front of the deployed service when it becomes reachable outside the local Docker network — the scaffolding for ADR-0018's graduation trigger. Terminates TLS and enforces basic-auth on every route except the health probe, so the api remains auth-less (per ADR-0018) while the public demo URL stays gated. Exists as scaffold only until the public/VPS phase executes it.

## Load Generator
The Locust-based traffic source that drives load runs against the deployed service, replaying eval-corpus queries against `/query` and `/dive` (with `/health` as a low-rate liveness task). Two run profiles: **volume** (mock upstream, sustained) and **smoke** (real API, budgeted). Success for a run is judged by error-rate and p95-latency ceilings that the alert rules and runbook key on.

## Alert Rule
A threshold-based monitoring condition that fires when a service symptom crosses a defined limit — service down, upstream 502/503 spike, high p95 latency, internal 5xx spike, or ingestion failed-state. Each rule carries a severity and maps to one runbook entry.

## Runbook
The operator-facing document that maps each alert to its incident response: the symptom that fires it, the diagnosis steps to confirm and localise the failure, and the recovery actions that restore service. Tied 1:1 to the alert rule set, so every alert has exactly one runbook entry.

## Backup Posture
The standing policy deciding which deployed data is preserved: only the ingested corpus (the app's pgvector database) is backed up, via `pg_dump` of that single database; every observability store (Langfuse, ClickHouse, MinIO, Prometheus, Grafana) is treated as ephemeral and regenerable, never backed up. Locks the what/where of backups now; the scheduling and off-box copy belong to the VPS/hybrid phase.