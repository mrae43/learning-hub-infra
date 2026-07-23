# 0016 — Retrieval QA Critical Gaps: Parent-Child Chunking, Hybrid Search, and Cross-Encoder Reranking

## Status
Accepted

## Context
Retrieval QA's existing pipeline (per-type structure-aware chunkers, pure dense pgvector retrieval, and direct top-K generation) has been proven in its tracer-bullet form. Before Depth Dive can be built on top of it, three systemic gaps from the RAG reference guide were identified during a structured audit against current 2025–2026 consensus practices (see `RAG-reference-guide.md`):

1. **No parent-child chunk linkage.** Chunks are the atomic unit for both retrieval and generation. Retrieving a small, precise chunk but handing its enclosing section to the LLM — the most widely adopted production pattern — is impossible.
2. **No hybrid search.** The system is dense-only (pgvector cosine distance). Exact-match queries (function names, API endpoints, error codes, LaTeX symbols) that the embedding model glosses over are silently lost.
3. **No cross-encoder reranking.** The system retrieves top-5 raw cosine distance and passes them directly to generation. There is no second-pass relevance filter, which the reference identifies as the single highest-leverage accuracy addition after hybrid search.

All three are independent of each other in design but sequential in implementation: parent-child changes the data model and ingestion pipeline (foundation), hybrid search and reranking sit on top of retrieval.

## Decision

### 1. Parent-child hierarchical chunking
Each existing structure-aware chunk (section, chapter, API page) becomes a **parent**. Children are produced by a recursive fixed-size text splitter at **512 tokens with 15% contextual overlap**, applied uniformly across all document types. Only child chunks are embedded (pgvector) and indexed (tsvector). At retrieval time, child matches are swapped to their parent via `parent_chunk_id` before generation.

Schema: self-referential `parent_chunk_id` nullable foreign key on the existing `chunks` table. Parent rows have `parent_chunk_id = NULL`.

Rationale for fixed-size (not semantic) splitting: the reference notes that some benchmarks find fixed-size splitting beats semantic chunking once cost is factored in. It also avoids adding a training-dependent step to the hand-rolled pipeline (ADR-0003).

### 2. Hybrid search (dense + sparse)
Add a PostgreSQL `tsvector` column and GIN index on child chunk content. At query time, run both a pgvector cosine search (unchanged) and a `ts_rank` full-text search against the same child chunk rows. Fuse results via Reciprocal Rank Fusion (RRF). Retrieve top-20 from each path, fuse into a single ranked set, then swap to parents.

Rationale for tsvector over pg_bm25/pg_textsearch: zero new infrastructure. The existing `pgvector/pgvector:pg16` Docker image works unchanged. No custom Docker image, no C extension build, no CI pipeline changes. Postgres FTS (`ts_rank`) is sufficient for MVP-level exact-match recovery; BM25-level scoring is a quantitative improvement that can be swapped in later if the eval set shows it matters.

### 3. Cross-encoder reranking
After RRF fusion and parent swap, pass the top-20 parent chunks through **Cohere Rerank** (trial API key for prototyping). Keep the top-5 for generation. This is a staged dependency: the Cohere free tier (1,000 calls/month, 10 req/min) is sufficient for prototyping and eval-set iteration. Replacement with a local `BAAI/bge-reranker-v2-m3` (CPU or GPU) is planned as a post-PoC optimization to cut the API dependency and monthly call cap.

### Execution order
The implementation follows dependency order: parent-child schema and ingestion changes first (foundation), then hybrid search, then reranking. Each stage is individually testable and can be evaluated against the existing eval set (ADR-0015) before proceeding to the next.

## Consequences
- The `chunks` table gains a `parent_chunk_id` column; the ingestion pipeline gains a recursive splitting step before embedding.
- New chunk content may be smaller on average (~512 tokens for children vs variable for current structure-aware chunks), which changes the embedding distribution and retrieval characteristics — the eval set will show whether this improves recall.
- Retrieval latency increases: hybrid search adds a tsquery + RRF step; reranking adds an external API call. For a personal learning tool this is acceptable, but it should be measured.
- The Cohere Rerank API dependency is temporary-friendly (free trial, no credit card) but adds a key rotation and availability dependency that the local BGE alternative would eliminate.
- The eval set (4 queries) is too small to confidently measure improvement from these changes. Expanding it is deferred to a follow-up phase ("important but gatable" items) but should be prioritized early to avoid silent regressions.
- Query decomposition (multi-hop sub-question generation) is explicitly deferred until after these three gaps are implemented and evaluated, per the RAG reference guide's priority order.
