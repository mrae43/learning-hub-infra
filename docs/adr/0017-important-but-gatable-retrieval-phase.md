# 0017 — Important but Gatable: Retrieval Evaluation Infrastructure and Chunk-Size Tuning

The three gaps identified in the RAG reference guide's review — chunk-size tuning, dedicated extraction for tables/equations/code blocks, and eval-set expansion — were deferred from Phase 1 (parent-child + hybrid search + reranking, ADR-0016) as "Important but Gatable." This ADR records the design decisions for that phase.

## Decision

### 1. Content-signature ground truth (not position ranges or page numbers)

Ground truth passages are labeled by **content signature** — a distinctive substring or SHA-256 hash of the passage text. At eval time, a retrieved chunk is counted as a hit if it contains that substring. This is invariant to chunk boundaries, so the same 50 labeled queries work across all chunk-size configurations without re-labeling.

Rejected alternatives:
- **Page-number ranges** — only paper chunkers track pages; books and docs don't have consistent page metadata.
- **Byte offsets in source file** — the extraction layer (pypdf, HTML strip, etc.) discards original byte positions; retrofitting them into all three chunkers is meaningful work that delays tuning.
- **Chunk-id ranges** — `chunk.position` is a sequence index within a document, not a source-offset proxy; re-chunking changes every position.

### 2. Tuning harness: single test DB, 3 parallel schemas, stratified 50-query eval

Chunk-size tuning compares 256/10%, 512/15%, and 1024/20% in a single test database with three parallel schema/table variants (`chunks_256_10`, `chunks_512_15`, `chunks_1024_20`). Each variant is seeded from the same 6 representative documents (2 books, 2 papers, 2 doc sets). The eval set has 50 queries stratified across four types:

| Type | Count | Purpose |
|------|-------|---------|
| Concept lookup (dense-friendly) | 12–15 | Single-fact retrieval |
| Exact-match / keyword (sparse-friendly) | 10–12 | API names, error codes, CLI flags |
| Context-dependent (parent-child matters) | 10–12 | Enclosing section required |
| Multi-hop / reasoning (decomposition prep) | 8–10 | Relates two concepts across corpus |

Metrics: recall@10 and MRR for dense alone, sparse alone, and fused (RRF). The winning config is re-ingested to production; the test DB is torn down.

### 3. Dedicated extraction for tables/equations/code blocks (MVP scope)

**Papers and books (PDF/EPUB):** Tables and equations are detected heuristically (lines dominated by numbers, operators, or symbolic content) and emitted as separate child chunks with cleaned text. No structured schema (row/cell/LaTeX). Post-MVP: dedicated extraction with caption metadata and structured table rows.

**Documentation (Markdown, HTML):** Code blocks are kept intact — the splitter never splits mid-code-block. Detection uses Markdown fences (`` ``` ``) and HTML `<pre><code>` tags. PDF docs have no code-block detection; they get flat-text splitting only.

## Consequences

- Building the 50-query eval set requires reading real documents and writing queries — a manual, one-time cost of approximately 2–3 hours.
- Content-signature matching is coarser than position-based overlap (a chunk "containing" the passage may also contain substantial nearby text), but this is acceptable for tuning rank-ordering across configurations.
- The 3-schema test DB requires ~3× storage for the evaluation corpus but avoids maintaining per-configuration code branches or config-driven chunking logic.
- Dedicated extraction for tables/equations is heuristic and will miss or misclassify some content — but structured extraction is deferred, not abandoned.
- Code-block protection applies to two of three documentation source formats; PDF docs remain unprotected, which is an accepted gap for MVP.

## Status

Accepted
