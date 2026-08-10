# 0019 — Shared Retrieval Primitives Live in `core`

## Status

Accepted

## Context

CONTEXT.md defined Depth Dive as "consuming Retrieval QA's retrieval layer," but ADR-0011's import-linter contracts forbid `depth_dive` from importing `retrieval_qa`. The redefined Harness B architecture (two-agent pipeline) needs corpus retrieval — parent-chunk fetch, semantic-neighbor search, and a similarity-search primitive backing the unanchored-passage gate — while Harness A already runs the full hybrid pipeline. Nothing in the retrieval layer is specific to Retrieval QA: it is a single module (`retrieval_qa/retrieval/query.py`: dense pgvector search, sparse tsvector search, RRF fusion, parent-swap, rerank orchestration) whose only in-repo dependencies are already `core` symbols. It was built under `retrieval_qa` first, but it is the corpus retrieval layer for both harnesses.

## Decision

The retrieval layer moves from `retrieval_qa/retrieval/` to a new `core/retrieval/` subpackage:

- The pipeline module moves wholesale and as-is; `retrieval_qa/retrieval/` is deleted. `retrieval_qa` keeps `chunking/`, `_utils.py`, and the path-gated eval harness (CI path gates unchanged; imports updated).
- The public `core` surface is exactly the three primitives the harnesses need: the full pipeline (`retrieve_relevant_chunks`), a dense neighbor search, and a parent-chunk fetch. Sparse search and RRF fusion stay private internals.
- Query embedding stays caller-side via `core.clients.Embedder` (already shared); primitives remain vector-in (`query_vector`, optional `query_text` for sparse), preserving the eval harness's pre-computed-vector path.
- The unanchored-passage similarity gate's threshold and grounded/unverified judgement are harness policy: they live in `depth_dive`, built on the `core` search primitive — not in `core`.
- Document-relative lookup for non-text passages will live in `core/retrieval` when built; implementation is deferred to the future ingestion-redesign map (it needs the non-text storage contract first).
- No import-linter contract change: `retrieval_qa → core` and `depth_dive → core` were already the sanctioned direction. ADR-0011 is satisfied, not amended.
- The migration is a precondition of Depth Dive implementation and lands as a standalone mechanical refactor before any `depth_dive` code — `depth_dive` cannot import `retrieval_qa` even temporarily, so "alongside" would mean duplicating retrieval.

## Consequences

- `core` grows beyond "types/clients/config/db" into retrieval logic — a deliberate broadening of ADR-0005's "only truly shared things" principle: retrieval is now genuinely shared by both harnesses.
- `retrieval_qa`'s identity narrows to chunking + retrieval eval; Harness A query orchestration remains in `api/`.
- The refactor touches `api`'s controller, `scripts/run_chunk_tuning.py`, one `ingestion` integration test, `retrieval_qa` eval + unit tests, and the root `conftest.py` — imports only, no behavior change; the suite stays green throughout.
- CONTEXT.md's Depth Dive entry is corrected in place; its full superseding rewrite remains with the Depth Dive redefinition spec.
