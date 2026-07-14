# 0007 — Retrieval Evaluation via recall@k Against a Hand-Labeled Eval Set

## Status
Accepted

## Context
Harness A (Retrieval QA) is, per ADR-0001 and ADR-0002, an experiment whose variable under test is retrieval quality — not serving infrastructure, not embedding model tuning, not vector-store choice. To measure whether the retrieval layer surfaces the right chunks for a given query, a repeatable evaluation methodology is needed; without one, "is a bad answer a retrieval problem or a generation problem?" becomes unanswerable, and every tuning change becomes guesswork.

Two categories of evaluation were considered. End-to-end "answer quality" scoring (LLM-as-judge, BLEU, ROUGE) conflates retrieval with generation — a fluent wrong answer reads as "good" and a brittle-but-correct answer reads as "bad," exactly inverting the signal we need. Retrieval-only metrics (recall@k, precision@k, nDCG) keep the variable under test isolated: they ask "did the top-k retrieved chunks include the passage a human judge expected?" and stop there, leaving generation quality for a separate measurement.

## Decision
Harness A's retrieval quality is measured by **recall@k** against a **hand-labeled evaluation set**. The eval set is a list of `(query, expected_chunk_id)` pairs, where each `expected_chunk_id` references a chunk in the corpus by its stable ID. recall@k asks: for each query, did the retrieval layer's top-k retrieved chunks include the expected passage?

The eval set is runnable offline against the corpus via direct retrieval queries (against pgvector, per ADR-0002); it does not go through the full Harness A API. The eval is the feedback mechanism for retrieval tuning — a poor score is the signal to turn the `hnsw.ef_search` knob (per ADR-0014's config-driven setting) before reaching for chunking-strategy or embedding-model changes.

Chunks must have stable, externally-referenceable IDs so the eval set can point to them durably across environments, re-embeddings, and corpus re-organization. This is satisfied by UUIDv7 primary keys on `chunks.chunk_id` (decided in ADR-0014).

## Consequences
- The eval set is a small, hand-curated artifact — its quality bounds the measurement's signal. Keeping it hand-labeled (rather than LLM-generated) is the cost of a trustworthy signal; the cost scales with corpus churn, not with retrieval volume.
- The retrieval layer can be evaluated *without* running the inference API, removing one cost and one source of variance (LLM output) from the eval loop.
- Retrieval-relevant tests in the pytest suite (chunking, embedding calls, query logic) are kept identifiable via consistent module naming (per `coding-standards.md`), so a future retrieval-evaluation CI job can gate on path.
- This is a measurement methodology decision, not a retrieval-strategy decision — it deliberately does not constrain *how* retrieval works (top-k value, similarity threshold, reranking), only *how we know whether it's working*. Tuning decisions are downstream of the signal this ADR provides.