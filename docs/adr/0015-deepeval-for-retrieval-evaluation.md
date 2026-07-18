# 0015 — Retrieval Evaluation Pivots to DeepEval (Supersedes ADR-0007's Scorer)

## Status
Accepted

## Context
ADR-0007 established a hand-labeled eval set (query, expected-passage-id pairs from real reading sessions) scored via a hand-rolled recall@k implementation, gated in CI only on PRs touching retrieval-relevant paths. With retrieval logic now fully implemented (embedding via OpenAI `text-embedding-3-small`, pgvector HNSW similarity search, generation via `gpt-4o-mini`, structured `HarnessAResponse`), the question of whether to keep the scorer hand-rolled or adopt a dedicated evaluation framework became concrete rather than hypothetical.

This does not conflict with ADR-0003 (hand-rolled RAG pipeline, no LangChain/LlamaIndex): that decision was scoped to retrieval *mechanics* — chunking, embedding, retrieval, prompt assembly — specifically because a framework there would hide the thing the project exists to build understanding of. Evaluation is a different layer sitting on top of an already-transparent, already-built pipeline; a framework testing it doesn't hide how retrieval works, any more than pytest or ruff do.

Two frameworks were considered: DeepEval (pytest-native, designed for CI-gating use cases — current comparisons describe it as "DeepEval for gates, Ragas for dashboards") and Ragas (dataset-first, exploratory, no ground-truth-label requirement, stronger for dashboards/iteration than CI blocking). Given ADR-0007 already committed to a CI-blocking, path-gated retrieval check integrated with pytest, DeepEval is the closer fit mechanically.

A cost-model risk was flagged during evaluation: both DeepEval and Ragas offer LLM-as-judge-based metrics (contextual relevancy, faithfulness) in addition to simple ground-truth metrics. ADR-0007 deliberately chose recall@k specifically to avoid LLM-judge cost at this stage; ADR-0008 deliberately deferred LLM-as-judge answer-faithfulness scoring to a later, explicit milestone. Adopting a framework that offers judge-based metrics "for free" risks pulling that deferred decision forward accidentally.

## Decision
Replace the hand-rolled recall@k scorer (ADR-0007) with DeepEval's ground-truth-based retrieval metric, run as a `pytest`-integrated check. The path-gated CI approach from ADR-0007 is unchanged: this eval still runs only on PRs touching chunking/embedding/retrieval-relevant code, not on every PR.

DeepEval's (and Ragas's) LLM-as-judge-based metrics — contextual relevancy, faithfulness scoring — are explicitly **not** adopted as part of this decision. They remain gated behind ADR-0008's graduation trigger (corpus/query volume outgrowing manual self-review, or a faithfulness regression caught too late), not pulled forward just because the tooling now makes them one function call away.

## Consequences
- The hand-labeled eval set itself (query, expected-passage-id pairs) is unchanged — only the scoring mechanism changes, from a custom implementation to DeepEval's built-in metric.
- Retrieval eval code now depends on the `deepeval` package as a test-time dependency — this is the first evaluation-layer framework dependency in the project, distinct from the hand-rolled pipeline dependencies themselves.
- Because DeepEval is pytest-native, this integrates naturally with the existing `ci.yml` pytest job rather than requiring a separate eval runner — likely reduces the CI wiring work anticipated when ADR-0007 was written.
- Ongoing discipline required: when the ADR-0008 trigger eventually fires and LLM-as-judge answer-faithfulness scoring is added, it should be a deliberate, separately-justified decision — not adopted implicitly just because DeepEval (now already a dependency) happens to offer a `FaithfulnessMetric` out of the box.
- ADR-0007 is retained for historical context (why recall@k and path-gating were originally chosen) but is superseded regarding the specific scorer implementation.