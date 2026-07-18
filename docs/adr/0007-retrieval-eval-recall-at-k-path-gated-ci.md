# 0007 — Retrieval Evaluation: recall@k, Path-Gated CI (Not Every PR)

## Status
Superseded by ADR-0015 — the recall@k metric and hand-rolled scorer described below were replaced with DeepEval's ground-truth retrieval metric. The path-gated CI approach (running only on retrieval-relevant PRs) remains unchanged; see ADR-0015 for what changed and why.


## Context
Harness A's evaluation splits into two layers: retrieval evaluation and answer evaluation (see `CONTEXT.md`). For the retrieval layer's MVP tier — a hand-labeled eval set of (query, expected-passage) pairs curated from real reading sessions — two things needed deciding: the metric, and where the eval runs in CI.

`tech-stack.md` establishes a simple rule: PR checks run on every push, merge is blocked on any failing job. Applying that rule uniformly to the retrieval eval would mean every PR — including docs-only or CI-config changes — triggers real embeddings-API calls (cost and latency) to re-run recall@k against the hand-labeled set, even when nothing retrieval-relevant changed.

## Decision
- **Metric:** recall@k — of the expected passage(s) for each labeled query, how many appear in the top-k retrieved chunks. Simple, directly answers "did retrieval find the right thing," and easy to reason about by hand at MVP's small eval-set scale.
- **CI gating:** the retrieval eval runs only on PRs that touch retrieval-relevant paths (chunking logic, embedding calls, query/retrieval logic) — not on every PR. This is a deliberate, path-based exception to the otherwise-uniform "every PR is checked" rule in `tech-stack.md`.

## Consequences
- Retrieval regressions are still caught before merge, without paying API cost/latency on every unrelated change (e.g. a docs edit or CI workflow tweak).
- This introduces the first conditional (not universal) CI gate in the project — worth remembering as precedent if similar cost/latency-bearing checks are added later (e.g. an LLM-as-judge answer-faithfulness check, which should likely follow the same path-based pattern rather than running on every PR).
- The hand-labeled eval set is MVP-scale only; it does not scale as the ingested corpus grows past what's practical to hand-label. LLM-as-judge is the staged next tier for retrieval evaluation once that limit is hit (see `tech-stack.md` post-MVP table).