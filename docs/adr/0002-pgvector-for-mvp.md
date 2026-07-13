# 0002 — pgvector for MVP; Qdrant Migration Deferred to Its Own Milestone

## Status
Accepted

## Context
Two candidate vector stores were considered for Harness A's retrieval layer: pgvector (a Postgres extension) and Qdrant (a dedicated vector database). This decision sits at the same fork as ADR-0001: "needs-driven, leanest stack" (validate retrieval quality fast) versus "deliberately inclusive of target infra" (this project's broader goal is AI/ML infra learning, and Qdrant is closer to what's run in production RAG systems at scale).

## Decision
MVP uses pgvector. Retrieval quality is the unknown under test in MVP, not vector-store infrastructure — introducing a second service (Qdrant) before Harness A's retrieval logic is proven would conflate two variables (is a bad answer a retrieval-algorithm problem or an infra problem?). Migration to Qdrant is deferred to its own later, explicitly infra-focused milestone, undertaken once Harness A's retrieval quality has been validated against real reading sessions.

## Consequences
- MVP has one fewer service to operate; retrieval quality can be judged in isolation.
- The Qdrant migration, when it happens, becomes a genuine learning exercise: the specific gaps pgvector has (vector-specific tuning, independent scaling, specialized tooling) will be concretely felt rather than assumed, which is a better teacher than starting with Qdrant untested.
- This is consistent with ADR-0001's needs-driven principle — both defer target-stack infra pieces until the thing they'd serve (proven retrieval) exists.