# 0006 — FastAPI BackgroundTasks for MVP Ingestion; Dedicated Queue Deferred

## Status
Accepted

## Context
Document ingestion (chunking, per-chunk embedding calls, writing to pgvector) can take real time for a full book, especially against a rate-limited hosted embeddings API (ADR-0004). Running this synchronously inside the request/response cycle risks timeouts and a poor upload experience. Something needs to run ingestion outside the request cycle.

Three options were considered, increasing in infra weight: FastAPI's built-in `BackgroundTasks` (in-process, no extra services); a lightweight queue (`arq` or `RQ` with Redis, adding one service plus a worker process, giving retries and persistence); and Celery with Redis/RabbitMQ (full-weight, production-grade, more configuration than a single-user MVP needs).

This mirrors the same fork resolved in ADR-0001 (hosted inference vs. self-hosted vLLM), ADR-0002 (pgvector vs. Qdrant), and ADR-0004 (hosted vs. local embeddings): defer infra weight until the thing it serves is proven necessary, rather than building for a failure mode that hasn't been observed yet.

## Decision
MVP uses FastAPI `BackgroundTasks` for ingestion. No message broker, no separate worker process, no persistent job queue for MVP.

## Consequences
- No extra infrastructure to run or operate during MVP — ingestion logic lives entirely inside the API service.
- Known limitation, accepted for now: if the server restarts mid-ingestion, the in-flight job is lost with no automatic retry, and there's no built-in way to check "is my document still processing" from outside the request that kicked it off.
- Explicit graduation trigger (so this isn't just deferred indefinitely without a decision point): move to `arq` + Redis as its own milestone the first time an in-flight ingestion job is lost to a restart, or the first time visibility into ingestion status becomes a real need — not preemptively.
- This decision should be revisited alongside the vLLM (ADR-0001) and Qdrant (ADR-0002) migrations, since all three represent the same category of deferred infra debt with the same resolution condition: proven necessity, not anticipated need.