# 0001 — Hosted Inference API for MVP; Self-Hosted vLLM Deferred to Its Own Milestone

## Status
Accepted

## Context
The target tech stack for this project's broader goal (AI/ML infrastructure engineering practice) centers on vLLM as the inference layer. However, the MVP job (Harness A: ingest → chunk → embed → retrieve → plain-text QA) does not strictly require self-hosted inference to be validated — retrieval quality is the thing under test, not serving infrastructure.

Two options were considered:
- Self-hosted vLLM from day one, matching the target stack immediately.
- A hosted inference API (e.g. Claude/OpenAI) for MVP, with vLLM deferred to a dedicated later milestone.

## Decision
MVP infra will be strictly needs-driven — the leanest stack that lets Harness A function — rather than deliberately inclusive of target-stack pieces it doesn't yet need. Inference for MVP will use a hosted API. Self-hosted vLLM is deferred to its own later milestone, once Harness A's retrieval quality is proven.

## Consequences
- MVP ships faster and isolates retrieval quality as the variable under test, without conflating it with inference-serving issues.
- Migrating to vLLM later is a real, non-trivial cut-over: the inference client, cost model, and latency profile all change. This should be planned as an explicit milestone with its own scope, not a drop-in swap.
- The project's infra-learning goal (vLLM depth) is intentionally not exercised in MVP. This is a deliberate sequencing choice, not an oversight — worth remembering if a future reader wonders why the "AI/ML infra" project doesn't touch vLLM at first.