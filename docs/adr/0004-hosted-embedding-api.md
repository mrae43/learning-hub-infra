# 0004 — Hosted Embeddings API (Google / OpenAI); Local sentence-transformers Deferred

## Status
Accepted

## Context
The hand-rolled RAG pipeline (ADR-0003) initially suggested using local embeddings (sentence-transformers with BGE or e5 models) to maintain transparency and cost control. However, hardware constraints shifted the cost-complexity equation.

The project's development environment has a 3050 Ti GPU with 4GB VRAM. While this can technically run a medium-weight embedding model (~400-500MB), it creates two operational friction points for MVP: (1) document ingestion becomes slow as thousands of chunks batch-embed sequentially, confounding the variable under test (retrieval quality); (2) GPU memory is tight enough that debugging retrieval performance becomes entangled with GPU memory pressure and optimization surface area that doesn't exist in the actual problem.

A hosted embeddings API (Google `text-embedding-004` or OpenAI `text-embedding-3-small`) costs ~$0.02 per 1M tokens. An MVP corpus of a few documents (500K–1M tokens total) costs cents, not dollars. It removes the compute constraint variable entirely.

## Decision
MVP uses a hosted embeddings API. Local embeddings (sentence-transformers) are deferred until Harness A's retrieval quality is proven and GPU-side embedding becomes a concrete optimization target, not a constraint on validation.

## Consequences
- Document ingestion is fast and deterministic — no GPU memory tuning required during MVP.
- Retrieval quality can be validated independently of compute hardware and embedding model tuning.
- The pipeline introduces an external dependency (API quota, rate limits, network latency for embedding calls) — acceptable for MVP since the call volume is bounded (one embedding per chunk, once).
- Cost is negligible at MVP scale (cents). Future cost optimization (local embeddings, Hugging Face inference endpoints) is a visible decision when ingestion volume makes it economically meaningful, not a premature optimization.
- The decision is partially reversible: once retrieval is proven, swapping to local embeddings is a matter of replacing one client library with another (though significant enough to warrant its own ADR when the time comes).