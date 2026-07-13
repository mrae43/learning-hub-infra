# 0003 — Hand-Roll the RAG Pipeline; No Orchestration Framework

## Status
Accepted

## Context
ADR-0001 and ADR-0002 both resolved in favor of "needs-driven, fastest path to validating Harness A" — hosted inference over self-hosted vLLM, pgvector over Qdrant. The same question arose for orchestration: use LangChain/LlamaIndex to move fast, or hand-roll chunk → embed → store → retrieve → prompt directly against raw client libraries.

This decision breaks the pattern of the first two. Unlike inference serving or vector-store operations (which are infrastructure *around* retrieval), orchestration framework choice governs the retrieval mechanics themselves — chunking strategy, retrieval logic, prompt assembly. Using a framework here doesn't just add ops surface, it hides the exact thing this project exists to build understanding of.

## Decision
The RAG pipeline is hand-rolled — no LangChain, no LlamaIndex. Chunking, embedding calls, retrieval queries, and prompt assembly are built directly against raw client libraries (Postgres/pgvector client, embedding model client, LLM API client).

## Consequences
- Slower to build MVP than reaching for a framework — every stage is explicit code, not a library call.
- No abstraction hides retrieval mechanics; debugging a bad answer means inspecting real chunks, real similarity scores, and the real assembled prompt, not a framework's internal representation.
- This is an intentional exception to the "needs-driven, fastest path" pattern established in ADR-0001 and ADR-0002 — not an inconsistency. Those decisions deferred *infrastructure* the retrieval logic depends on; this decision keeps the *retrieval logic itself* transparent, since that's the specific thing under test and the specific skill being built.
- If retrieval logic proves to need capabilities a framework would have given "for free" (e.g. advanced re-ranking, agentic retrieval), revisit this — but only after hand-rolled retrieval has been tested against real use, per Step 14 of the project blueprint.