# 0009 — Structured `HarnessAResponse` Shape

## Status
Accepted

## Context
Harness A (Retrieval QA) needs a structured response shape so a client can distinguish a grounded answer (with cited sources) from a non-grounded "couldn't answer from the corpus" response. A bare `str` answer buries the citation and grounding signals inside free-text parsing, which is fragile, untestable, and forces every client to re-implement the same textual heuristics.

CONTEXT.md defines Retrieval QA's output as plain text only (no artifacts, no dual-coding — that's Depth Dive's job). The "plain text" boundary is about *output modality*, not *response structure*: a structured Pydantic response carrying an `answer: str` field honors the plain-text contract while exposing the citation and grounding signals as typed fields the client can branch on.

CONTEXT.md also defines "Injected Context" as the per-query chunk subset placed into the prompt. The response's citation mechanism is the inverse: the chunks that *were* placed into the prompt are surfaced back to the client as the citations of the answer. This duality (injected on the way in, cited on the way out) is what makes the response shape meaningful, not decorative.

## Decision
`HarnessAResponse` is a Pydantic v2 model with three fields:

```python
class CitedPassage(BaseModel):
    chunk_id: UUID
    text: str  # full chunk content; client decides how much to render

class HarnessAResponse(BaseModel):
    answer: str
    cited_passages: list[CitedPassage]
    grounded: bool
```

- **`answer: str`** — always populated, never nullable. When the model grounds in retrieved context, this is the grounded answer; when retrieval finds nothing relevant, this is the model's generated refusal text (e.g. a natural-language "I don't have enough information in your corpus to answer that"), not a fixed sentinel string.
- **`cited_passages: list[CitedPassage]`** — each entry has `chunk_id` (UUID anchor referencing `chunks.chunk_id`, per ADR-0014) and `text` (the chunk's full content). Empty list when `grounded=False`.
- **`grounded: bool`** — False when retrieval found no relevant context (empty corpus or below relevance threshold); True when the answer is grounded in retrieved context. The not-found case is **a valid response, not an exception** (per `coding-standards.md`): exceptions are reserved for genuine failures (API errors, malformed documents), while "no good answer found" is an expected branch of the response shape.

## Considered Options

- **Bare `str` answer** — rejected; citation and grounding signals buried inside free text parsing.
- **`cited_passage_ids: list[str]`** (the original sketch this ADR supersedes) — rejected in favor of `cited_passages: list[CitedPassage]` because the chunk text is already loaded into retrieval's memory for prompt construction (the chunks *are* the injected context), so exposing `text` costs no extra DB call. A bare list of IDs is harder to extend later without breaking API contract (the `list[str]` shape freezes for any client that ships against it); the nested `CitedPassage` type can grow optional fields backward-compatibly.
- **Enum `not_found_reason`** on the response — rejected; the three internal reasons (empty corpus, threshold miss, retrieval-internal state) map to retrieval-internal concepts that don't belong in the client contract. The binary `grounded: bool` is all the client needs; the *reason* belongs in server logs.
- **Observability fields (`latency_ms`, `model_used`) on the response** — rejected; observability belongs in the runtime/log layer (per ADR-0014). Adding a `metadata` block later is a backward-compatible extension, so deferring costs nothing.

## Consequences
- The response shape is uniform across found and not-found branches — both have `answer` populated, only `cited_passages` and `grounded` vary. Clients can treat the response shape as constant and branch on `grounded` alone.
- The model is always in the loop, even on not-found: the refusal text is model-generated, not a fixed sentinel. This produces more natural refusal language than a repetitive sentinel and localizes naturally if the UI later supports multiple languages.
- `CitedPassage` is a domain term aligned with CONTEXT.md's "Captured Passage" (a user-selected excerpt); a cited passage is the system-side analog — the passage the system selected (retrieved and grounded on) and is now surfacing as the citation. Naming the model after the domain term keeps the API honest.
- This ADR records the *shape and contract* of `HarnessAResponse`. The deeper schema (chunks, embeddings, documents) and their supporting decisions (UUIDv7 IDs, embedding storage, type-aware chunk metadata) are recorded separately in ADR-0014, which this ADR depends on for the chunk-identity and chunk-content guarantees the `CitedPassage` fields rely on.