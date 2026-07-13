# 0012 — Depth Dive Web Search: Agentic Trigger, Safe Quotation Pattern

## Status
Accepted

## Context
Depth Dive (CONTEXT.md) is permitted to extend beyond the ingested corpus via web search, unlike Retrieval QA's strictly closed-corpus scope. Two trigger models were considered: a deterministic rule tied to artifact type (web search always fires when rendering a coding-example Depth Dive, never otherwise), or an agentic model where the LLM judges per-request whether external material would strengthen this specific Depth Dive.

The deterministic rule was ruled out once it became clear the need isn't confined to coding examples — a conceptual passage might also benefit from an external quote or reference that strengthens the idea, independent of which dual-coding format (diagram, carousel, coding example) is chosen.

Separately, whenever web search surfaces material for inclusion in a Depth Dive (a quote, a blog post's phrasing, a code pattern), reproducing it at length raises real copyright/attribution concerns for content that might end up in a published or open-sourced portfolio project — this needed an explicit constraint, not an implicit assumption.

## Decision
- **Trigger:** Depth Dive's system prompt instructs the model to decide, per-request, whether external material (a quote, a real-world code pattern, a best-practice reference) would meaningfully strengthen the specific Depth Dive being generated. Web search is called only when the model judges yes — not on a fixed schedule tied to artifact type or query keywords.
- **Quotation safety:** any external material surfaced via web search follows a safer pattern — short quotes only (not full paragraphs reproduced verbatim), always paired with a source citation/URL. Paraphrasing with attribution is preferred over verbatim reproduction wherever the exact wording isn't essential to the point being made.

## Consequences
- Depth Dive needs a real tool-use loop (the model reasons about whether to call web search), not a fixed two-source blend — this is architecturally closer to an agentic system than Retrieval QA's single-hop retrieval.
- Web search calls are unpredictable in frequency (dependent on the model's judgment per request), unlike Retrieval QA's deterministic cost profile — worth accounting for in any future cost/latency budgeting for Depth Dive.
- The safer quotation pattern needs to be enforced in the system prompt and ideally checked (similar in spirit to Retrieval QA's `grounded` field in ADR-0009) — a structured Depth Dive response type that tracks quote length and source attribution would let this be verified deterministically rather than trusted to prompt compliance alone. Left open as an implementation detail for when Depth Dive's response schema is designed.
- This decision should be revisited if the project is ever published/open-sourced with real user-uploaded content, since attribution requirements may need to be stricter than "short quote + link."