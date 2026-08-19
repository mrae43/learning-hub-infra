# 0020 — Depth Dive Harness B Architecture

## Status

Accepted — partially superseded: the scene-graph generation described here (the assembly agent's LLM turn producing the `interactive_animation` scene graph) is superseded by ADR-0022; the "separate UI repo owns learner state" boundary is amended by ADR-0023. The two-agent pipeline, web-search survival, statelessness boundary, and retrieval rules remain in force.

## Context

Depth Dive was originally defined in `CONTEXT.md` as a dual-coding explanation of a captured passage — text paired with a diagram, carousel, or coding example. That definition was too coarse to build against: it did not pin the output contract, the agent topology, the statelessness boundary, or how web search fits in. A wayfinding effort (see map #215) redefined Depth Dive around an `interactive_animation` scene graph, five passage types, and a stateless two-agent pipeline. This ADR records the architectural decisions that came out of that effort.

## Decision

### Two-agent stateless pipeline

Depth Dive runs two sequential agents in this repo:

1. **Framing agent** — consumes `CapturedPassage` + optional request hints, decides whether external grounding (web search) would help, picks the output type and treatments, and emits a creative brief.
2. **Assembly agent** — consumes the brief, calls corpus retrieval and/or web search as needed, prompts the LLM for the artifact payload, and returns `HarnessBResponse`.

The split makes the output decision visible and auditable before generation starts, and separates "what to make" from "how to ground and build it."

### Web-search survival

`ADR-0012` and `ADR-0013` survive the redefinition:

- The framing agent decides per-request whether external material would strengthen the dive (`ADR-0012`).
- The assembly agent owns the web-search tool wrapper, retries once on failure, and falls back with a note if the retry also fails (`ADR-0013`).
- There is no fixed search schedule by output type or query keyword.

### Statelessness boundary

**Forbidden inside this repo:** any per-learner state — learner profile, progress history, spaced repetition, scheduling, per-passage notes/ratings, per-user preferences, per-user document libraries.

**Allowed per request:** retrieval, web search, LLM calls, building the interactive artifact. Deterministic memoization keyed strictly by request content + corpus version + search results is allowed as a performance optimization, but must not accumulate across requests in a way that personalizes future output.

**Owner of learner state:** a separate UI repo. Cross-device sync is v2 for that repo.

### Retrieval rules

| Passage type | Anchored? | Retrieval performed |
|---|---|---|
| `text` / `code` | yes (`chunk_id`) | Fetch parent chunk + up to K semantic neighbors from the global corpus |
| `image` / `diagram` / `table` | yes (`document_id` + ordinal) | Fetch document-relative text context near the figure/table + semantic neighbors on the passage's embeddable representation |
| any type | no (unanchored) | Run corpus-similarity gate; if grounded, optionally include top neighbors; if unverified, proceed but flag |

The similarity gate's threshold and grounded/unverified judgement are `depth_dive` harness policy built on `core/retrieval/` primitives.

### UI-repo / backend boundary

- Document ownership stays in this backend (global ingested corpus).
- The UI repo may assign a `client_passage_id` at capture time; the backend ignores it.
- Auth stays at static API keys per `ADR-0018`.

## Consequences

- `depth_dive/` will contain framing and assembly agents plus the similarity-gate policy; it will not contain retrieval primitives or per-learner state.
- `core/retrieval/` (per `ADR-0019`) is the only retrieval surface `depth_dive` may use; direct imports from `retrieval_qa` remain forbidden by `ADR-0011`.
- The framing brief is an internal artifact and is not exposed in `HarnessBResponse`.
- Cost/latency budgeting for Depth Dive must account for an unpredictable number of web-search calls and two LLM-agent turns.
