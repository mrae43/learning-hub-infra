# Depth Dive Redefinition Spec

> Contract-altitude spec for the redefined Depth Dive harness. No code, no migration design — this is the destination artifact handed to implementation planning.
>
> Assembled from decisions in the [Wayfinder Map: Depth Dive Redefinition](https://github.com/mrae43/learning-hub-infra/issues/215) child tickets.

## 1. Destination

Depth Dive is the synthesis harness that takes a single **Captured Passage** selected by a learner and produces a self-contained, interactive explanation. The MVP output is an **`interactive_animation`** scene graph driven by dual-coding principles. The harness is stateless, may call agentic web search, and never maintains per-learner state in this repo.

This spec supersedes the dual-coding MVP definition in `CONTEXT.md` and provides the contracts — domain model, output taxonomy, request/response shapes, agent architecture, and storage assumptions — that the implementation map will build against.

## 2. Scope boundary

### In scope

- The `CapturedPassage` domain model (five passage types, anchors, carried content).
- The passage-to-model transform (carrier selection, validation, size bounds).
- The output-type taxonomy and routing rules.
- The `interactive_animation` payload contract (declarative scene graph).
- The Harness B agent architecture (two-agent pipeline, statelessness boundary).
- The `HarnessBRequest` / `HarnessBResponse` API contract (`POST /dive`).
- The storage-contract assumptions handed to the future ingestion-redesign map.
- The module-boundary consequences for `core` vs. `retrieval_qa` vs. `depth_dive`.

### Out of scope

- Ingestion pipeline redesign for non-text extraction — its own wayfinder map, consuming §10 of this spec.
- Client/UI implementation, repo topology, and start timing — the UI lives in a separate repo; contracts here are sufficient to mock against.
- Backend deployment, auth, multi-tenancy — MVP uses static API keys per `ADR-0018`; per-user auth is a future map.
- Stateful learner modeling (quizzing, scheduling, spaced repetition, progress history) — post-MVP per `CONTEXT.md`.
- Retrieval Practice / Spaced Repetition as output types — post-MVP.

## 3. Domain terms

These entries refine or replace the `CONTEXT.md` definitions.

### Depth Dive

The synthesis harness that consumes the shared retrieval layer (`core/retrieval/`, per `ADR-0019`) and produces a richer, non-plain-text explanation of a single Captured Passage. The MVP output is one **`interactive_animation`** scene graph. Depth Dive is permitted to extend beyond the ingested corpus via agentic web search (`ADR-0012`, `ADR-0013`). It is stateless: no per-learner state, no quizzing, no scheduling. Term retained from the original definition.

### Captured Passage

The specific content a user selects or pastes to anchor a Depth Dive request. One of five passage types — **text, image, diagram, table, or code snippet** — modeled as a discriminated union keyed on `passage_type`. The content is always carried in the request; the store is never the content source. Corpus anchoring is optional:

- `text` and `code` may anchor via `chunk_id` (`ADR-0014`).
- `image`, `diagram`, and `table` anchor via `document_id` + document-relative ordinal/figure number.

An unanchored passage carries an optional `source` URL (provenance only, not authoritative). The harness similarity-checks unanchored passages against the ingested corpus before producing output.

### Interactive Animation

The flagship Depth Dive output type: a declarative, stateless scene graph that a client renders without further server calls. It pairs explanatory text with a visual (dual coding) and is composed of persistent `elements` and ordered `steps` that mutate `element_states`.

### Treatment

A pedagogical pattern layered onto an `interactive_animation`. The three MVP treatments are `worked_example`, `prediction_reveal`, and `segmented_carousel`. Three additional treatments (`analogy_mapping`, `elaborative_prompt`, `interactive_concept_map`) are deferred and eval-gated.

## 4. Captured Passage model

`CapturedPassage` is a single Pydantic discriminated union in `core/types/captured_passage.py`.

### Common fields

| Field | Type | Description |
|---|---|---|
| `passage_type` | `Literal["text", "image", "diagram", "table", "code"]` | Discriminator. Explicit, not inferred. |
| `source` | `str \| None` | Optional provenance URL for unanchored captures. |

### Per-variant fields

**`TextPassage`**

| Field | Type | Description |
|---|---|---|
| `content` | `str` | The captured text excerpt. |
| `chunk_id` | `UUID \| None` | Optional corpus anchor (`ADR-0014`). |

**`CodePassage`**

| Field | Type | Description |
|---|---|---|
| `content` | `str` | The captured code snippet. |
| `language` | `str \| None` | Hint only; unvalidated free text. |
| `chunk_id` | `UUID \| None` | Optional corpus anchor (`ADR-0014`). |

**`ImagePassage`**

| Field | Type | Description |
|---|---|---|
| `content` | `bytes` | Raw image bytes. |
| `media_type` | `Literal["image/png", "image/jpeg", "image/webp", "image/gif"]` | Non-animated GIF only. |
| `caption` | `str \| None` | Supplementary; not a replacement. |
| `document_id` | `UUID \| None` | Optional parallel anchor. |
| `ordinal` | `str \| None` | Document-relative figure/position (e.g., "Figure 3"). |

**`DiagramPassage`**

Same carrier as `ImagePassage`, but semantically distinct: diagrams may feed an internal graph IR for harness routing/assembly. The IR is never sent to the model as a graph.

| Field | Type | Description |
|---|---|---|
| `content` | `bytes` | Raw image bytes. |
| `media_type` | `Literal["image/png", "image/jpeg", "image/webp", "image/gif"]` | Non-animated GIF only. |
| `caption` | `str \| None` | Supplementary; not a replacement. |
| `document_id` | `UUID \| None` | Optional parallel anchor. |
| `ordinal` | `str \| None` | Document-relative figure/position. |

**`TablePassage`**

| Field | Type | Description |
|---|---|---|
| `rows` | `list[list[str]]` or `list[dict[str, str]]` | Structured canonical form. |
| `headers` | `list[str] \| None` | Optional header metadata. |
| `caption` | `str \| None` | Supplementary; not a replacement. |
| `document_id` | `UUID \| None` | Optional parallel anchor. |
| `ordinal` | `str \| None` | Document-relative table ordinal. |

### Identity

A Captured Passage is identified by its **content + anchor**. It is ephemeral: no server-assigned id, never persisted. The optional `client_passage_id` in `HarnessBRequest` is UI-only and ignored by the backend.

## 5. Passage transform

A pre-processing **Passage Transform** stage runs before the framing agent. It validates the carried content against the declared `passage_type`, normalizes, chooses model-ready carriers, and emits flat content blocks.

### Transform rules

| Type | Input | Output carrier | Notes |
|---|---|---|---|
| `text` | `content: str` | Text block | Plain text. |
| `code` | `content: str`, `language` | Text block with language hint | No tokenization or special transform. |
| `image` | `content: bytes`, `media_type` | Base64 image block | Direct to Claude/OpenAI image block. |
| `diagram` | `content: bytes`, `media_type` | Base64 image block | Optional internal graph IR for harness use only. |
| `table` | `rows`, `headers` | Structured text + rendered image | If structured form exceeds bounds, image-only fallback. |

### Validation & size bounds (MVP)

| Type | Bounds |
|---|---|
| `image` / `diagram` | ≤ 8192 × 8192 px, ≤ 5 MB decoded, PNG/JPEG/WebP/non-animated GIF. |
| `table` | ≤ 200 rows × 50 columns in structured form. |
| `code` | ≤ 4000 tokens. |
| `text` | Bounded by model token budget. |

Requests exceeding bounds are rejected (422), not silently cropped.

### What the transform does *not* do

- No server-side tokenization (image tokens are provider-side only).
- No graph structure sent to the model.
- No request-side embedding vectors.
- No public URL fetching for MVP.

## 6. Output-type taxonomy & routing

### Taxonomy

- **One primary output type:** `interactive_animation`.
- **Design principle:** dual-coding explanation — every animation pairs explanatory text with a visual.
- **MVP treatments:**
  - `worked_example` — narration-driven walkthrough with hidden next steps.
  - `prediction_reveal` — predict-the-next-step / predict-the-output reveals.
  - `segmented_carousel` — concept split into swipeable slides.
- **Deferred / eval-gated treatments:**
  - `analogy_mapping`
  - `elaborative_prompt`
  - `interactive_concept_map`

### Mapping from the old CONTEXT.md trio

- **Diagram** → visual layer of the animation (not a standalone type).
- **Carousel** → `segmented_carousel` treatment.
- **Coding example** → `worked_example` treatment when `passage_type=code`.

### Routing

The harness recommends treatments; the user's explicit request wins.

Precedence:

1. `requested_output_type` / `requested_treatments` (explicit ask) wins.
2. `preferred_treatments` (UI-supplied, request-time only) wins over harness recommendation.
3. Harness recommendation wins when neither user input is present.
4. Incompatible / unsupported requests fall back through the chain with a `routing_note`.

`requested_output_type` is kept for forward compatibility; MVP only supports `interactive_animation`.

## 7. Interactive-animation output contract

The `interactive_animation` payload is a declarative JSON scene graph. It is fully self-contained: the client renders it without further server calls.

### Top-level shape

The ratified top-level contract (finalized during implementation, ticket #257):

```json
{
  "output_type": "interactive_animation",
  "title": "Self-attention: how a word sees its neighbors",
  "concept": "attention",
  "viewport": {"width": 800, "height": 520},
  "elements": [...],
  "steps": [...],
  "initial_state": {...},
  "interactions": {"click_to_advance": true}
}
```

| Field | Type | Description |
|---|---|---|
| `output_type` | `Literal["interactive_animation"]` | Discriminated union root; the MVP's only output type. |
| `title` | `str` | Short headline for the animation. |
| `concept` | `str` | The concept being animated. |
| `viewport` | `Viewport` (`width`, `height`) | Design-time coordinate space that element `x`/`y` positions are relative to. Screen-size independence is the client's job; these are nominal dimensions (prototype: 800 × 520). |
| `elements` | `list[SceneElement]` | Persistent scene primitives with stable IDs (non-empty). |
| `steps` | `list[AnimationStep]` | Ordered states that mutate `element_states` by element ID (non-empty). |
| `initial_state` | `dict[str, ElementState]` | Seeds the initial state of every declared element. |
| `interactions` | `InteractionHints` | Interaction affordances the client may expose (`click_to_advance`, `reveal_on_last_step`, `segments`). |

`title`, `concept`, `viewport`, and `interactions` are ratified parts of the contract beyond the bare `{output_type, elements, steps, initial_state}` skeleton: `viewport` gives `x`/`y` positions a reference space (the spec's "relative coordinates" constraint below), and `interactions` carries the per-treatment affordances that §6 treatments rely on (`click_to_advance` for `worked_example`/`prediction_reveal`, `segments` for `segmented_carousel`, `reveal_on_last_step` for `prediction_reveal`).

### `elements`

Persistent scene primitives with stable IDs. The finalized primitive set (originally a candidate set to be settled during implementation):

- `text` — labels, narration, math.
- `token` — individual tokens / vectors.
- `vector` — numeric vectors (e.g., embeddings, attention scores).
- `score` — scalar attention/similarity scores.
- `arrow` — connections between elements.
- `group` — layout containers.

Each element carries an `id`, a `type`, and an `x`/`y` position; type-specific payload lives in optional fields (`text`, `label`, `color`, `value`) and a `highlight` flag, with an optional `style` (`fontSize`, `fontWeight`, `textAnchor`, `fill`).

### `steps`

Ordered states that mutate `element_states` by ID. Each step carries:

- `id: str` — stable step identifier.
- `label: str` — narration/caption for the step.
- `element_states: dict[str, ElementState]` — per-element mutations (`opacity`, `highlight`, `value`, `text`).
- `duration_ms: int | None` — transition duration hint (the sketch's "easing, duration" transition hints; duration was finalized, easing was dropped).
- `segment: int | None` — zero-based segment index when the animation is a `segmented_carousel`.

### Design constraints

- Stateless: the entire artifact ships in one response.
- Screen-size independent: relative coordinates or responsive layout.
- Concept-centric: primitives are learning concepts (tokens, vectors, scores), not arbitrary shapes.
- Text/math-first: must render technical notation well.

A worked example (self-attention for `[river, bank, money]`) exists on the `prototype/interactive-animation-contract` branch for reference. That prototype is illustrative only: it carries demo-only fields not present in the ratified contract — a top-level `version` (omitted) and a top-level `treatment` (superseded by the `HarnessBResponse.applied_treatments` list). The shape above is authoritative.

## 8. Harness B agent architecture

Depth Dive runs a **two-agent stateless pipeline** in this repo.

### Agent topology

1. **Framing agent**
   - Consumes `CapturedPassage` + optional request hints.
   - Decides whether external grounding (web search) would help (per `ADR-0012`).
   - Picks output type and treatments (applying routing rules from §6).
   - Emits a creative brief: concept to animate, key takeaways, visual metaphors, required corpus context, and `search_intent: str | None`.

2. **Assembly agent**
   - Consumes the brief.
   - Calls corpus retrieval and/or web search as needed.
   - Prompts the LLM for the artifact payload.
   - Returns the structured `HarnessBResponse`.

The framing/assembly split makes the output decision visible and auditable before generation starts.

### Web-search ownership

- `ADR-0012` survives: the framing agent decides per-request whether external material would strengthen the dive.
- `ADR-0013` survives: the assembly agent owns the web-search tool wrapper, retries once on failure, and falls back with a note if the retry also fails.
- No fixed schedule by output type or query keyword.

### Retrieval a dive performs

| Passage type | Anchored? | Retrieval performed |
|---|---|---|
| `text` / `code` | yes (`chunk_id`) | Fetch parent chunk + up to K semantic neighbors from the global corpus |
| `image` / `diagram` / `table` | yes (`document_id` + ordinal) | Fetch document-relative text context near the figure/table + semantic neighbors on the passage's embeddable representation |
| any type | no (unanchored) | Run corpus-similarity gate; if grounded, optionally include top neighbors; if unverified, still proceed but flag |

The parent-child model for `text`/`code` is reused from Harness A / `ADR-0014`. Non-text passages use the parallel anchor from §4.

### Statelessness boundary

**Forbidden inside this repo:** per-learner state of any kind — learner profile, progress history, spaced repetition, scheduling, per-passage notes/ratings, per-user preferences, per-user document libraries.

**Allowed per request:** retrieval, web search, LLM calls, building the interactive artifact. Deterministic memoization keyed strictly by request content + corpus version + search results is allowed as a performance optimization, but must not accumulate across requests in a way that personalizes future output.

**Owner of learner state:** a separate UI repo. Cross-device sync is v2 for that repo.

### UI-repo / backend boundary

- Document ownership stays in this backend (global ingested corpus).
- The UI repo may assign a `client_passage_id` at capture time; the backend ignores it.
- Auth stays at static API keys per `ADR-0018`.

## 9. `HarnessBRequest` / `HarnessBResponse` contract

New standalone Pydantic models in `core/types/depth_dive.py`. `CapturedPassage` lives in `core/types/captured_passage.py`.

### Endpoint

- `POST /dive`
- 200 OK + `HarnessBResponse` on success
- 422 validation; 502/503 upstream failures; FastAPI default `{"detail": ...}` error body

### `HarnessBRequest`

| Field | Type | Description |
|---|---|---|
| `captured_passage` | `CapturedPassage` | One captured passage per request. |
| `requested_output_type` | `str \| None` | Forward-compatible; unsupported values fall back to `interactive_animation` with a `routing_note` (never a 422). |
| `requested_treatments` | `list[Treatment] \| None` | Explicit ask. |
| `preferred_treatments` | `list[Treatment] \| None` | UI-supplied, request-time only. |
| `client_passage_id` | `str \| None` | UI-only; backend ignores. |

### `HarnessBResponse`

| Field | Type | Description |
|---|---|---|
| `output` | `InteractiveAnimation` | Discriminated union root; MVP only `interactive_animation`. |
| `recommended_treatments` | `list[Treatment]` | Harness recommendation. |
| `applied_treatments` | `list[Treatment]` | Final resolved set. |
| `routing_note` | `str \| None` | Explains overrides. |
| `grounded` | `bool` | Same semantics as Harness A; false for unanchored passages that fail the corpus-similarity gate. |
| `external_search_attempted` | `bool` | Whether web search was attempted. |
| `external_search_failed` | `bool` | Whether the retry also failed. |
| `external_search_note` | `str \| None` | User-facing note when search failed. |
| `cited_passages` | `list[CitedPassage]` | Reuses `core.types.responses.CitedPassage`; empty when not grounded. |

The framing brief stays internal and is not exposed in the response.

## 10. Storage-contract assumptions

This spec does not design tables, columns, or migrations. It records the contract the future ingestion-redesign map must satisfy so that Depth Dive can retrieve non-text passages.

| Passage type | Stored retrievable form | Identity | Required metadata | Relationship to documents/chunks | Corpus-side representation |
|---|---|---|---|---|---|
| **Image** | Retrievable reference to image asset | `document_id` + figure/ordinal | Format, width, height, optional caption/alt text, optional page reference | Child of `document_id`; optional nearest-chunk position | Multimodal embedding *and* text-serializable caption/summary |
| **Diagram** | Retrievable reference to image asset | `document_id` + figure/ordinal | Format, width, height, optional caption/alt text, optional page reference | Child of `document_id`; optional nearest-chunk position | Multimodal embedding *and* text-serializable caption/summary |
| **Table** | Structured rows/cols as canonical; optional rendered image | `document_id` + table ordinal | Row count, col count, headers (if extractable), optional caption, optional page reference | Child of `document_id`; optional nearest-chunk position | Embedding of normalized textual serialization *and* the serializable form |
| **Code** | Existing `chunks.content` text | `chunk_id` (`ADR-0014`) | Language hint (best-effort) | Existing chunk/document FK | Existing text-chunk embedding |

Cross-cutting rules:

- `CapturedPassage` content is carried in the request; the store is never the content source.
- Anchors are optional on every type; unanchored passages carry an optional `source` URL.
- Stored representations must be searchable for the corpus-similarity gate and usable as web-search query material and format-output prominence context.
- Image/diagram/table passages are children of a `document_id`, parallel to chunks, not nested inside chunks.
- Code requires no new storage artifact beyond today's text chunks.

## 11. Module boundaries

Depth Dive consumes retrieval through `core/retrieval/`, not through `retrieval_qa`. `ADR-0011` is satisfied, not amended.

- `core/retrieval/` public surface: full pipeline, dense-neighbor search, parent-chunk fetch.
- `depth_dive/` owns the similarity-gate policy and the two-agent harness.
- `retrieval_qa/` narrows to chunking and retrieval evaluation.
- The mechanical move of `retrieval_qa/retrieval/` to `core/retrieval/` is a precondition refactor before any `depth_dive` implementation lands.

## 12. Preconditions & next steps

Before implementation planning begins:

1. Land the `retrieval_qa/retrieval/` → `core/retrieval/` refactor (`ADR-0019`).
2. Update `CONTEXT.md` with the redefined Depth Dive and Captured Passage entries (this spec §3).
3. Resolve the deferred eval approach for generative interactive artifacts (fog item on the map).
4. Open the ingestion-redesign wayfinder map, using §10 as its storage-contract requirement.

This spec is the boundary. Implementation code belongs to the next map.
