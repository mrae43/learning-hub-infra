# Multimodal passage inputs in hosted LLM APIs

> Research note resolving Wayfinder ticket #217. Feeds the
> "Non-text passage → model-consumable transform (detection + tokens/graph)"
> grilling ticket.
>
> Survey question: for each non-text passage type — image, diagram, table,
> code snippet — what input representations do the current Claude and OpenAI
> APIs actually accept (image blocks, base64, URLs, structured text)? What are
> the token/size limits and cost implications? What Python SDK shapes do these
> take today? And what does the charting hypothesis — "a dedicated harness
> detects non-text passages and transforms them into tokens or graph
> structure" — actually mean against these APIs?

## 1. Scope and locked constraints

Per the map (not re-litigated here):

- **Hosted inference only.** ADR-0001 locks MVP inference to hosted
  Claude/OpenAI APIs; self-hosted vLLM is deferred. Whatever the transform
  does, it must be expressible through those two hosted surfaces.
- **Non-text is first-class.** A table is not markdown; an image is not its
  caption (charting lock). Depth Dive must feed captured non-text passages to
  the model as non-text.
- **Passage types.** The Captured Passage model recognises five types:
  **text, image, diagram, table, code snippet**. This survey covers the four
  non-text ones, plus the document form that embeds them.
- **No public URL for user content.** The user's client hands captured
  passages to the harness, which holds the API credentials. Base64-in-the-body
  is the natural transport — no need for a publicly reachable URL.

### Method note

Both halves were traced to **primary sources** — docs.claude.com (Anthropic),
platform.openai.com / developers.openai.com (OpenAI), and PyPI for SDK
versions — read 9 August 2026. Every claim carries its source URL (inline
links and §7). No secondary write-ups were used as evidence. Numbers are
current as of that date; model names, contexts, and prices change quickly
(both providers now ship 1M-context-token flagships and have changed their
image-token schemes since the GPT-4o era).

## 2. The short answer

| Passage type | Claude (Messages API) | OpenAI (Chat Completions / Responses) |
|---|---|---|
| **Text** | `text` block | `input_text` / `text` content part |
| **Image** | `image` block, base64 or URL; media types jpeg/png/gif/webp | `image_url` (Chat) / `input_image` (Responses); https URL or base64 data URL; png/jpeg/webp/non-animated gif |
| **Diagram** | `image` block (same as image — diagrams are images) | `input_image` / `image_url` |
| **Table** | `image` block, or `document`(PDF) block where every page is read as image **and** text | `input_file` (Responses) — PDF (extracted text + page images), or spreadsheet (parsed rows/summaries) — or image/text |
| **Code snippet** | `text` block; no special carrier | `input_text` / `text` content |
| **Graph structure** | **No native input type.** Input is a flat list of typed content blocks; there is no graph/tree/nodes-edges block | **No native input type.** Content parts/items are a flat enumerated union; no hierarchical type |

### Key size / cost numbers

| | Claude | OpenAI (2026 generation) |
|---|---|---|
| Per-image size | 10 MB base64 (5 MB on Bedrock/Vertex); 8000×8000 px max | No per-image byte limit documented; 512 MB total payload / 1500 images per request |
| Image count/request | 100 for 200k-context models, 600 for 1M-context | up to 1500 |
| Image→token | visual tokens of 28×28 px patches | 512×512 px tiles (legacy, GPT-4o family) or 32×32 px patches (2026 generation); flagship `original` keeps native pixels |
| Document/PDF | `document` block: 600 pages/request (100 <1M ctx), each page ≈ image + extracted text | `input_file` (Responses): 50 MB/file, PDF → text + page images |

### 2.1 What this means for the harness's transform

The charting hypothesis was *"a dedicated harness detects non-text passages
and transforms them into tokens or graph structure."* Against the two APIs,
that decomposes into:

1. **"Transform into tokens" — the harness does no tokenization.** Token
   counts are computed **inside** the providers from pixel patches/tiles
   (see §3.1, §4.2); `tokens` are not something the harness produces. The
   harness's job is **representation selection + encoding** — raw bytes into a
   `base64` data blob, or an image to `image`/`input_image` block, or a
   document to `document`/`input_file` block. Nothing more.

2. **"Or graph structure" — graphs are never a model input.** Neither API
   accepts a graph/tree/node-edge structure as a first-class content type
   (verified against both type unions, §3.3, §4.4). A graph IR (e.g. the
   parent→child chunk graph, a diagram-structure graph) is useful
   **inside the harness** — for detection, routing, knowledge assembly — but
   before it reaches the model it must be **serialized to text** (DOT/JSON/
   adjacency list) or **rasterized to an image** (rendered diagram) attached
   as an image/document block. "Transforming to tokens or graph" therefore
   reads as: *the harness builds a structured payload of the passage
   (possibly via an internal graph), and the payload to the model is just
   flat text/image/document blocks.*

3. **Non-text passage → model.** Capture (bytes + type) → choose a
   representation the API accepts (image block | document block | structured
   text) → encode (base64) → send. The transform is representation selection
   plus serialization, not server-side tokenization.

## 3. Claude (Anthropic) — Messages API

Sources: [vision](https://docs.claude.com/en/docs/build-with-claude/vision),
[pdf-support](https://docs.claude.com/en/docs/build-with-claude/pdf-support),
[Messages API reference](https://docs.claude.com/en/api/messages),
[context windows](https://docs.claude.com/en/docs/build-with-claude/context-windows),
[models overview](https://docs.claude.com/en/docs/about-claude/models/overview),
[pricing](https://docs.claude.com/en/docs/about-claude/pricing).
SDK `anthropic==0.121.0` (PyPI, 2026-08-07).

### 3.1 Image input

Three source types for an `image` block: `base64`, `url`, and `file_id`
(beta Files API). On Bedrock/Vertex, base64 only. Field names (from the API
reference):

```json
{ "type": "image",
  "source": { "type": "base64", "media_type": "image/jpeg", "data": "<BASE64>" } }
```

```json
{ "type": "image", "source": { "type": "url", "url": "https://example.com/image.jpg" } }
```

- **Media types:** `image/jpeg`, `image/png`, `image/gif`, `image/webp`.
  "Animations are unsupported, and only the first frame is used."
- **Per-image limits:** 10 MB (base64) on the API; 8000×8000 px max.
  Above 20 images per request a stricter per-image dimension limit applies
  (docs advise resizing so neither dimension exceeds 2000 px).
- **Counts:** 100 images/request for 200k-context models; 600 for 1M-context
  models.
- **Token cost (documented formula):** "Claude views images in patches instead
  of pixels. Each patch is a 28×28-pixel block of the image, referred to as a
  visual token. An image costs `⌈width/28⌉ × ⌈height/28⌉`." Standard
  resolution caps at 1568 visual tokens/image (images larger than that are
  downscaled preserving aspect ratio); models "Claude 4.7 and later" get a
  high-resolution tier capping at 4784 tokens/image. Worked examples: 1000×1000
  → 1296 tokens; 4K → 4784 tokens (high-res). At Haiku 4.5's $1/MTok a
  1000×1000 image costs ~$1.30/1000 images; at Opus 5's $5/MTok the 4K image
  ~$23.9/1000.
- **Tokens are input tokens** — they count against the context window and are
  billed at the text input rate.

### 3.2 Document / PDF input

`document` block (base64 or URL). **How PDFs are processed:** every page is
converted to an image, and the page text is extracted and provided alongside —
Claude analyzes both. That matters for **tables** (a table in a PDF is covered
twice: as extracted text and as a rendered page image) and for **diagrams
embedded in documents**.

```json
{ "type": "document",
  "source": { "type": "base64", "media_type": "application/pdf", "data": "<BASE64>" } }
```

- **Limits:** max 600 pages/request (100 when context < 1M); request stays
  within the 32 MB body cap; standard PDFs (no password/encryption). All
  active models support PDFs.
- **Cost:** "each page typically uses 1,500–3,000 tokens depending on content
  density" — plus image token cost per rendered page. Dense PDFs "can fill the
  context window before reaching the page limit."
- **Not supported:** binary office formats like .xlsx/.docx in a `document`
  block — must be converted to text or PDF first.

### 3.3 Input block types — flat, no graph

`messages[].content` is a string or an array of typed blocks. The full set
(from the API reference): `text`, `image`, `document`, `search_result`,
`thinking`, `redacted_thinking`, `tool_use`, `tool_result`, `server_tool_use`,
`web_search_tool_result`, `web_fetch_tool_result`,
`code_execution_tool_result`, `bash_code_execution_tool_result`,
`text_editor_code_execution_tool_result`, tool search result blocks,
`container_upload`, `mid_conv_system`. **No audio block, no video block, no
graph block, no generic file-blob block** (files only via beta Files API `file_id`,
and only as image/document/container_upload). The only nesting is
`document`.ContentBlockSource (text/image children) and `tool_result.content`.

### 3.4 Models, contexts, prices (9 Aug 2026)

All current models support text + image input and PDFs.

| Model | Context | Input $/MTok | Output $/MTok |
|---|---|---|---|
| Claude Opus 5 | 1M | $5 | $25 |
| Claude Sonnet 5 | 1M | $3 (intro $2 to Aug 31 2026) | $15 (intro $10) |
| Claude Haiku 4.5 | 200k | $1 | $5 |
| Claude Fable 5 | 1M | $10 | $50 |

High-res tier (4784 tokens) is Claude 4.7+; standard (1568) for older models.
Prompt caching: cache read = 0.1× input, 5-min write = 1.25×, 1-h write =
2×. Batch API 50% off. Note: "Claude 4.7 and later models… use a newer
tokenizer… approximately 30% more tokens for the same text."

### 3.5 Python SDK shape

SDK `anthropic` 0.121.0; request params are TypedDicts / Pydantic models.

```python
import anthropic

client = anthropic.Anthropic()
msg = client.messages.create(
    model="claude-opus-5",
    max_tokens=1024,
    messages=[{"role": "user", "content": [
        {
            "type": "image",
            "source": {"type": "base64", "media_type": "image/png", "data": image_data},
        },
        {"type": "text", "text": "Describe this diagram."},
    ]}],
)
```

URL variant: `source: {"type": "url", "url": "..."}`. Document:
`{"type": "document", "source": {"type": "base64", "media_type": "application/pdf", "data": pdf_data}}`.

## 4. OpenAI — Chat Completions and Responses API

Sources: [vision guide](https://platform.openai.com/docs/guides/vision),
[file inputs guide](https://platform.openai.com/docs/guides/pdf-files),
[Chat Completions reference](https://platform.openai.com/docs/api-reference/chat/create),
[Responses reference](https://developers.openai.com/api/reference/resources/responses),
[models](https://developers.openai.com/api/docs/models),
[pricing](https://developers.openai.com/api/docs/pricing).
SDK: repo pins `openai>=2.51.0` (2.51.0 released 2026-07-30).

### 4.1 Image input — both API surfaces

Three image encodings on both surfaces: https URL, base64 data URL
(`data:image/png;base64,...`, `data:image/jpeg;base64,...`), or a Files API
`file_id`. Formats: PNG, JPEG, WEBP, **non-animated** GIF. Total payload up to
512 MB, up to 1500 images per request. A per-image byte limit is no longer
documented (the old 20 MB guideline has been dropped from the docs).

**Chat Completions** — `image_url` content part (`detail`: `auto|low|high`):

```json
{ "type": "image_url",
  "image_url": { "url": "https://example.com/image.jpg", "detail": "auto" } }
```

**Responses API** — `input_image` item (`detail`: `low|high|auto|original`;
`image_url` or `file_id`):

```json
{ "type": "input_image",
  "image_url": "data:image/png;base64,...", "detail": "auto" }
```

**`detail` semantics (documented):**

- `low` — model receives a low-resolution 512×512 version. Fast, low-cost.
- `high` — standard high-fidelity.
- `original` — new in the 2026 generation (gpt-5.4+; Responses API):
  preserves input pixel dimensions, no downscaling; the docs reserve it for
  OCR, small-object detection, and bounding-box tasks. Billed at the
  original (largest) patch count.
- `auto` — default. On gpt-5.6/5.5 is equivalent to `original`; on gpt-5.4
  equivalent to `high`; on mini/nano/o4-mini and tile-based models it behaves
  as `high`.

Note a doc inconsistency worth flagging: the vision guide says `original`
exists in "both APIs", but the Chat Completions reference enum only lists
`auto/low/high` — treat `original` as guaranteed on the Responses API only.

### 4.2 Image token cost — two schemes

OpenAI now has **two documented image-token schemes**:

- **Tile-based (legacy).** Models: GPT-4o/4.1/4o-mini and the o-series
  (except o4-mini). For `high`: scale to fit 2048×2048, then scale so the
  shortest side is 768px, count 512×512 tiles, add a base token count.
  GPT-4o/4.1: base 85, 170/tile (high); `low` = 85. gpt-5 family: base 70,
  140/tile. o-series: base 75, 150/tile.
- **Patch-based (current 2026 generation).** Models 32×32 px patches:
  `⌈w/32⌉ × ⌈h/32⌉`, downscale if over the model's patch budget, then × a
  model multiplier. Patch budget 1536 for gpt-5-mini/nano, gpt-5.4-mini/nano,
  o4-mini, gpt-4.1-mini/nano (2025 snapshots); multiplier 1.62 for
  mini-class, 2.46 for nano-class, 1.72 for o4-mini. gpt-5.4/5.5 `high`
  allows up to 2500 patches or ≤2048 px; `original` up to 10,000 patches or
  6000 px — large images can cost far more than before. Worked examples in the
  guide: 1024×1024 → 1024 patches; 1800×2400 → resized → 1452 patches.

Image tokens are billed at the model's per-input-token rate (text rate).

### 4.3 Files / PDF input

- **API surface:** the **Responses API** fully supports files via the
  `input_file` item (`file_id`, `file_url`, or inline `file_data` +
  `filename`). Chat Completions has a `file` content part but it does not
  support `detail` and treats richer formats as text.
- **How PDFs are processed:** "the API extracts both text and page images and
  sends both to the model". A table inside a PDF is therefore preserved as
  both extracted text and a rendered page image.
- **Spreadsheets** (.xlsx, .csv, .tsv): parsed to the first ~1,000 rows/sheet
  plus generated summaries — not full content. Non-PDF images embedded in
  office docs are **not** extracted ("to preserve chart and diagram fidelity,
  convert the file to PDF first").
- **Limits:** each file < 50 MB, combined < 50 MB per request; multiple files
  per request.
- **Models:** `gpt-4o` and later support file inputs.

```json
{ "type": "input_file",
  "filename": "document.pdf",
  "file_data": "data:application/pdf;base64,...",
  "detail": "high" }
```

### 4.4 Other non-text input types

- **Audio:** Chat Completions has an `input_audio` content part (wav/mp3),
  for audio-capable models (gpt-audio, realtime). The Responses API does not
  document an audio input item.
- **Video:** no documented video input; video is generation-side (Sora).
- **Graph/tree:** none. The Chat content-part union is `text | image_url |
  input_audio | file`; the Responses union is `input_text | input_image |
  input_file`. Flat, enumerated, no hierarchy.

### 4.5 Models, contexts, prices (9 Aug 2026)

All latest models: text + image input, text output.

| Model | Context | Input $/MTok | Output $/MTok |
|---|---|---|---|
| gpt-5.6 (sol/flagship) / gpt-5.5 | 1,050,000 | $5 | $30 |
| gpt-5.6 terra | 1,050,000 | $2 | $12 |
| gpt-5.6 luna | 1,050,000 | $0.20 | $1.20 |
| gpt-5.4 | 1,050,000 | $2.50 | $15 |
| gpt-5.4-mini | — | $0.75 | $4.50 |
| gpt-4.1 | ~1,047,576 | $2 | $8 |
| gpt-4o | 128k | $2.50 | $10 |
| o4-mini | 200k | $1.10 | $4.40 |

Long-context tier (gpt-5.4/5.5/5.6): prompts > 272K input tokens bill at
2× input / 1.5× output for the whole session. Cached input cheaper. Image
tokens count against TPM (tokens per minute); the rate-limit docs add an
IPM (images per minute) metric.

### 4.6 Python SDK shape (openai v2)

```python
from openai import OpenAI

client = OpenAI()
resp = client.responses.create(
    model="gpt-5.6",
    input=[{"role": "user", "content": [
        {"type": "input_text", "text": "What is in this diagram?"},
        {"type": "input_image",
         "image_url": "data:image/png;base64,...", "detail": "high"},
    ]}],
)
print(resp.output_text)
```

Chat Completions equivalent uses `client.chat.completions.create(...)` with an
`image_url` content part. SDK types: `ResponseInputImageParam`,
`ChatCompletionContentPartImageParam`, `ImageURL`.

## 5. Findings at a glance

| Passage type | Consumable as | Recommended carrier | Notes |
|---|---|---|---|
| **text** | text block | `text` / `input_text` | No special handling. |
| **image** | `image` (Claude) / `input_image` (Responses) / `image_url` (Chat) | base64 or URL | PNG/JPEG/WEBP/GIF natively. Image tokens ~patch-count server-side. |
| **diagram** | same as image | base64 or URL | Diagrams are images; no re-serialization needed. Graph *structure* is not consumable — only the rendering is. |
| **table** | `image`, `document` (PDF), or text | base64 PDF/image, or structured text | Table fidelity: PDF preserves layout + extracted text; structured text (CSV/JSON) is cheaper; markdown is rejected by the chart ("a table is not markdown"). |
| **code snippet** | string | `text` | No special block type. |
| **document/passage image** | `document` (Claude) / `input_file` (Responses) | base64 PDF | Page-level: text + page images. Dense PDFs can blow the context window (1.5–3k tokens/page). |

## 6. What the harness should remember

1. **Base64 in the body is enough** for both APIs for user-captured non-text
   passages; no public URL, no upload step, no auth on the client. Inline
   base64 is a first-class source for image/document blocks on both providers.
2. **Diagrams/images go straight in** as image blocks — the model consumes
   them natively; no extraction, no "transformation into tokens" by the
   harness.
3. **Tables have real carrier options** (structured text vs PDF vs image) and
   the choice is a cost/fidelity trade the penning tickets should make
   explicit — relevant for the "convert → tokens-or-graph" grilling ticket.
4. **Watch token economics, not just raw bytes.** Image token formulas differ
   by model generation; prompting Claude: intro pricing changes Aug 31 2026;
   an "original"/"auto" gpt-5.x image costs more tokens than the downscaled
   `high`. A single diagram can cost $5/MTok at Opus 5 vs $1 at Haiku 4.5.
5. **If the harness builds a graph IR, it stays internal** — serialize to
   text or an image for the model input; the APIs have no graph block.

## 7. Cited primary sources

Claude / Anthropic:

- https://docs.claude.com/en/docs/build-with-claude/vision
- https://docs.claude.com/en/docs/build-with-claude/pdf-support
- https://docs.claude.com/en/api/messages
- https://docs.claude.com/en/docs/build-with-claude/context-windows
- https://docs.claude.com/en/docs/about-claude/models/overview
- https://docs.claude.com/en/docs/about-claude/pricing
- https://pypi.org/project/anthropic/

OpenAI:

- https://platform.openai.com/docs/guides/vision
- https://platform.openai.com/docs/guides/pdf-files
- https://platform.openai.com/docs/api-reference/chat/create
- https://developers.openai.com/api/reference/resources/responses
- https://developers.openai.com/api/docs/models
- https://developers.openai.com/api/docs/pricing
- https://developers.openai.com/api/docs/guides/rate-limits

## 8. Could not verify

- Exact per-image dimension limit Claude enforces when >20 images in a
  request (the error message reports it; docs advise ≤2000 px).
- Whether `file`-source image blocks (beta Files API) are formally part of the
  GA Messages schema (the reference union shows base64/url only).
- OpenAI's per-image byte limit (docs state only total payload / count).
- The per-token multiplier for the gpt-5.6/5.5/5.4 base models' images
  (pricing table omits; docs defer to the pricing calculator).
- Whether `detail: "original"` is usable via Chat Completions (guide says both
  APIs; reference enum lists only auto/low/high).