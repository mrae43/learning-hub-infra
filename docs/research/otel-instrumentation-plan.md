# OpenTelemetry instrumentation plan for the hand-rolled RAG stack

> Research note resolving Wayfinder ticket **#266 — "Research: OpenTelemetry
> instrumentation plan for the RAG stack"**, part of the
> **"Deployed & Monitored Demo"** map. This is the "Deployed & Monitored Demo" →
> "instrument with OpenTelemetry" standing decision: ONE OTel instrumentation
> path exports stage-level spans to BOTH Langfuse (LLM traces) and
> Prometheus/Grafana (service metrics). That lock is not re-litigated here.
>
> Survey question: which OTel Python instrumentors fit this FastAPI +
> hand-rolled RAG stack on Python 3.12; where the stage-level spans
> (retrieve / rerank / generate / embed / web-search) should be created;
> whether the app exports directly or through a standalone collector, and with
> what protocol; and whether self-hosted Langfuse consumes OTel spans directly
> or requires its own SDK — so one instrumentation can reach both backends.

## 1. Scope and locked constraints

Per the map (not re-litigated here):

- **One instrumentation path, two backends.** Instrument with OpenTelemetry
  so a single code path exports stage-level spans to both **Langfuse**
  (LLM traces) and **Prometheus/Grafana** (service metrics).
- **Langfuse is self-hosted via docker-compose.** (The repo already runs its
  local demo via docker compose.)
- **Hand-rolled RAG.** No LangChain/LlamaIndex (ADR-0003); generation via
  hosted Claude/OpenAI (ADR-0001), embeddings via Google/OpenAI (ADR-0004),
  retrieval via PostgreSQL/pgvector (ADR-0002), rerank via Cohere. Everything
  is reached through the repo's own thin clients, not a framework.
- **Python >= 3.12; mypy --strict + ruff green.** Any instrumentation must be
  typed cleanly and formatted under ruff. It must not break the workspace
  import-linter boundaries (ADR-0011) — note `retrieval_qa`/`depth_dive` may
  depend on `core` but not on each other.
- **Pinned deps that the instrumentors must coexist with:** `api` →
  `fastapi>=0.141.1`, `uvicorn>=0.52.1`; `core` → `pydantic>=2`,
  `sqlalchemy>=2`, `psycopg2-binary>=2.9`, `openai>=2.51.0`, `httpx>=0.27`,
  `cohere>=5.0`. `depth_dive` runs transform → framing → assembly → generation
  with a web-search step.

### Method note

Every claim below was traced to a **primary source** — opentelemetry.io docs,
the `open-telemetry/opentelemetry-python` and `opentelemetry-python-contrib`
repos, pypi.org project pages, `open-telemetry/opentelemetry-collector(-contrib)`
docs, Langfuse docs (langfuse.com/docs) and its GitHub — read on 15 August
2026. No secondary write-ups were used as evidence. Versions and status are
current as of that date; OTel instrumentor versions and Langfuse's OTel support
move quickly.

## 2. The short answer

- **Instrumentors that fit:** `opentelemetry-instrumentation-fastapi` (server
  request spans + metrics), `opentelemetry-instrumentation-httpx` (the
  underlying HTTP calls to OpenAI/Cohere), and
  `opentelemetry-instrumentation-sqlalchemy` (DB queries). All are
  OTel-contrib "migration"-semconv instrumentors with wide version ranges that
  cover this repo's pinned deps and Python 3.12. There is **no official GA
  OpenAI instrumentor**; the PyPI `opentelemetry-instrumentation-openai` is a
  **third-party Traceloop/OpenLLMetry** package (experimental), and for a
  hand-rolled stack the cleaner fit is a **manual span per stage**.
- **The five stages are manual child spans** under the FastAPI request root
  span, created with `tracer.start_as_current_span` at the seams already
  present in the code (see §4): the controller orchestrates, the clients own
  the provider calls.
- **Export via a standalone OTel Collector in docker-compose**, receiving
  OTLP/**HTTP** from the app and fanning out — traces → Langfuse, metrics →
  Prometheus. This is what the OTel docs recommend for anything beyond a
  throwaway, and it is the only way to keep "one instrumentation" while
  reaching two backends.
- **Langfuse consumes OTel spans directly** on its `/api/public/otel` OTLP
  endpoint (HTTP only; **gRPC not supported**). The collector needs **no
  special "Langfuse exporter"** — it points a standard `otlphttp` exporter at
  that endpoint with Basic Auth. The Langfuse SDK v4 exists and is OTel-native,
  but using the collector keeps the "one path" lock cleanly.

## 3. Q1 — OTel + FastAPI integration patterns

### 3.1 Instrumentors, versions, and compatibility

Sources: [instrumentation README (package/status
table)](https://github.com/open-telemetry/opentelemetry-python-contrib/blob/main/instrumentation/README.md),
[FastAPI README](https://github.com/open-telemetry/opentelemetry-python-contrib/blob/main/instrumentation/opentelemetry-instrumentation-fastapi/README.rst),
[FastAPI instrument source](https://github.com/open-telemetry/opentelemetry-python-contrib/blob/main/instrumentation/opentelemetry-instrumentation-fastapi/src/opentelemetry/instrumentation/fastapi/__init__.py),
[httpx README](https://github.com/open-telemetry/opentelemetry-python-contrib/blob/main/instrumentation/opentelemetry-instrumentation-httpx/README.rst),
[sqlalchemy README](https://github.com/open-telemetry/opentelemetry-python-contrib/blob/main/instrumentation/opentelemetry-instrumentation-sqlalchemy/README.rst),
[psycopg2 README](https://github.com/open-telemetry/opentelemetry-python-contrib/blob/main/instrumentation/opentelemetry-instrumentation-psycopg2/README.rst).

| Instrumentor | Version range it supports | Repo's pinned dep | Compatible? | Metrics? | Semconv status |
|---|---|---|---|---|---|
| `opentelemetry-instrumentation-fastapi` | `fastapi ~= 0.92` | `>=0.141.1` | ✅ | Yes | migration |
| `opentelemetry-instrumentation-httpx` | `httpx >= 0.18.0` | `>=0.27` | ✅ | Yes | migration |
| `opentelemetry-instrumentation-sqlalchemy` | `sqlalchemy >= 1.0, < 2.1.0` | `>=2` (2.0.x) | ✅ | Yes | migration |
| `opentelemetry-instrumentation-psycopg2` | `psycopg2-binary >= 2.7.3.1` | `>=2.9` | ✅ | No | migration |

Notes:

- **FastAPI** (`~= 0.92`) is a floor; FastAPI 0.141.1 is well above it. The
  instrumentor is applied in-process via
  `FastAPIInstrumentor.instrument_app(app)`, which wraps the ASGI app and
  creates one **server span per request** (the natural root span) plus
  internal `http send`/`http receive` spans (suppressible via
  `exclude_spans=["receive", "send"]`). It also emits
  `http.server.request.*` metrics (Metrics: **Yes**). It carries an
  idempotency guard (`_is_instrumented_by_opentelemetry`) so it is safe to
  call once at app creation.
- **httpx** is included because the OpenAI and Cohere SDKs are built on
  httpx; it gives per-outbound-call spans under a stage span. The repo's
  `httpx>=0.27` is supported.
- **sqlalchemy vs psycopg2.** Prefer the **SQLAlchemy** instrumentor here:
  the repo already wraps every query through `Session.execute` in
  `core/retrieval/query.py`, so a single SQLAlchemy instrumentor covers the
  retrieval DB work (dense HNSW + sparse tsvector + parent swap) without
  needing the lower-level DB-API instrumentor. Version note: the SQLAlchemy
  instrumentor requires `< 2.1.0`; the repo's `>=2` resolves to 2.0.x, so this
  is fine — but pin it if the lockfile would otherwise pull 2.1+.
- **uvicorn** needs **no dedicated instrumentor.** uvicorn is the ASGI server
  that runs the FastAPI app (`docker-entrypoint.sh` runs
  `uvicorn api.main:app`); the request span is created by the FastAPI/ASGI
  instrumentor, not uvicorn. There is no `opentelemetry-instrumentation-uvicorn`.

### 3.2 OpenAI instrumentation — no official GA package

The package named `opentelemetry-instrumentation-openai` **does exist on PyPI
(0.62.3, 2026-08-10)**, but its stated source repository is
[`github.com/traceloop/openllmetry`](https://pypi.org/project/opentelemetry-instrumentation-openai/)
— it is the **OpenLLMetry (Traceloop) package**, a third-party, experimental
instrumentor, not the OpenTelemetry SIG's. It supports Python `>=3.10` and
auto-instruments the official `openai` SDK, but **by default logs prompts,
completions, and embeddings to span attributes** (disable with
`TRACELOOP_TRACE_CONTENT=false`) — a privacy/volume consideration. The
OpenTelemetry GenAI instrumentors in contrib (e.g.
`instrumentation/genai/`) remain experimental and are not GA.

Implication: because this repo reaches OpenAI (and Cohere) through its **own
thin clients** (`core/clients/llm_client.py`, `embeddings_client.py`,
`reranker_client.py`, `depth_dive/web_search/client.py`), the idiomatic,
stable choice is a **manual span inside each client method** rather than a
third-party auto-instrumentor. That gives exact stage boundaries, keeps
`mypy --strict`/ruff happy, and avoids pulling an experimental dependency.
Optionally add `opentelemetry-instrumentation-httpx` for low-level provider
call detail as children of those spans.

### 3.3 Python 3.12 and semconv notes

- **Python 3.12** is supported by `opentelemetry-api`, `opentelemetry-sdk`,
  and all instrumentors above (they support `>=3.9`/`>=3.10`); no 3.12-specific
  caveats were found in the primary sources. The collector is a separate Go
  binary and is unaffected by the Python version.
- The instrumentors' semconv status is **"migration"**: HTTP semantic
  conventions are migrating from legacy `http.*` to `http.request.*` /
  `url.*`, selectable via the standard
  `OTEL_SEMCONV_STABILITY_OPT_IN` mechanism ([HTTP semconv stability,
  opentelemetry.io](https://opentelemetry.io/docs/specs/semconv/general/semconv-stability/)).
  Set the env var explicitly so span/attribute names are stable once wired.
- GenAI attributes (`gen_ai.*`) used by Langfuse's mapping are still evolving
  (Langfuse states this explicitly); treat `gen_ai.*` as the recommended
  convention to follow, not a frozen one.

## 4. Q2 — Stage-level spans: where the seams are and what to tag

The five stages map onto existing call sites. The controller
(`api/src/api/controllers/qa_controller.py:run_query`) orchestrates
Harness A; `core/src/core/retrieval/query.py:retrieve_relevant_chunks` owns
retrieval + rerank; the clients own the provider calls; `depth_dive` owns
Harness B. The pattern that avoids touching the orchestration everywhere is:
**open the span at the start of the named unit, close it at the end** via
`tracer.start_as_current_span`, and set attributes on the returned span.

Sources for the span API: [OTel Python basic-tracing
example](https://github.com/open-telemetry/opentelemetry-python/blob/main/docs/examples/basic_tracer/basic_trace.py),
[OTLP HTTP exporter
config](https://github.com/open-telemetry/opentelemetry-python/blob/main/exporter/opentelemetry-exporter-otlp-proto-http/src/opentelemetry/exporter/otlp/proto/http/trace_exporter/__init__.py),
[Langfuse OTel property mapping](https://langfuse.com/docs/opentelemetry/get-started).

| Stage | Seam (where to open the span) | Meaningful span attributes |
|---|---|---|
| **embed** | `core/clients/embeddings_client.py:EmbeddingsClient.embed` (and `aembed`) | `gen_ai.system="openai"`, `gen_ai.operation.name="embedding"`, `gen_ai.request.model` (e.g. `text-embedding-3-small`), `embedding.batch_size`, count of vectors, input char count, latency (auto from span). |
| **retrieve** | `core/retrieval/query.py:retrieve_relevant_chunks` | `db.system="postgresql"`, `db.operation.name` (dense / sparse), `retrieval.hybrid_search` (bool), `retrieval.ef_search`, candidate count from dense + sparse, fused count, `retrieval.grounded` (bool), latency. SQLAlchemy instrumentor adds the underlying `db.statement` child spans. |
| **rerank** | `core/clients/reranker_client.py:CohereReranker.rerank` | `gen_ai.system="cohere"`, `gen_ai.operation.name="rerank"`, `gen_ai.request.model`, candidate count in, `top_k` out, whether a rate-limit fallback to RRF occurred. |
| **generate** | `core/clients/llm_client.py:LLMClient.chat` (Harness A) and `depth_dive/generation/generation_agent.py:run_generation` (Harness B) | `gen_ai.system="openai"`, `gen_ai.operation.name="chat"`, `gen_ai.request.model`, `gen_ai.usage.input_tokens`, `gen_ai.usage.output_tokens`, `gen_ai.response.id`, fallback-note flag for malformed depth-dive output. |
| **web-search** | `depth_dive/web_search/client.py:OpenAIWebSearchClient.search` | `gen_ai.system="openai"`, `gen_ai.operation.name="web_search"`, `web_search.query` (the search intent), result count, `web_search.failed` (bool) for the ADR-0013 degraded path. |

Span naming, shape, and parentage:

- The FastAPI instrumentor creates a **root server span** per request. Every
  manual stage span opened inside the request handler (via
  `tracer.start_as_current_span`) is automatically a **child** of that root
  because it runs on the same context. No explicit parent linking is needed.
- Recommended names follow the OTel convention `<kind> <name>` style, e.g.
  `retrieve`, `rerank`, `generate`, `embed`, `web-search` — or, to align with
  Langfuse's observation model, open them with a name plus the
  `langfuse.observation.type` attribute where a "generation" type is wanted
  for LLM calls.
- **Token usage** is available from the OpenAI response object
  (`completion.usage` / `CreateEmbeddingResponse.usage`) already parsed in the
  clients; set `gen_ai.usage.*` there so Langfuse records token usage/cost and
  Prometheus can derive token metrics.
- **Batch context spans** (Harness B): the same client seams serve both
  harnesses (both call `LLMClient.chat`, `Embedder.embed`), so instrumenting
  the clients covers Harness A and Harness B with one change; `run_generation`
  additionally tags the depth-dive fallback state.
- **Do not set the deprecated `trace.input`/`trace.output` on every span.**
  Langfuse v4 maps trace/observation input/output from `langfuse.*` or
  `gen_ai.*` attributes; set `gen_ai.prompt` / `gen_ai.completion` only where
  content is wanted (and be deliberate, since it stores prompt text).

## 5. Q3 — Exporter wiring: direct vs collector, HTTP vs gRPC

### 5.1 Why a standalone collector (fan-out)

The [OTel Collector
docs](https://opentelemetry.io/docs/collector/) explicitly frame the choice:
sending directly to a backend is fine "for trying out… or in a development or
small-scale environment," but they **recommend a collector alongside the
service** so the app offloads quickly and the collector handles **retries,
batching, and filtering**. That guidance plus the "one instrumentation, two
backends" lock makes the collector the correct answer here: the app exports
OTLP once (to the collector), and the collector fans out to Langfuse (traces)
and Prometheus (metrics). No in-app dual-exporter code, no per-backend SDK in
the app.

### 5.2 OTLP/HTTP vs OTLP/gRPC — choose HTTP

- The Python SDK ships both
  [`opentelemetry-exporter-otlp-proto-http`](https://github.com/open-telemetry/opentelemetry-python/blob/main/exporter/opentelemetry-exporter-otlp-proto-http/README.rst)
  and a gRPC exporter. The collector's `otlp` receiver listens on both
  (`grpc: 4317`, `http: 4318`).
- **Langfuse only accepts OTLP over HTTP** (HTTP/JSON and HTTP/protobuf);
  **gRPC is not supported** ([Langfuse OTel
  docs](https://langfuse.com/docs/opentelemetry/get-started)). Therefore the
  end-to-end protocol must be **OTLP/HTTP** at least on the final leg.
- To keep a single protocol all the way through, export **OTLP/HTTP from the
  app to the collector** (port 4318). This also avoids pulling `grpcio` as a
  runtime dep in the Python app. The app config is simply
  `OTEL_EXPORTER_OTLP_ENDPOINT=http://otel-collector:4318`
  (the SDK appends `/v1/traces` and `/v1/metrics`; see the exporter source
  linked above).

### 5.3 Recommended collector topology

```
api (FastAPI) ──OTLP/HTTP:4318──▶ otel-collector ──traces (otlphttp)──▶ langfuse (/api/public/otel)
                                  │   otelcol-contrib
                                  └──metrics (prometheus :8889/metrics)──▶ Prometheus ──▶ Grafana
```

docker-compose gains two services: an `otel-collector` (image
`otel/opentelemetry-collector-contrib`) and Prometheus (plus Grafana, and the
self-hosted Langfuse stack). Collector config has **two pipelines** fed by one
`otlp` receiver:

- `traces`: receivers `[otlp]` → processors `[memory_limiter, batch]` →
  exporter `otlphttp/langfuse` (endpoint `http://langfuse:3000/api/public/otel`,
  Basic-Auth headers).
- `metrics`: receivers `[otlp]` → processors `[memory_limiter, batch]` →
  exporter `prometheus` (endpoint `0.0.0.0:8889`, exposing `/metrics`), which
  Prometheus scrapes. The [Prometheus exporter
  README](https://github.com/open-telemetry/opentelemetry-collector-contrib/blob/main/exporter/prometheusexporter/README.md)
  confirms it "exports data in the Prometheus format… which allows it to be
  scraped by a Prometheus server," stability **beta (metrics)**, present in the
  core and contrib distributions.

To turn stage **spans** into service **metrics** (request rate / latency /
error by stage) without duplicating manual metric code, use the collector's
span-metrics **connector** (derives RED metrics from spans) feeding the
Prometheus exporter — this keeps the app emitting only spans and lets
Prometheus/Grafana get stage-level metrics. (See §9 for the verification caveat.)

## 6. Q4 — Langfuse ingestion path

Primary source: [Langfuse "OpenTelemetry (OTEL) for LLM Observability"
docs](https://langfuse.com/docs/opentelemetry/get-started), read 2026-08-15.

- **Self-hosted Langfuse consumes OTel spans directly.** It exposes an OTel
  backend on `/api/public/otel` (the OTLP endpoint; trace endpoint
  `/api/public/otel/v1/traces`). Auth is HTTP **Basic Auth** over a base64
  `pk-lf-…:sk-lf-…` keypair, plus the header `x-langfuse-ingestion-version: 4`
  for real-time ingestion on the v4 model.
- **There is NO dedicated "Langfuse exporter" component** in the
  OpenTelemetry collector contrib. The collector reaches Langfuse with a
  **standard `otlphttp` exporter** configured with Langfuse's endpoint and
  headers — exactly the collector YAML Langfuse publishes
  (`exporters: otlphttp/langfuse: endpoint: …/api/public/otel; headers:
  Authorization: "Basic ${AUTH_STRING}"; x-langfuse-ingestion-version: "4"`).
- **So it must NOT receive data via its own SDK to be part of this wiring.**
  The OTel-native **Langfuse SDK v4** exists and is a thin layer over the
  official OpenTelemetry client (it auto-converts spans into Langfuse
  observations, handles token/cost/prompt helpers). But adopting it in the app
  would add a second, Langfuse-specific path. The collector fan-out keeps the
  lock: **the app emits standard OTel only; the collector sends traces to
  Langfuse via `otlphttp`.** This satisfies "single instrumentation exporting
  to both."
- **Attribute mapping matters for useful traces.** Langfuse maps OTel spans to
  its data model from `gen_ai.*` conventions plus the `langfuse.*` namespace
  (e.g. `langfuse.trace.metadata.*`, `langfuse.observation.*`). Any span with a
  `model` attribute is tracked as a **generation**. Token usage comes from
  `gen_ai.usage.*`; cost from `gen_ai.usage.cost`. For trace-level fields
  (userId/sessionId/metadata/tags), Langfuse recommends OpenTelemetry
  **Baggage + BaggageSpanProcessor** so those attributes appear on **all**
  spans — relevant if we want per-user filtering.
- **Version note:** the OTel endpoint landed in Langfuse **v3.22.0** and is
  improved since; the docs advise upgrading self-hosted installs to the latest
  if 4xx errors appear on the endpoint. Self-hosted Langfuse runs web + worker
  + Postgres + Clickhouse + Redis + S3 ([self-hosting
  docs](https://langfuse.com/docs/deployment/self-host)) — heavier than the
  repo's current two-service compose, and docker-compose is the supported
  low-scale path.

## 7. Findings at a glance

| Decision point | Finding |
|---|---|
| FastAPI instrumentor | `opentelemetry-instrumentation-fastapi`, `fastapi ~= 0.92` floor; repo's 0.141.1 ✅; server request spans + metrics; applied via `FastAPIInstrumentor.instrument_app(app)`. |
| HTTP-layer / DB | `opentelemetry-instrumentation-httpx` (✅) and `opentelemetry-instrumentation-sqlalchemy` (`< 2.1.0`; repo 2.0.x ✅). |
| OpenAI instrumentor | No official GA package; PyPI `opentelemetry-instrumentation-openai` is third-party OpenLLMetry. Use manual stage spans instead. |
| Stage spans | Manual children of the FastAPI root span at the seams in §4 (controller orchestrates, clients own provider calls). |
| Protocol | OTLP/**HTTP** end-to-end (Langfuse is HTTP-only, no gRPC). |
| Topology | App → `otel-collector` (compose) → fan-out: `otlphttp`→Langfuse, `prometheus`→Prometheus/Grafana. |
| Langfuse ingestion | Direct OTel ingestion on `/api/public/otel` via standard `otlphttp` exporter; no dedicated Langfuse collector exporter; SDK not required for the single-path lock. |
| Baggage | Use OTel Baggage + `BaggageSpanProcessor` to propagate user/session/tags to all spans for Langfuse filtering. |

## 8. What the implementation should remember

1. **One dependency set, typed.** Add `opentelemetry-api`, `opentelemetry-sdk`,
   `opentelemetry-exporter-otlp-proto-http`, and the fastapi/httpx/sqlalchemy
   instrumentors. Instantiate the tracer via `trace.get_tracer(__name__)` and
   keep manual spans inside the existing client classes so the code stays
   `mypy --strict` clean.
2. **Instrument at the factory + client seams.** Call
   `FastAPIInstrumentor.instrument_app(app)` in `api/main.py` after
   `create_app()` (the server span becomes the root); open stage spans in the
   clients/retrieval seams in §4. Both harnesses are covered because they share
   the same clients.
3. **OTLP/HTTP to a collector, once.** Set
   `OTEL_EXPORTER_OTLP_ENDPOINT=http://otel-collector:4318`; do not wire
   Langfuse/Prometheus exporters into the app.
4. **Tag for Langfuse mapping.** Use `gen_ai.*` + `langfuse.*` attributes so
   LLM stages become generations with token usage/cost; use Baggage for
   trace-level user/session filtering.
5. **Respect module boundaries.** Keep OTel setup code out of `retrieval_qa`;
   `depth_dive` and `api` may pull from `core`, not from each other.

## 9. Cited primary sources

OpenTelemetry (Python / SDK / collector):

- https://github.com/open-telemetry/opentelemetry-python-contrib/blob/main/instrumentation/README.md
- https://github.com/open-telemetry/opentelemetry-python-contrib/blob/main/instrumentation/opentelemetry-instrumentation-fastapi/README.rst
- https://github.com/open-telemetry/opentelemetry-python-contrib/blob/main/instrumentation/opentelemetry-instrumentation-fastapi/src/opentelemetry/instrumentation/fastapi/__init__.py
- https://github.com/open-telemetry/opentelemetry-python-contrib/blob/main/instrumentation/opentelemetry-instrumentation-httpx/README.rst
- https://github.com/open-telemetry/opentelemetry-python-contrib/blob/main/instrumentation/opentelemetry-instrumentation-sqlalchemy/README.rst
- https://github.com/open-telemetry/opentelemetry-python-contrib/blob/main/instrumentation/opentelemetry-instrumentation-psycopg2/README.rst
- https://github.com/open-telemetry/opentelemetry-python/blob/main/docs/examples/basic_tracer/basic_trace.py
- https://github.com/open-telemetry/opentelemetry-python/blob/main/exporter/opentelemetry-exporter-otlp-proto-http/README.rst
- https://github.com/open-telemetry/opentelemetry-python/blob/main/exporter/opentelemetry-exporter-otlp-proto-http/src/opentelemetry/exporter/otlp/proto/http/trace_exporter/__init__.py
- https://opentelemetry.io/docs/collector/
- https://github.com/open-telemetry/opentelemetry-collector-contrib/blob/main/exporter/prometheusexporter/README.md
- https://opentelemetry.io/docs/specs/semconv/general/semconv-stability/
- https://pypi.org/project/opentelemetry-instrumentation-openai/

Langfuse:

- https://langfuse.com/docs/opentelemetry/get-started
- https://langfuse.com/docs/deployment/self-host
- https://langfuse.com/docs/observability/sdk/overview

Repo (seams, verified in-tree):

- `api/src/api/controllers/qa_controller.py:run_query`
- `core/src/core/retrieval/query.py:retrieve_relevant_chunks`
- `core/src/core/clients/embeddings_client.py`, `llm_client.py`, `reranker_client.py`
- `depth_dive/src/depth_dive/generation/generation_agent.py`, `depth_dive/src/depth_dive/web_search/client.py`
- `docker-entrypoint.sh`, `api/src/api/server.py`

## 10. Could not verify

- Whether a dedicated, SIG-owned `opentelemetry-instrumentation-openai`
  exists and is published under the same PyPI name as the OpenLLMetry one, or
  whether the OpenTelemetry GenAI instrumentors are shipped under different
  package names/status. (PyPI unambiguously maps the package to
  `github.com/traceloop/openllmetry`.)
- The exact current stability level of the collector **span-metrics
  connector** (used to derive RED metrics from stage spans); the OTel
  collector docs describe connectors at component level but the per-component
  stability badge for `spanmetricsconnector` was not re-read directly.
- Whether `uvicorn` will ever ship its own server instrumentor; today the
  FastAPI/ASGI instrumentor owns the request span and none is needed.
- The precise Langfuse collector-component status (we confirmed only that the
  Langfuse docs use a **standard `otlphttp` exporter**, not a bespoke Langfuse
  exporter component) — consistent with no "langfuse exporter" existing in
  collector-contrib, but not independently enumerated.

## Recommendation

Instrument the app with the standard OpenTelemetry Python SDK and a standalone
OTel Collector (`otel/opentelemetry-collector-contrib`) added to the existing
docker-compose: in the app, call `FastAPIInstrumentor.instrument_app(app)`
after `create_app()` in `api/main.py`, add `opentelemetry-instrumentation-httpx`
and `opentelemetry-instrumentation-sqlalchemy`, and open the five stage spans —
embed, retrieve, rerank, generate, web-search — as manual named child spans via
`tracer.start_as_current_span` at the seams already in the code
(`core/clients/embeddings_client.py`, `core/retrieval/query.py`,
`core/clients/reranker_client.py`, `core/clients/llm_client.py` and
`depth_dive/generation/generation_agent.py`, `depth_dive/web_search/client.py`)
under the FastAPI request root span, tagging `gen_ai.*`/`langfuse.*` attributes
(model, chunk count, token usage) so Langfuse maps them into generations; the
app exports everything over OTLP/HTTP once via
`OTEL_EXPORTER_OTLP_ENDPOINT=http://otel-collector:4318`, and the collector
fans out: a traces pipeline sends spans to self-hosted Langfuse through a
standard `otlphttp` exporter pointed at `http://langfuse:3000/api/public/otel`
with Basic-Auth `pk-lf-…:sk-lf-…` headers plus `x-langfuse-ingestion-version: 4`,
and a metrics pipeline (optionally fed by the collector's span-metrics
connector) exposes `/metrics` via the Prometheus exporter for Prometheus/Grafana
to scrape — no in-app dual exporter, no Langfuse SDK in the app, and no
third-party OpenAI instrumentor (the PyPI one is OpenLLMetry/Traceloop). Rationale: a single OTLP/HTTP path to one fan-out collector is the OTel-recommended topology, reaches both backends from one instrumentation, and matches Langfuse's HTTP-only OTel ingestion.
