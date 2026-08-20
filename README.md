# Learning Hub

A hand-rolled RAG study tool for learning AI/ML from papers, books, and documentation through grounded interactive Q&A.

## What problem

**For the learner.** Reading AI/ML papers, books, and documentation alone is slow. This tool lets you query your personal corpus interactively and get answers grounded in your sources — building toward richer, multi-modal learning sessions (Depth Dive) that accelerate understanding through neuroplasticity triggers.

**For me as the builder.** Hand-rolling the RAG pipeline — chunking, embedding, retrieval, prompt assembly — against raw client libraries is the only way to deeply understand every mechanism. That understanding is the foundation for the Depth Dive features. Using LangChain/LlamaIndex would hide the mechanics I'm here to learn.

## How it works

**Upload & ingest.** Submit a PDF or EPUB via `POST /ingest`. The server validates the file, chunks it by document type (paper, book, documentation), embeds each chunk via Google `text-embedding-004` or OpenAI `text-embedding-3-small`, and stores it in a pgvector HNSW index. You get a document ID immediately; ingestion continues in the background. Poll progress with `GET /documents/{id}`.

**Query (MVP).** Send `{query: str}` to `POST /query`. The system retrieves candidate chunks from your entire corpus using a hybrid dense + sparse retrieval pipeline, reranks the top candidates with a cross-encoder, and assembles the most relevant grounded context for model generation. The response is a structured `HarnessAResponse` — answer text, cited passages, and a `grounded: bool` flag so you know whether the answer actually came from your documents.

**Depth Dive (post-MVP 1).** Send a `CapturedPassage` to `POST /dive`. The framing agent resolves the treatment set and search intent, the assembly agent grounds the passage against the corpus (with retry-once web search when the brief calls for it), and the harness produces an `interactive_animation` scene graph — the richer, multi-modal explanation format ([ADR-0020](./docs/adr/0020-depth-dive-harness-b-architecture.md)). The scene graph moves to deterministic template generation, rendered to self-contained HTML delivered through a Claude Code plugin ([ADR-0022](./docs/adr/0022-deterministic-scene-graph-and-self-contained-html.md), [ADR-0023](./docs/adr/0023-claude-code-plugin-as-mvp-client.md)).

**Roadmap.**
| Phase | What | Status |
|---|---|---|
| MVP | Grounded Q&A against your personal corpus | ✅ Operational |
| Post-MVP 1 | **Depth Dive** — richer explanations (text + diagrams + code) with agentic web search | ✅ Operational |
| Post-MVP 2 | **Concept Linking + Retrieval Practice / Spaced Repetition** — query-independent relationship surfacing and review-style learning triggers | 📅 Planned |

> **Status:** Early implementation — tracer bullet complete. All seven packages (`core/`, `retrieval_qa/`, `depth_dive/`, `api/`, `ingestion/`, `mock_upstream/`, `scripts/`) have implementation code (~7,800 source lines, plus ~10,700 lines of tests). `depth_dive/` runs the full Harness B: transform → framing → assembly (corpus grounding + retry-once web search) → LLM generation of the `interactive_animation` scene graph. Ingestion pipeline, Harness A query pipeline, five API endpoints (`POST /ingest`, `GET /documents/{id}`, `POST /query`, `POST /dive`, `GET /health`), and the `scripts/` eval & chunk-size-tuning tooling are operational. The `mock_upstream/` package stands in for hosted OpenAI APIs during volume load runs (activated by the `load` compose profile). See [docs/](./docs/) for architecture decisions and plans.

## Local development

The root `Makefile` wraps the local dev container lifecycle defined in `docker-compose.yml` (Postgres + pgvector): `make up` to start the stack in the background, `make logs` to tail service logs (`make logs SERVICE=postgres` scopes to one service), and `make down` to stop and remove containers/network — `down` never deletes the pgvector data volume. `make load-up` starts the volume-load stack: the `load` compose profile adds the OpenAI-compatible `mock-upstream` service and points the api at it (`OPENAI_BASE_URL`), so `/query` and `/dive` return deterministic answers with no real API key or outbound calls (`make load-down` stops it). Run `make help` to list every target. Docker's daemon must be running; `up`/`logs`/`down` fail fast with a clear message if it isn't. Per-package checks (`ruff`, `mypy`, `pytest`) stay explicit `uv run` invocations.

**Deploying the demo.** CI is the single image build path — the deployed stack pulls, never builds. `make deploy` pulls the published image (`ghcr.io/mrae43/learning-hub-infra:latest` by default, via the `compose.deploy.yml` override), starts the stack with `--no-build`, and polls `GET /health` until ready (bounded). Copy `.env.example` to `.env` first and fill in `OPENAI_API_KEY`. Rollback is manual — re-run with a previous tag, `make deploy IMAGE_TAG=1.2.3` (published semver tags carry no `v` prefix). `cd.yml` only pushes an image after a boot + postgres + `/health` smoke test passes. The deploy override uses the Compose `!reset` tag, so Docker Compose v2.24+ is required.

**The edge gate (Caddy).** For a public demo URL there is a scaffolded edge gate (decision #269): a `Caddyfile` + `compose.edge.yml` override that path-routes one domain to the api (and the Langfuse/Grafana/Prometheus UIs once the observability profile is up), terminates TLS, and enforces basic-auth on every api route except `GET /health` — the API docs routes (`/docs`, `/redoc`) are exposed but sit behind the auth wall. ADR-0018 stays intact: the api is still auth-less; gating happens only at the edge. **It is inert by default** — `make up` / `make deploy` never load it.

To activate locally (self-signed TLS, no domain needed):
```bash
# one-time: generate a bcrypt hash for the demo credential
docker run --rm caddy:2-alpine caddy hash-password
# then put the credential in .env (compose doubles every `$` in the hash):
#   EDGE_GATE_AUTH="demo $$2a$$10$$..."        (see .env.example)
#   EDGE_GATE_DOMAIN=localhost
#   EDGE_GATE_TLS=tls internal
make up-edge
# https://localhost/health stays open; everything else needs the credentials
make down-edge
```
For the public phase, layer the override on the deploy files and let Caddy provision a Let's Encrypt certificate for the real domain:
```bash
# in .env: EDGE_GATE_DOMAIN=demo.example.com and a real EDGE_GATE_AUTH
# (leave EDGE_GATE_TLS empty so automatic HTTPS issues a Let's Encrypt cert)
docker compose -f docker-compose.yml -f compose.deploy.yml -f compose.edge.yml up -d --no-build
```
`EDGE_GATE_AUTH` lives in `.env` (gitignored) — the committed `.env.example` holds only placeholders. Set `EDGE_GATE_TLS="tls internal"` for self-signed local runs (the `make up-edge` default); leave it empty for Let's Encrypt. Passing the credential inline on the shell needs single quotes (`EDGE_GATE_AUTH='demo $2a$...' make up-edge`) so the shell doesn't expand the hash's `$`.

**Observability (Langfuse + OTel Collector + Prometheus + Grafana + Alertmanager + exporters).** `make up-observability` starts the full `observability` compose profile (issues #289/#291/#292): self-hosted Langfuse (`http://localhost:3000` — traces), an `otel-collector` receiving the api's OTLP/HTTP spans, Prometheus scraping on `http://localhost:9090` with a 15-day retention window, Grafana serving dashboards-as-code on `http://localhost:3001`, Alertmanager on `http://localhost:9093` (issue #292), and node/postgres exporters. The collector fans every request's spans (the root span plus the five RAG stage spans from issue #295) out to Langfuse — headless-provisioned with the dev keys in `.env.example` — and derives RED metrics (request rate / error rate / latency by stage) via the span-metrics connector for Prometheus/Grafana. Langfuse runs with its own Postgres container/database/user (the app's pgvector database is untouched), and the exporters surface host metrics plus database metrics including an ingestion-failed counter (`learning_hub_ingestion_failed`) driven by a custom postgres-exporter query. The profile points the api's `OTEL_EXPORTER_OTLP_ENDPOINT` at the collector automatically. All host ports are loopback-only; `make down-observability` stops the stack without deleting the named data volumes. Dashboards and datasources are provisioned from `observability/` (dashboard JSON is committed as dashboards-as-code); the default Grafana login is `admin`/`admin` on a fresh volume, and the Langfuse demo user is `demo@learning-hub.local` / `demo-demo-1234`. The profile is additive — `make up` stays untouched.

**Alerting + runbook (issue #292).** Five Prometheus alert rules live in `observability/prometheus/alerts.yml` (service-down via a blackbox `/health` probe, upstream 502/503, p95 latency, internal 5xx, and ingestion-failed via the postgres-exporter custom query). Firing alerts fan out to Alertmanager, whose webhook sink URL comes from `ALERTMANAGER_WEBHOOK_URL` (`.env`) — **UI-only by default** (alerts visible at `http://localhost:9093`), delivery enabled once a real URL is set. Every alert maps one-to-one to a diagnosis + recovery entry in [`docs/runbook.md`](./docs/runbook.md), including how to induce each alert under a load run.

**Load generator (Locust).** A Locust load generator lives in `scripts/loadgen/` (decision #271, issue #290): one locustfile drives `/query` (~75%) and `/dive` (~25%) as weighted user scenarios plus a low-rate `/health` liveness task, replaying the `eval_corpus/eval_set_tuning.yaml` queries against the deployed api. The `LOAD_PROFILE` env var selects the run shape (default `volume`):
- **volume** — sustained run against the mock upstream (`make load-up` first): 10 users, 1/s spawn, no run-time limit (Ctrl+C to stop). Deterministic mock means the error-rate and p95 ceilings should hold.
- **smoke** — ~1 upstream user (plus a low-rate liveness user), no ramp-up, against the real API: stops after 50 upstream-backed requests (`/query` + `/dive`) so spend is bounded; 429 rate limits are logged but not counted as errors.

Run with `make load-run` / `make smoke-run` (override the target with `LOAD_TARGET_URL=http://...`, or bound a volume run with `LOAD_RUN_TIME=5m`). Each run evaluates the profile's success gates — run-level error rate ≤1% and per-endpoint p95 ceilings (`/health` <100ms, `/query` <2s volume / <10s smoke, `/dive` <5s volume / <30s smoke) — prints a plain-text report, and exits non-zero if any gate fails. The gates are first-pass placeholders that the alert rules key on and get sharpened against smoke baselines.

## Architecture

Structured monorepo with extractable module boundaries:

| Package         | Role                                                           |
| --------------- | -------------------------------------------------------------- |
| `core/`         | Shared types, config, API clients, database (`pgvector`), retrieval primitives |
| `retrieval_qa/` | Closed-corpus retrieval QA: chunking + retrieval eval (MVP)    |
| `depth_dive/`   | Depth Dive generation (post-MVP 1)                             |
| `api/`          | FastAPI server (thin routes → controllers)                     |
| `ingestion/`    | Document upload + background ingestion pipeline                |
| `mock_upstream/`| OpenAI-compatible mock of hosted embeddings/inference/web-search for volume load runs |
| `scripts/`      | Retrieval eval & chunk-size tuning tooling over `eval_corpus/`; Locust load generator under `scripts/loadgen/` |
| `eval_corpus/`  | Retrieval eval & tuning corpus (books, papers, synthetic)      |

## Why this tech stack

**Python + uv + FastAPI + Pydantic v2.** Python is the AI/ML ecosystem's lingua franca. uv is fast and natively supports the monorepo workspace structure. FastAPI is async-native, matching the I/O-bound calls to hosted embedding/inference APIs. Pydantic v2 pairs with FastAPI for typed request/response contracts.

**PostgreSQL + pgvector.** One production-proven database for app data and vectors — one fewer service to operate in MVP. Migration to Qdrant is deferred until pgvector's limits are concretely felt ([ADR-0002](./docs/adr/0002-pgvector-for-mvp.md)).

**Hand-rolled RAG (no LangChain/LlamaIndex).** Retrieval mechanics are what I'm here to build and understand. A framework would hide chunking strategy, similarity scoring, and prompt assembly behind abstractions. Retrieval is tuned for the project’s document structure with parent-child chunking, hybrid dense+sparse search, and reranking as described in [ADR-0017](./docs/adr/0017-important-but-gatable-retrieval-phase.md). This is the core pedagogical choice ([ADR-0003](./docs/adr/0003-handroll-rag-pipeline.md)).

**Hosted APIs for embeddings and inference.** My GPU (4GB VRAM) makes local embedding slow and conflates retrieval quality with GPU tuning. Hosted APIs let me judge retrieval in isolation. Self-hosted vLLM is deferred ([ADR-0001](./docs/adr/0001-hosted-inference-api-for-mvp.md), [ADR-0004](./docs/adr/0004-hosted-embedding-api.md)).

**FastAPI BackgroundTasks for ingestion.** No lost jobs yet — a dedicated queue (arq/Redis) gets added when that becomes a real pain point ([ADR-0006](./docs/adr/0006-backgroundtask-for-mvp-ingestion.md)).

**Structured monorepo.** Seven packages with independent `pyproject.toml` files, test suites, and CI jobs — extractable later via `git subtree split` ([ADR-0005](./docs/adr/0005-structured-monorepo.md)). Module boundaries enforced by `import-linter` in CI ([ADR-0011](./docs/adr/0011-import-linter-module-boundaries.md)).

## Challenges solved

**Multi-format document chunking.** Papers, books, and documentation have different structures. Each gets a structure-aware chunker (section boundaries for papers, chapter boundaries for books, page/API-entry boundaries for documentation) that produces typed metadata via a JSONB registry — keeping each doc type's schema explicit and extensible.

**Module boundary enforcement.** A six-package monorepo needs real boundaries or they become fictional. `import-linter` in CI (`uv run lint-imports`) catches cross-package leaks before merge — e.g., `retrieval_qa` and `depth_dive` must never import each other.

**Document state machine.** Ingestion is async: validating → chunking → embedding → ready (or failed). The state is tracked in the database and pollable via `GET /documents/{id}`, so the user always knows where their upload stands.

**Structured response with groundedness.** Every query response is a typed `HarnessAResponse` with `answer`, `cited_passages`, and a `grounded: bool` flag. Clients can distinguish grounded answers (with sources) from ungrounded ones without text parsing — essential for trust in a learning tool.

## Key docs

- [CONTEXT.md](./CONTEXT.md) — domain glossary
- [docs/tech-stack.md](./docs/tech-stack.md) — MVP stack and post-MVP milestones
- [docs/ai-system-tree.md](./docs/ai-system-tree.md) — full directory layout
- [docs/security-mvp-guide.md](./docs/security-mvp-guide.md) — security hygiene rules for the MVP
- [docs/adr/](./docs/adr/) — architecture decision records (project constraints)
- [docs/coding-standards.md](./docs/coding-standards.md) — typing, docstrings, testing, error handling
- [docs/commit-instructions.md](./docs/commit-instructions.md) — Conventional Commits format
- [AGENTS.md](./AGENTS.md) — session notes for AI coding tools
- [.agents/skills/ask-matt/SKILL.md](./.agents/skills/ask-matt/SKILL.md) — router over the engineering skills; `/chain` is the AFK continuation for a non-empty work queue
