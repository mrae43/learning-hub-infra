# Codebase Structure

```
learning-hub/
├── retrieval_qa/                        # Retrieval QA (extractable later)
│   ├── src/retrieval_qa/
│   │   ├── chunking/                 # Document-type-specific chunkers
│   │   │   ├── base.py               # Base chunker class
│   │   │   ├── paper_chunker.py
│   │   │   ├── book_chunker.py
│   │   │   ├── documentation_chunker.py
│   │   │   └── _html_utils.py         # HTML parsing helpers
│   │   ├── _utils.py                 # Internal utility functions
│   │   └── __init__.py
│   ├── tests/retrieval_qa/           # chunker tests
│   ├── tests/eval/                   # retrieval eval (path-gated, eval_set.yaml + eval_vectors.json)
│   └── pyproject.toml
├── depth_dive/                        # Depth Dive generation (Harness B, ADR-0020)
│   ├── src/depth_dive/
│   │   ├── transform.py               # Passage transform → model-ready carrier
│   │   ├── harness.py                 # Harness B entrypoint (ADR-0020)
│   │   ├── framing/                   # Framing agent (treatment routing + search intent)
│   │   │   ├── framing_agent.py
│   │   │   └── __init__.py
│   │   ├── assembly/                  # Assembly agent (corpus grounding + web search + generation)
│   │   │   ├── assembly_agent.py
│   │   │   └── __init__.py
│   │   ├── generation/                # LLM generation turn + fallback scene graph
│   │   │   ├── generation_agent.py
│   │   │   ├── fallback_animation.py
│   │   │   └── __init__.py
│   │   ├── web_search/                # Web-search wrapper (retry-once + fallback)
│   │   │   ├── client.py
│   │   │   ├── wrapper.py
│   │   │   └── __init__.py
│   │   └── __init__.py
│   ├── tests/depth_dive/              # transform/harness/framing/assembly/generation/web_search tests
│   └── pyproject.toml
├── core/                             # Shared (may stay here or move to common/ later)
│   ├── src/core/
│   │   ├── types/                    # Shared schemas
│   │   │   ├── chunk.py
│   │   │   ├── document.py
│   │   │   ├── chat.py               # Chat / conversation models
│   │   │   ├── responses.py          # HarnessAResponse, etc.
│   │   │   ├── retrieval_config.py   # Retrieval config models
│   │   │   ├── depth_dive.py         # HarnessBRequest/Response, scene graph types
│   │   │   ├── captured_passage.py   # Captured Passage five-type model (ADR-0021)
│   │   │   └── __init__.py
│   │   ├── config/
│   │   │   ├── settings.py           # Pydantic settings
│   │   │   └── __init__.py
│   │   ├── clients/                  # API clients (hosted inference, embeddings)
│   │   │   ├── llm_client.py
│   │   │   ├── embeddings_client.py
│   │   │   └── reranker_client.py
│   │   ├── retrieval/                # Shared retrieval primitives (ADR-0019)
│   │   │   ├── query.py              # Full hybrid pipeline
│   │   │   ├── neighbors.py          # Dense semantic-neighbor search
│   │   │   ├── parents.py            # Parent-chunk fetch
│   │   │   └── documents.py          # Document-level lookup helpers
│   │   ├── database/                 # pgvector wrapper, Alembic migrations
│   │   │   ├── connection.py
│   │   │   ├── schema.py
│   │   │   └── migrations/           # env.py + script.py.mako + versions/ (2 revisions)
│   │   ├── exceptions.py             # Named exception types
│   │   ├── utils.py                  # Shared text utilities (e.g. count_tokens)
│   │   └── __init__.py
│   ├── tests/core/                   # types, clients, retrieval, migration tests
│   └── pyproject.toml
├── api/                              # FastAPI server (thin controller layer)
│   ├── src/api/
│   │   ├── routes/
│   │   │   ├── retrieval_qa.py       # /query endpoint
│   │   │   ├── health.py             # /health endpoint
│   │   │   ├── ingest.py             # /ingest endpoint
│   │   │   ├── documents.py          # /documents/{id} endpoint
│   │   │   └── dive.py               # /dive endpoint (Harness B)
│   │   ├── controllers/
│   │   │   └── qa_controller.py      # Orchestrates Harness A
│   │   ├── dependencies.py           # FastAPI dependency injection
│   │   ├── prompt.py                 # Prompt templates
│   │   ├── server.py                 # FastAPI app factory
│   │   ├── main.py                   # Entry point
│   │   └── __init__.py
│   ├── tests/api/                    # route + controller + prompt tests
│   │   ├── controllers/               # qa_controller tests
│   │   └── test_dive.py               # /dive route tests
│   ├── tests/conftest.py
│   └── pyproject.toml
├── ingestion/                        # Document upload & background task logic
│   ├── src/ingestion/
│   │   ├── models.py                 # Pydantic models for ingestion
│   │   ├── tasks.py                  # FastAPI BackgroundTasks logic
│   │   ├── pipeline.py               # Ingest → chunk → embed → store
│   │   ├── splitting.py              # Document splitting logic
│   │   └── __init__.py
│   ├── tests/ingestion/              # pipeline/splitting/tasks unit tests
│   ├── tests/integration/            # real-API ingestion tests (integration-marked)
│   └── pyproject.toml
├── mock_upstream/                   # OpenAI-compatible mock of hosted APIs for volume load runs
│   ├── src/mock_upstream/
│   │   ├── app.py                   # FastAPI app: /v1/embeddings, /v1/chat/completions, /v1/responses
│   │   ├── settings.py              # MOCK_* latency configuration
│   │   └── __init__.py
│   ├── tests/mock_upstream/         # endpoint shape + OpenAI SDK compatibility tests
│   ├── Dockerfile                   # Standalone image (never enters the api image)
│   └── pyproject.toml
├── scripts/                          # Eval & chunk-size tuning tooling
│   ├── __init__.py
│   ├── eval_metrics.py               # Recall@k, MRR, is_hit metrics
│   ├── generate_eval_vectors.py      # Eval vector generation utility
│   ├── manual_ingest_smoke.py        # Manual ingestion smoke script
│   ├── run_chunk_tuning.py           # Chunk-size tuning orchestrator
│   ├── seed_schema.py                # Schema seeding for tuning
│   ├── pyproject.toml
│   └── (tests live in root tests/scripts/)
├── eval_corpus/                      # Retrieval eval & tuning corpus
│   ├── books/                        # DDIA excerpts, deep learning concepts
│   ├── papers/                       # FlashAttention, vLLM paged attention
│   ├── synthetic/                    # RAG reference, SOLID principles
│   ├── eval_set_tuning.yaml          # Tuning query set
│   └── eval_vectors_*.json           # Pre-generated eval vectors (256/512/1024-dim)
├── tests/                            # Root-level tests (scripts/ eval tooling)
│   └── scripts/
├── docs/
│   ├── adr/                          # 0001–0021 (skip 0008; 0015 supersedes 0007 scorer)
│   ├── agents/                       # Agent skill docs (issue tracker, triage labels, domain)
│   │   ├── issue-tracker.md
│   │   ├── triage-labels.md
│   │   └── domain.md
│   ├── research/                     # Research notes (multimodal inputs, neuroscience methods)
│   │   ├── multimodal-passage-inputs.md
│   │   └── neuroscience-methods-output-types.md
│   ├── specs/                        # Feature specs
│   │   └── depth-dive-redefinition.md
│   ├── ai-system-tree.md
│   ├── coding-standards.md
│   ├── commit-instructions.md
│   ├── security-mvp-guide.md
│   ├── tech-stack.md
│   └── .out-of-scope/                # Deliberately deferred/decided-against notes
│       ├── custom-deepeval-metric-async.md
│       ├── docker-compose.md
│       ├── gha-cache-config-dedup.md
│       └── pytest-module-level-test-data.md
├── .github/workflows/
│   ├── ci.yml                        # ruff, mypy, pytest, import-linter, commitlint, actionlint, docker build gate
│   ├── cd.yml                        # Docker build + push to GHCR, changelog
│   ├── security.yml                  # CodeQL, secret scanning (gitleaks), Trivy, license audit
│   └── dependabot-auto-merge.yml     # Auto-merge for low-risk Dependabot bumps
├── pyproject.toml                    # Root: uv workspace, ruff config, import-linter contracts
├── conftest.py                       # Root test fixtures (282 lines, shared by all packages)
├── alembic.ini                       # Alembic configuration for DB migrations
├── Makefile                          # Thin docker-compose wrapper for local dev (up/logs/down/deploy/up-edge/up-observability/down-observability)
├── docker-entrypoint.sh              # Container entrypoint: alembic upgrade + uvicorn
├── commitlint.config.mjs             # Conventional Commits enforcement
├── docker-compose.yml                # Local dev: PostgreSQL + pgvector (+ api); observability/load profiles
├── compose.deploy.yml                # Deploy override: pull GHCR image instead of building
├── compose.edge.yml                  # Edge gate override (Caddy basic-auth + TLS), inert by default
├── Caddyfile                         # Edge gate reverse-proxy config (decision #269), inert by default
├── observability/                    # Metrics-path configs (issue #289): dashboards-as-code, scrapers, collector
│   ├── otel-collector/config.yml     # OTLP receiver + span-metrics connector + Prometheus exporter
│   ├── prometheus/prometheus.yml     # Scrape target: otel-collector:8889
│   └── grafana/                      # Provisioning (datasource + provider) and committed dashboard JSON
├── Dockerfile                        # Multi-stage build (all 7 packages)
├── .env.example
├── skills-lock.json                  # Pinned agent skill versions
├── tuning_results.json               # Chunk-size tuning run output
├── AGENTS.md                         # Session notes for AI coding tools
├── CONTEXT.md                        # Domain glossary
└── README.md
```

**Why this is better for your goals:**

1. **Retrieval-centered** — Retrieval QA and Depth Dive are top-level, self-contained modules. When you extract Retrieval QA into its own repo later (post-MVP), the `git subtree split` is clean.
2. **Shared core/** — only truly shared things (types, API clients, config, DB, retrieval) live here. No false "shared" abstractions you don't need yet.
3. **No generic agent cargo** — no planner, no executor, no memory abstraction that doesn't apply to RAG. Retrieval is deterministic; it doesn't need those patterns.
4. **Tool-specificity** — web search lives _inside_ Depth Dive, not a generic tool, making it clear it's a Depth Dive-specific capability.
5. **Clean import boundaries** — matches ADR-0011's import-linter rules exactly (retrieval_qa ↔ core, depth_dive ↔ core, never retrieval_qa ↔ depth_dive).
6. **Extractable ingestion** — ingestion logic is modular enough that when you graduate to `arq` + Redis (ADR-0006), you can slot it in without restructuring.
7. **Eval/tuning kept outside packages** — `scripts/` tooling plus `eval_corpus/` material stay out of runtime package dirs, and the eval tests live at root `tests/scripts/`, so per-package CI stays clean.

**Where to add things** — the `AGENTS.md` file at the root now documents exactly where things go (the "Where to add things" section). Use that as the canonical reference for future contributors.
