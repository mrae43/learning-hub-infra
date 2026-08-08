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
│   │   ├── retrieval/                # Retrieve from pgvector
│   │   │   └── query.py
│   │   ├── _utils.py                 # Internal utility functions
│   │   └── __init__.py
│   ├── tests/retrieval_qa/           # chunker + retrieval tests
│   ├── tests/eval/                   # retrieval eval (path-gated, eval_set.yaml + eval_vectors.json)
│   ├── pyproject.toml
│   └── README.md
├── depth_dive/                        # Depth Dive generation (stub — TODO)
│   ├── src/depth_dive/
│   │   └── __init__.py               # Package marker only
│   ├── tests/depth_dive/
│   │   └── test_smoke.py
│   └── pyproject.toml
├── core/                             # Shared (may stay here or move to common/ later)
│   ├── src/core/
│   │   ├── types/                    # Shared schemas
│   │   │   ├── chunk.py
│   │   │   ├── document.py
│   │   │   ├── chat.py               # Chat / conversation models
│   │   │   ├── responses.py          # HarnessAResponse, etc.
│   │   │   ├── retrieval_config.py   # Retrieval config models
│   │   │   └── __init__.py
│   │   ├── config/
│   │   │   ├── settings.py           # Pydantic settings
│   │   │   └── __init__.py
│   │   ├── clients/                  # API clients (hosted inference, embeddings)
│   │   │   ├── llm_client.py
│   │   │   ├── embeddings_client.py
│   │   │   └── reranker_client.py
│   │   ├── database/                 # pgvector wrapper, Alembic migrations
│   │   │   ├── connection.py
│   │   │   ├── schema.py
│   │   │   └── migrations/           # env.py + script.py.mako + versions/
│   │   ├── exceptions.py             # Named exception types
│   │   ├── utils.py                  # Shared text utilities (e.g. count_tokens)
│   │   └── __init__.py
│   ├── tests/core/                   # types, clients, migration tests
│   ├── pyproject.toml
│   └── README.md
├── api/                              # FastAPI server (thin controller layer)
│   ├── src/api/
│   │   ├── routes/
│   │   │   ├── retrieval_qa.py       # /query endpoint
│   │   │   ├── health.py             # /health endpoint
│   │   │   ├── ingest.py             # /ingest endpoint
│   │   │   └── documents.py          # /documents/{id} endpoint
│   │   ├── controllers/
│   │   │   └── qa_controller.py      # Orchestrates Harness A
│   │   ├── dependencies.py           # FastAPI dependency injection
│   │   ├── prompt.py                 # Prompt templates
│   │   ├── server.py                 # FastAPI app factory
│   │   ├── main.py                   # Entry point
│   │   └── __init__.py
│   ├── tests/api/                    # route + controller + prompt tests
│   ├── tests/conftest.py
│   ├── pyproject.toml
│   └── README.md
├── ingestion/                        # Document upload & background task logic
│   ├── src/ingestion/
│   │   ├── models.py                 # Pydantic models for ingestion
│   │   ├── tasks.py                  # FastAPI BackgroundTasks logic
│   │   ├── pipeline.py               # Ingest → chunk → embed → store
│   │   ├── splitting.py              # Document splitting logic
│   │   └── __init__.py
│   ├── tests/ingestion/              # pipeline/splitting/tasks unit tests
│   ├── tests/integration/            # real-API ingestion tests (integration-marked)
│   ├── pyproject.toml
│   └── README.md
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
│   └── eval_set_tuning.yaml          # Tuning query set
├── tests/                            # Root-level tests (scripts/ eval tooling)
│   └── scripts/
├── docs/
│   ├── adr/                          # 0001–0018 (skip 0008; 0015 supersedes 0007 scorer)
│   ├── agent/issue-tracker.md        # Code review tracker
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
├── conftest.py                       # Root test fixtures (276 lines, shared by all packages)
├── alembic.ini                       # Alembic configuration for DB migrations
├── Makefile                          # Thin docker-compose wrapper for local dev (up/logs/down)
├── docker-entrypoint.sh              # Container entrypoint: alembic upgrade + uvicorn
├── commitlint.config.mjs             # Conventional Commits enforcement
├── docker-compose.yml                # Local dev: PostgreSQL + pgvector
├── Dockerfile                        # Multi-stage build (all 6 packages)
├── .env.example
├── AGENTS.md                         # Session notes for AI coding tools
├── CONTEXT.md                        # Domain glossary
└── README.md
```

**Why this is better for your goals:**

1. **Retrieval-centered** — Retrieval QA and Depth Dive are top-level, self-contained modules. When you extract Retrieval QA into its own repo later (post-MVP), the `git subtree split` is clean.
2. **Shared core/** — only truly shared things (types, API clients, config, DB) live here. No false "shared" abstractions you don't need yet.
3. **No generic agent cargo** — no planner, no executor, no memory abstraction that doesn't apply to RAG. Retrieval is deterministic; it doesn't need those patterns.
4. **Tool-specificity** — web search lives _inside_ Depth Dive, not a generic tool, making it clear it's a Depth Dive-specific capability.
5. **Clean import boundaries** — matches ADR-0011's import-linter rules exactly (retrieval_qa ↔ core, depth_dive ↔ core, never retrieval_qa ↔ depth_dive).
6. **Extractable ingestion** — ingestion logic is modular enough that when you graduate to `arq` + Redis (ADR-0006), you can slot it in without restructuring.
7. **Eval/tuning kept outside packages** — `scripts/` tooling plus `eval_corpus/` material stay out of runtime package dirs, and the eval tests live at root `tests/scripts/`, so per-package CI stays clean.

**Where to add things** — the `AGENTS.md` file at the root now documents exactly where things go (the "Where to add things" section). Use that as the canonical reference for future contributors.
