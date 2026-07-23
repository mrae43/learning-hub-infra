# Codebase Structure

```
learning-hub/
├── retrieval_qa/                        # Retrieval QA (extractable later)
│   ├── src/retrieval_qa/
│   │   ├── chunking/                 # Document-type-specific chunkers
│   │   │   ├── base.py               # Base chunker class
│   │   │   ├── paper_chunker.py
│   │   │   ├── book_chunker.py
│   │   │   └── documentation_chunker.py
│   │   ├── retrieval/                # Retrieve from pgvector
│   │   │   └── query.py
│   │   └── __init__.py
│   ├── tests/retrieval_qa/           # chunker + retrieval tests
│   ├── pyproject.toml
│   └── README.md
├── depth_dive/                        # Depth Dive generation (stub — TODO)
│   ├── src/depth_dive/
│   │   └── __init__.py               # Package marker only
│   ├── tests/depth_dive/
│   │   └── test_smoke.py
│   ├── pyproject.toml
│   └── (README.md)
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
│   │   │   └── embeddings_client.py
│   │   ├── database/                 # pgvector wrapper, Alembic migrations
│   │   │   ├── connection.py
│   │   │   ├── schema.py
│   │   │   └── migrations/
│   │   ├── exceptions.py             # Named exception types
│   │   └── __init__.py
│   ├── tests/core/                   # types, clients, migration tests
│   ├── pyproject.toml
│   └── README.md
├── api/                              # FastAPI server (thin controller layer)
│   ├── src/api/
│   │   ├── routes/
│   │   │   ├── retrieval_qa.py       # /query endpoint
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
│   │   └── __init__.py
│   ├── tests/ingestion/
│   ├── pyproject.toml
│   └── README.md
├── scripts/
│   └── generate_eval_vectors.py      # Eval vector generation utility
├── docs/
│   ├── adr/                          # 0001–0017 (skip 0008; 0015 supersedes 0007 scorer)
│   ├── ai-system-tree.md
│   ├── tech-stack.md
│   ├── coding-standards.md
│   └── commit-instructions.md
├── .github/workflows/
│   ├── ci.yml                        # ruff, mypy, pytest, import-linter, commitlint
│   ├── cd.yml                        # Docker build + changelog
│   ├── security.yml                  # Dependency scanning (pip-audit, etc.)
│   └── dependabot-auto-merge.yml     # Auto-merge for low-risk Dependabot bumps
├── .out-of-scope/
│   ├── custom-deepeval-metric-async.md   # Deepeval a_measure pattern rationale
│   ├── docker-compose.md                 # Notes on docker-compose scoping decision
│   └── pytest-module-level-test-data.md  # Module-level test data I/O rationale
├── pyproject.toml                    # Root: uv workspace, ruff config, import-linter contracts
├── conftest.py                       # Root test fixtures (271 lines, shared by all packages)
├── alembic.ini                       # Alembic configuration for DB migrations
├── commitlint.config.mjs             # Conventional Commits enforcement
├── docker-compose.yml                # Local dev: PostgreSQL + pgvector
├── Dockerfile                        # Multi-stage build (all 5 packages)
├── .env.example
├── AGENTS.md                         # Session notes for AI coding tools
├── CONTEXT.md                        # Domain glossary
└── README.md
```

**Why this is better for your goals:**

1. **Harness-centered** — Harness A and B are top-level, self-contained modules. When you extract A into its own repo later (post-MVP), the `git subtree split` is clean.
2. **Shared core/** — only truly shared things (types, API clients, config, DB) live here. No false "shared" abstractions you don't need yet.
3. **No generic agent cargo** — no planner, no executor, no memory abstraction that doesn't apply to RAG. Retrieval is deterministic; it doesn't need those patterns.
4. **Tool-specificity** — web search lives *inside* Harness B, not a generic tool, making it clear it's a Harness-B-specific capability.
5. **Clean import boundaries** — matches ADR-0011's import-linter rules exactly (retrieval_qa ↔ core, depth_dive ↔ core, never retrieval_qa ↔ depth_dive).
6. **Extractable ingestion** — ingestion logic is modular enough that when you graduate to `arq` + Redis (ADR-0006), you can slot it in without restructuring.

**Where to add things** — the `AGENTS.md` file at the root now documents exactly where things go (the "Where to add things" section). Use that as the canonical reference for future contributors.