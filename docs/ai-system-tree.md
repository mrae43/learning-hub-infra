# Codebase Structure

```
learning-hub/
├── retrieval_qa/                        # Retrieval QA (extractable later)
│   ├── src/retrieval_qa/
│   │   ├── chunking/                 # Document-type-specific chunkers
│   │   │   ├── paper_chunker.py
│   │   │   ├── book_chunker.py
│   │   │   └── documentation_chunker.py
│   │   ├── retrieval/                # Retrieve from pgvector
│   │   │   ├── query.py
│   │   │   └── ranking.py
│   │   └── __init__.py
│   ├── tests/retrieval_qa/
│   ├── pyproject.toml
│   └── README.md
├── depth_dive/                        # Depth Dive generation (extractable later)
│   ├── src/depth_dive/
│   │   ├── generation/               # Depth Dive response generation
│   │   │   └── generator.py
│   │   ├── web_search/               # Web search tool logic
│   │   │   ├── search_tool.py
│   │   │   └── citation.py
│   │   └── __init__.py
│   ├── tests/depth_dive/
│   ├── pyproject.toml
│   └── README.md
├── core/                             # Shared (may stay here or move to common/ later)
│   ├── src/core/
│   │   ├── types/                    # Shared schemas
│   │   │   ├── document.py
│   │   │   ├── chunk.py
│   │   │   ├── responses.py          # HarnessAResponse, etc.
│   │   │   └── __init__.py
│   │   ├── config/
│   │   │   ├── settings.py           # Pydantic settings
│   │   │   └── __init__.py
│   │   ├── clients/                  # API clients (hosted inference, embeddings)
│   │   │   ├── llm_client.py
│   │   │   ├── embeddings_client.py
│   │   │   └── web_search_client.py
│   │   ├── database/                 # pgvector wrapper, Alembic migrations
│   │   │   ├── connection.py
│   │   │   ├── schema.py
│   │   │   └── migrations/
│   │   └── __init__.py
│   ├── tests/core/
│   ├── pyproject.toml
│   └── README.md
├── api/                              # FastAPI server (thin controller layer)
│   ├── src/api/
│   │   ├── routes/
│   │   │   ├── retrieval_qa.py          # /query endpoint
│   │   │   └── depth_dive.py          # /depth-dive endpoint
│   │   ├── controllers/
│   │   │   ├── qa_controller.py      # Orchestrates Harness A
│   │   │   └── depth_dive_controller.py  # Orchestrates Harness B
│   │   ├── server.py                 # FastAPI app factory
│   │   ├── main.py                   # Entry point
│   │   └── __init__.py
│   ├── tests/api/
│   ├── pyproject.toml
│   └── README.md
├── ingestion/                        # Document upload & background task logic
│   ├── src/ingestion/
│   │   ├── tasks.py                  # FastAPI BackgroundTasks logic
│   │   ├── pipeline.py               # Ingest → chunk → embed → store
│   │   └── __init__.py
│   ├── tests/ingestion/
│   ├── pyproject.toml
│   └── README.md
├── docs/
│   ├── adr/                          # 0001–0015 all live here
│   ├── tech-stack.md
│   ├── coding-standards.md
│   └── commit-instructions.md
├── .github/workflows/
│   ├── ci.yml                        # ruff, mypy, pytest, import-linter, commitlint
│   ├── cd.yml                        # Docker build + changelog
│   ├── security.yml                  # Dependency scanning (pip-audit, etc.)
│   └── dependabot-auto-merge.yml     # Auto-merge for low-risk Dependabot bumps
├── pyproject.toml                    # Root: ruff config, import-linter contracts
├── .env.example
├── Dockerfile
├── AGENTS.md
├── CONTEXT.md
└── README.md
```

**Why this is better for your goals:**

1. **Harness-centered** — Harness A and B are top-level, self-contained modules. When you extract A into its own repo later (post-MVP), the `git subtree split` is clean.
2. **Shared core/** — only truly shared things (types, API clients, config, DB) live here. No false "shared" abstractions you don't need yet.
3. **No generic agent cargo** — no planner, no executor, no memory abstraction that doesn't apply to RAG. Retrieval is deterministic; it doesn't need those patterns.
4. **Tool-specificity** — web search lives *inside* Harness B, not a generic tool, making it clear it's a Harness-B-specific capability.
5. **Clean import boundaries** — matches ADR-0011's import-linter rules exactly (retrieval_qa ↔ core, depth_dive ↔ core, never retrieval_qa ↔ depth_dive).
6. **Extractable ingestion** — ingestion logic is modular enough that when you graduate to `arq` + Redis (ADR-0006), you can slot it in without restructuring.

**One addition:** Add a `CLAUDE.md` at the root (or link to it from README) that documents exactly where things go — this is what you'd hand to Claude Code or a future contributor:

```
## Where to add things

- **New chunking strategy for a doc type** → `retrieval_qa/src/retrieval_qa/chunking/`
- **Bug fix in retrieval logic** → `retrieval_qa/src/retrieval_qa/retrieval/`
- **New Depth Dive generation feature** → `depth_dive/src/depth_dive/generation/`
- **Web search improvements** → `depth_dive/src/depth_dive/web_search/`
- **New shared type** → `core/src/core/types/`
- **New API endpoint** → `api/src/api/routes/`
- **New ADR** → `docs/adr/`
```