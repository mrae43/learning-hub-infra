# Learning Hub

A hand-rolled RAG study tool for learning AI/ML from papers, books, and documentation through grounded interactive Q&A.

## What problem

**For the learner.** Reading AI/ML papers, books, and documentation alone is slow. This tool lets you query your personal corpus interactively and get answers grounded in your sources — building toward richer, multi-modal learning sessions (Depth Dive) that accelerate understanding through neuroplasticity triggers.

**For me as the builder.** Hand-rolling the RAG pipeline — chunking, embedding, retrieval, prompt assembly — against raw client libraries is the only way to deeply understand every mechanism. That understanding is the foundation for the Depth Dive features. Using LangChain/LlamaIndex would hide the mechanics I'm here to learn.

## How it works

**Upload & ingest.** Submit a PDF or EPUB via `POST /ingest`. The server validates the file, chunks it by document type (paper, book, documentation), embeds each chunk via OpenAI `text-embedding-3-small`, and stores it in a pgvector HNSW index. You get a document ID immediately; ingestion continues in the background. Poll progress with `GET /documents/{id}`.

**Query (MVP).** Send `{query: str}` to `POST /query`. The system retrieves the top-k relevant chunks from your entire corpus, assembles a prompt with those chunks as context, and calls a hosted LLM. The response is a structured `HarnessAResponse` — answer text, cited passages, and a `grounded: bool` flag so you know whether the answer actually came from your documents.

**Roadmap.**
| Phase | What | Status |
|---|---|---|
| MVP | Grounded Q&A against your personal corpus | ✅ Operational |
| Post-MVP 1 | **Depth Dive** — richer explanations (text + diagrams + code) with agentic web search | 🔧 Scaffold |
| Post-MVP 2 | **Synapse** — multi-sensory interactive learning with gamification and neuroplasticity triggers | 📅 Planned |

> **Status:** Early implementation — tracer bullet complete. Four of five packages (`core/`, `retrieval_qa/`, `api/`, `ingestion/`) have implementation code (~5760 lines total). Only `depth_dive/` remains as a scaffold. Ingestion pipeline, Harness A query pipeline, and three API endpoints (`POST /ingest`, `GET /documents/{id}`, `POST /query`) are operational. See [docs/](./docs/) for architecture decisions and plans.

## Architecture

Structured monorepo with extractable module boundaries:

| Package | Role |
|---|---|
| `core/` | Shared types, config, API clients, database (`pgvector`) |
| `retrieval_qa/` | Closed-corpus retrieval QA (MVP) |
| `depth_dive/` | Depth Dive generation (post-MVP 1) |
| `api/` | FastAPI server (thin routes → controllers) |
| `ingestion/` | Document upload + background ingestion pipeline |

## Why this tech stack

**Python + uv + FastAPI + Pydantic v2.** Python is the AI/ML ecosystem's lingua franca. uv is fast and natively supports the monorepo workspace structure. FastAPI is async-native, matching the I/O-bound calls to hosted embedding/inference APIs. Pydantic v2 pairs with FastAPI for typed request/response contracts.

**PostgreSQL + pgvector.** One production-proven database for app data and vectors — one fewer service to operate in MVP. Migration to Qdrant is deferred until pgvector's limits are concretely felt ([ADR-0002](./docs/adr/0002-pgvector-for-mvp.md)).

**Hand-rolled RAG (no LangChain/LlamaIndex).** Retrieval mechanics are what I'm here to build and understand. A framework would hide chunking strategy, similarity scoring, and prompt assembly behind abstractions. This is the core pedagogical choice ([ADR-0003](./docs/adr/0003-handroll-rag-pipeline.md)).

**Hosted APIs for embeddings and inference.** My GPU (4GB VRAM) makes local embedding slow and conflates retrieval quality with GPU tuning. Hosted APIs let me judge retrieval in isolation. Self-hosted vLLM is deferred ([ADR-0001](./docs/adr/0001-hosted-inference-api-for-mvp.md), [ADR-0004](./docs/adr/0004-hosted-embedding-api.md)).

**FastAPI BackgroundTasks for ingestion.** No lost jobs yet — a dedicated queue (arq/Redis) gets added when that becomes a real pain point ([ADR-0006](./docs/adr/0006-backgroundtask-for-mvp-ingestion.md)).

**Structured monorepo.** Five packages with independent `pyproject.toml` files, test suites, and CI jobs — extractable later via `git subtree split` ([ADR-0005](./docs/adr/0005-structured-monorepo.md)). Module boundaries enforced by `import-linter` in CI ([ADR-0011](./docs/adr/0011-import-linter-module-boundaries.md)).

## Challenges solved

**Multi-format document chunking.** Papers, books, and documentation have different structures. Each gets a structure-aware chunker (section boundaries for papers, chapter boundaries for books, page/API-entry boundaries for documentation) that produces typed metadata via a JSONB registry — keeping each doc type's schema explicit and extensible.

**Module boundary enforcement.** A five-package monorepo needs real boundaries or they become fictional. `import-linter` in CI (`uv run lint-imports`) catches cross-package leaks before merge — e.g., `retrieval_qa` and `depth_dive` must never import each other.

**Document state machine.** Ingestion is async: validating → chunking → embedding → ready (or failed). The state is tracked in the database and pollable via `GET /documents/{id}`, so the user always knows where their upload stands.

**Structured response with groundedness.** Every query response is a typed `HarnessAResponse` with `answer`, `cited_passages`, and a `grounded: bool` flag. Clients can distinguish grounded answers (with sources) from ungrounded ones without text parsing — essential for trust in a learning tool.

## Key docs

- [CONTEXT.md](./CONTEXT.md) — domain glossary
- [docs/tech-stack.md](./docs/tech-stack.md) — MVP stack and post-MVP milestones
- [docs/ai-system-tree.md](./docs/ai-system-tree.md) — full directory layout
- [docs/adr/](./docs/adr/) — architecture decision records (project constraints)
- [docs/coding-standards.md](./docs/coding-standards.md) — typing, docstrings, testing, error handling
- [docs/commit-instructions.md](./docs/commit-instructions.md) — Conventional Commits format
- [AGENTS.md](./AGENTS.md) — session notes for AI coding tools
