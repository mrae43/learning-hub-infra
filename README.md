# Learning Hub

A RAG-powered study tool that turns uploaded papers, books, and documentation into interactive learning sessions — retrieval QA (Harness A) and dual-coding Depth Dives (Harness B).

> **Status:** Early implementation — tracer bullet complete. Four of five packages (`core/`, `retrieval_qa/`, `api/`, `ingestion/`) have implementation code (~3650 lines total). Only `depth_dive/` remains as a scaffold. Ingestion pipeline (upload PDF → chunk → embed → store in pgvector with HNSW index), Harness A query pipeline (`POST /query`), and three API endpoints (`POST /ingest`, `GET /documents/{id}`, `POST /query`) are operational. See [docs/](./docs/) for architecture decisions and plans.

## Architecture

Structured monorepo with extractable module boundaries:

| Package | Role |
|---|---|
| `core/` | Shared types, config, API clients, database (`pgvector`) |
| `retrieval_qa/` | Closed-corpus retrieval QA (no web search) |
| `depth_dive/` | Depth Dive generation (web search allowed) |
| `api/` | FastAPI server (thin routes → controllers) |
| `ingestion/` | Document upload + background ingestion pipeline |

Hand-rolled RAG pipeline — no LangChain/LlamaIndex. PostgreSQL + `pgvector` for storage and retrieval. Hosted APIs for embeddings and inference in MVP.

## Key docs

- [CONTEXT.md](./CONTEXT.md) — domain glossary
- [docs/tech-stack.md](./docs/tech-stack.md) — MVP stack and post-MVP milestones
- [docs/ai-system-tree.md](./docs/ai-system-tree.md) — full directory layout
- [docs/adr/](./docs/adr/) — architecture decision records (project constraints)
- [docs/coding-standards.md](./docs/coding-standards.md) — typing, docstrings, testing, error handling
- [docs/commit-instructions.md](./docs/commit-instructions.md) — Conventional Commits format
- [AGENTS.md](./AGENTS.md) — session notes for AI coding tools
