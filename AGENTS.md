# AGENTS.md
> Compact operating notes for OpenCode sessions in this repo.

## Current state

This repository is in **early implementation (tracer bullet complete)**. ADRs, stack decisions, coding standards, and the monorepo layout are in place, plus the build/CI scaffolding: root + per-module `pyproject.toml` (uv workspace), `uv.lock`, `.github/workflows/` (`ci.yml`, `cd.yml`, `security.yml`, `dependabot-auto-merge.yml`), `commitlint.config.mjs`, and `Dockerfile`. Four of five packages have real implementation code (~2300 lines total); only `depth_dive/` still has just placeholders. The toolchain is green: `uv sync` installs all deps; `uv run ruff check`, `uv run ruff format --check`, `uv run mypy`, `uv run lint-imports`, and `uv run pytest` all pass.

## Authoritative sources — read before acting

These docs are **not** auto-loaded into session context. Only this `AGENTS.md` is. You must explicitly Read the relevant file(s) with the Read tool before acting on a task. The digests below are a quick reference, not a substitute for the source.

**Before any implementation task** (writing, editing, or reviewing code):
- `CONTEXT.md` — domain glossary (Harness A/B, Depth Dive, Captured Passage, Injected Context, etc.). Use the exact domain terms in code, docstrings, and tests.
- `docs/coding-standards.md` — typing, docstrings, testing, and error-handling rules. Apply every rule; do not rely on the summary in this file.

**Before touching package structure or inter-package dependencies:**
- `docs/ai-system-tree.md` — intended monorepo layout and "where to add things" map.
- `docs/adr/0005-structured-monorepo.md` — why the structure is what it is.
- `docs/adr/0011-import-linter-module-boundaries.md` — the enforced boundary rules.

**Before choosing or adding a library, model, or external service:**
- `docs/tech-stack.md` — MVP stack and staged post-MVP milestones.
- `docs/adr/` — read the ADR(s) most relevant to the area (e.g., ADR-0001 inference, ADR-0002 pgvector, ADR-0003 hand-rolled RAG, ADR-0004 embeddings, ADR-0006 background ingestion). Treat ADRs as **constraints, not suggestions**. If a decision contradicts an ADR, stop and propose a new/updated ADR rather than silently deviating.

**Before writing a commit message:**
- `docs/commit-instructions.md` — Conventional Commits format and allowed types.

**Before adding a new ADR:**
- Read all existing `docs/adr/*.md` first to avoid contradicting a prior decision. Number the new ADR sequentially.

## Architecture overview

`README.md` § Architecture has the canonical package table (roles at a glance). `docs/ai-system-tree.md` has the full intended internal layout (`chunking/`, `retrieval/`, `generation/`, etc. — created alongside first implementation, not pre-seeded). Each package has its own `pyproject.toml`, `src/<name>/`, and `tests/<name>/` (inter-package deps declared as uv workspace sources).

## Constraints that are already decided

- **No LangChain / LlamaIndex.** The RAG pipeline is hand-rolled (ADR-0003). Build chunking, embedding, retrieval, and prompt assembly against raw client libraries.
- **MVP uses hosted APIs only:**
  - Embeddings: Google `text-embedding-004` or OpenAI `text-embedding-3-small` (ADR-0004).
  - Inference: Claude/OpenAI (ADR-0001).
- **Database:** PostgreSQL + `pgvector` extension (ADR-0002). Qdrant is deferred.
- **Background ingestion:** FastAPI `BackgroundTasks` (ADR-0006). No Redis/Celery/arq in MVP.
- **Module boundaries:** `retrieval_qa` and `depth_dive` may depend on `core`, but **never on each other** (ADR-0011). Enforced via `import-linter` in `pr-checks.yml` (contracts declared in the root `pyproject.toml`).
- **Commits:** Conventional Commits, enforced in CI (ADR-0010). Allowed types: `feat`, `fix`, `refactor`, `test`, `docs`, `chore`, `ci`, `perf`. No local pre-commit hook yet.

## Coding standards to apply now

- `mypy --strict` is configured in the root `pyproject.toml` (with `mypy_path` + `explicit_package_bases` for the src layout); enforced in `pr-checks.yml`.
- All public functions/classes use Google-style docstrings.
- Pydantic v2 models are the boundary type for all I/O.
- Use named exceptions (`IngestionError`, `RetrievalError`, etc.), not bare `Exception`.
- Retrieval QA's "not found" case is a valid response (`grounded=False`), not an exception.
- `ruff` handles linting and formatting; config lives in the root `pyproject.toml`.

## Testing conventions

- Mirror package structure: `retrieval_qa/retrieval.py` → `tests/retrieval_qa/test_retrieval.py`.
- Mock hosted API calls (embeddings, LLM, web search) in unit tests.
- Keep retrieval-relevant tests identifiable (consistent module naming) so they can be path-gated in CI later.

## Where to add things

- **New chunking strategy for a doc type** → `retrieval_qa/src/retrieval_qa/chunking/`
- **Bug fix in retrieval logic** → `retrieval_qa/src/retrieval_qa/retrieval/`
- **New Depth Dive generation feature** → `depth_dive/src/depth_dive/generation/`
- **Web search improvements** → `depth_dive/src/depth_dive/web_search/`
- **New shared type** → `core/src/core/types/`
- **New API endpoint** → `api/src/api/routes/`
- **New ADR** → `docs/adr/`

## Mandatory verification before completing any task

Every code change — even a one-line edit — must pass the relevant checks before you declare the task done.

**First step — always sync the workspace:**

```bash
uv sync --all-packages
```

Editable installs (`.pth` files) only discover modules that exist at sync time. Any new source file, new module, or changed dependency requires a re-sync before imports resolve. This is the single most common cause of `ModuleNotFoundError` during local development, and it wastes tokens when the agent retries instead of syncing.

Then run the relevant checks from the repo root with `uv run`:

| Check | When to run | Command |
|-------|-------------|---------|
| Ruff lint | **Always** — after any code change | `uv run ruff check .` |
| Ruff format | **Always** — after any code change | `uv run ruff format --check .` |
| Mypy | **Always** — after any code change | `uv run mypy .` |
| Pytest | When tests are added/modified, or when production code with existing tests changes | `uv run pytest` |
| Import-linter | When inter-package imports change (new modules, new cross-package deps) | `uv run lint-imports` |

Rules:
- All required checks must **pass** before reporting a task complete. If a check fails, fix it — do not hand off a broken state.
- If you add a new source file or module, run `uv sync --all-packages` before attempting to import or test it.
- If you add a new runtime dependency to a member `pyproject.toml`, run `uv sync --all-packages` first to update the lockfile, then run all checks.
- If `ruff format --check` fails, run `uv run ruff format .` to fix, then re-verify.
- Do not add `# type: ignore` or `noqa` comments to silence errors without justification in a preceding line or PR description.

## What does not exist yet

- **Retrieval QA query logic** — `retrieval_qa/src/retrieval_qa/retrieval/` is not yet created; there is no `POST /query` endpoint or similarity-search pipeline.
- **Depth Dive implementation** — `depth_dive/src/depth_dive/generation/` and `web_search/` are still placeholders.
- **`api/routes/__init__.py`** — route modules are imported directly in `server.py`; no package init file exists.

Bootstrap order is already complete for the first four packages (`core/` → `retrieval_qa/` → `api/` → `ingestion/`). `depth_dive/` is the remaining package to implement.
