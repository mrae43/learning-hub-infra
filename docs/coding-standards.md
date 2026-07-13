# coding-standards.md
> Code-level conventions for the Learning Hub project. See `tech-stack.md` for tooling choices, `docs/adr/` for reasoning, and `CONTEXT.md` for domain terminology.

## Guiding principle
This project hand-rolls its RAG pipeline specifically so retrieval mechanics stay transparent (ADR-0003) — not hidden behind a framework. Coding standards follow the same philosophy: explicit over implicit, typed over dynamic, nothing "just working" without a clear reason why.

---

## Typing

- **`mypy --strict`** — every function has typed parameters and return values. No implicit `Any`. This is enforced in CI (see `tech-stack.md`), not just a style suggestion.
- Pydantic v2 models are the boundary type for all I/O — API requests/responses, and Harness A's internal `HarnessAResponse`. Don't pass bare dicts across module boundaries where a typed model should exist.
- Prefer precise types over `str`/`dict` catch-alls (e.g. a `PassageId` type alias over a bare `str`, if it's used consistently enough to warrant one).

## Docstrings

- **Google-style** for all public functions, classes, and modules.
- Every public function needs at minimum a one-line summary. Add `Args:`/`Returns:`/`Raises:` blocks when the signature isn't self-explanatory from types alone.
- Internal/private functions (prefixed `_`) don't require docstrings unless the logic is non-obvious.

```python
def chunk_paper(document: IngestedDocument) -> list[Chunk]:
    """Split a paper into section-aware chunks.

    Args:
        document: The ingested paper, already parsed into sections.

    Returns:
        A list of chunks, one per section/subsection boundary.
    """
```

## Module boundaries

- Enforced via `import-linter` in CI (ADR-0011). Contracts are declared per module — check `import-linter`'s config before adding a new cross-module dependency.
- If Retrieval QA and Depth Dive need to share logic, it goes in a shared `core` package that both depend on — never a direct import from one harness into the other's internals.
- When adding a new shared utility, decide explicitly which side of the boundary it belongs on before writing it — don't let it default to wherever's convenient in the moment.

## Testing

- **pytest**, one test module per source module, mirroring the package structure (`retrieval_qa/retrieval.py` → `tests/retrieval_qa/test_retrieval.py`).
- Test names describe behavior, not implementation: `test_recall_at_k_flags_missing_expected_passage`, not `test_eval_function`.
- Retrieval-relevant tests (chunking, embedding calls, query logic) are the ones most likely to be gated by a future retrieval-evaluation CI job — keep this code path identifiable (e.g. consistent module naming) so path-based CI gating stays accurate as the codebase grows.
- Mock hosted API calls (embeddings, inference) in unit tests — real API calls belong in the retrieval eval job specifically, not scattered across the general test suite.

## Error handling

- Raise specific, named exceptions rather than bare `Exception` — e.g. `IngestionError`, `RetrievalError`, `GroundingFailure` — so callers (and CI test assertions) can distinguish failure modes.
- Harness A's contract: when the system prompt can't ground an answer in the injected context, this is a **valid response** (`grounded=False` with a "not found" answer), not an exception. Reserve exceptions for genuine failures (API errors, malformed documents), not "no good answer found."

## Linting and formatting

- **ruff** handles both linting and formatting — no separate `black`/`isort` needed.
- Line length, rule sets, and any project-specific ignores live in `pyproject.toml` at the repo root (shared config across all monorepo modules, per ADR-0005's package structure).

## Commits

- See `commit-instructions.md` for Conventional Commits format and enforcement (ADR-0010).