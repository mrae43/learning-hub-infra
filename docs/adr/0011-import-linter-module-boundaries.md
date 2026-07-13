# 0011 — import-linter to Enforce Monorepo Module Boundaries

## Status
Accepted

## Context
ADR-0005 established a structured monorepo where each component (Retrieval QA, Depth Dive, shared core) is packaged as an independently extractable module, but explicitly flagged that this property depends on boundaries being genuinely maintained: "module boundaries must be genuinely maintained... or the 'extractable' property becomes fictional." No enforcement mechanism was specified at the time.

Without automated enforcement, nothing stops a convenient cross-module import (e.g. Depth Dive reaching directly into Retrieval QA's internals to save time) from silently accumulating, at which point the "extract into its own repo later" plan in ADR-0005 becomes false — the modules are coupled in practice even if the folder structure suggests otherwise.

## Decision
Use `import-linter` as a CI-enforced check. Import contracts are declared explicitly (e.g. "retrieval-qa must not import from depth-dive," "both may depend on a shared `core` package, but not on each other's internals") and a CI job fails the build if a commit violates them — following the same "merge blocked on failing job" pattern already established for lint/type/test checks.

## Consequences
- ADR-0005's "extractable module" premise is now actually enforced, not just aspirational — a violation is caught in CI before merge, not discovered later when attempting extraction.
- Import contracts must be declared and kept up to date as the module structure evolves (e.g. when a new shared utility is introduced, decide which side of the boundary it lives on).
- This closes the specific gap ADR-0005 flagged; it does not address every form of coupling (e.g. two modules could still be implicitly coupled through shared database schema or API contracts) — only import-level boundaries.