# 0005 — Structured Monorepo with Extractable Module Boundaries

## Status
Accepted

## Context
This project is explicitly a portfolio piece meant to signal AI/ML infrastructure engineering skill, not just a working app. Repository structure itself is part of that signal — a polyrepo layout (separate repos per architectural layer, e.g. core retrieval logic vs. serving infra vs. deployment configs) reads as evidence of understanding real system boundaries. But polyrepo coordination overhead (cross-repo PRs, dependency versioning, multiple CI configs) works against MVP iteration speed, at a point where Harness A's retrieval quality is still unvalidated.

## Decision
Start with a single repository, structured so each major component (Harness A, Harness B, docs) is organized as an independently packaged module (its own `pyproject.toml`/`setup.py`, its own test suite, its own CI job within a shared workflow file) even though physically co-located. Extract components into separate repositories later, once a component represents a genuinely distinct infrastructure concern (e.g. self-hosted vLLM serving, post-MVP) rather than splitting preemptively.

## Consequences
- MVP retains fast iteration: one repo, atomic commits across the whole system, no cross-repo coordination while retrieval logic is still being proven.
- Module boundaries are real from day one (separate packages, separate test suites, per-module CI checks) — not just folders, so extraction later is a mechanical `git subtree split`, not an untangling exercise.
- The portfolio narrative benefits from showing an explicit evolution (monorepo → extracted service) rather than starting polyrepo and never explaining why boundaries were drawn where they were.
- Risk: module boundaries must be genuinely maintained (no cross-module imports that bypass the package structure) or the "extractable" property becomes fictional. This should be enforced via CI or a documented coding standard, not left to discipline alone.