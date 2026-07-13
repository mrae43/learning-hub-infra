# commit-instructions.md
> Commit message conventions for the Learning Hub project. Enforced in CI (see ADR-0010) so `release.yml`'s automated changelog stays trustworthy.

## Format

```
<type>(<optional scope>): <short imperative summary>

<optional body>

<optional footer>
```

## Allowed types

| Type | When to use |
|---|---|
| `feat` | A new user-facing capability (e.g. a new endpoint, a new Harness A/B behavior) |
| `fix` | A bug fix |
| `refactor` | Code change that doesn't alter behavior (restructuring, renaming, extracting) |
| `test` | Adding or updating tests only |
| `docs` | Documentation only (`CONTEXT.md`, ADRs, this file, README, etc.) |
| `chore` | Tooling, dependency bumps, config changes with no behavior impact |
| `ci` | Changes to GitHub Actions workflows |
| `perf` | Performance improvement with no behavior change |

Breaking changes: add `!` after the type/scope (e.g. `feat(retrieval-qa)!: change response schema`) and explain the break in the footer as `BREAKING CHANGE: ...`.

## Git hygiene rules

- **Subject line**: imperative mood ("add retrieval endpoint," not "added" or "adds"), no period at the end, under ~50 characters where practical.
- **Body** (when needed): wrap at ~72 characters, explain *why* the change was made, not just what — the diff already shows what changed.
- **Scope** (optional): the module the change touches, matching the monorepo's structure (e.g. `retrieval-qa`, `depth-dive`, `docs`, `ci`).
- **One logical change per commit** — don't bundle an unrelated `fix` and `feat` into one commit just because they touched nearby code.

## Examples

```
feat(retrieval-qa): add recall@k retrieval evaluation job

fix(retrieval-qa): correct chunk boundary off-by-one in paper chunker

docs: describe structured groundedness response schema in coding standards

chore: bump fastapi to 0.115

feat(retrieval-qa)!: change HarnessAResponse to include cited_passage_ids

BREAKING CHANGE: response schema now requires cited_passage_ids and
grounded fields. Any consumer relying on the previous bare-string
response must be updated.
```

## Enforcement

- **CI-only for now** (see ADR-0010): a GitHub Actions step lints commit messages on every PR against the Conventional Commits format. Non-conforming commits fail the check, blocking merge — consistent with the project's existing "merge blocked on any failing job" pattern.
- **No local pre-commit hook yet.** This project currently has a single developer, so a hook's fastest-feedback benefit doesn't outweigh the setup/maintenance cost. Add a pre-commit hook (`commitlint` + a git hook manager) once the project grows past solo development — e.g. contributors joining, or CI feedback loop feeling too slow for comfortable iteration.