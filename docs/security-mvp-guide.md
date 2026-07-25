# security-mvp-guide.md
> Source of truth for implementing security hygiene in Learning Hub's MVP. Read this before touching `core/`, `api/`, `ingestion/`, or `retrieval_qa/` for anything security-adjacent. See [ADR-0018](./adr/0018-no-auth-access-control-for-mvp.md) for what is *deliberately out of scope* — read that first so you don't build things this project doesn't need yet.

## Scope

This guide covers request-level and data-level hygiene that applies **regardless of deployment topology** — it's required whether Learning Hub stays local-Docker-single-user or eventually goes multi-user on the internet. It does not cover identity, sessions, or access control — that's ADR-0018's territory, and it's explicitly deferred until one of its graduation triggers fires. Do not add auth middleware, login routes, JWTs, or API-key request gating unless a human has told you a trigger has fired.

If you're an agent picking up a ticket and it smells like "add login" or "protect this route" — stop and check whether ADR-0018 has been superseded before writing any code.

---

## 1. Secrets and environment variables

**Requirement.** `OPENAI_API_KEY`, `COHERE_API_KEY`, `DATABASE_URL`, and any future provider key live in environment variables, loaded via Pydantic Settings (`core/config.py` or equivalent) — never hardcoded, never committed.

- `.env` must be in `.gitignore`. Verify this hasn't regressed before merging any PR that touches config.
- Never log a full settings object or request object that could contain a key — if you add logging middleware, explicitly exclude/redact secret fields.
- `.env.example` (committed, no real values) should stay in sync with what `core/config.py` actually reads — a coding agent adding a new setting must add the placeholder too.
- Exceptions and error responses must never include raw settings or environment dumps. FastAPI's default exception handlers are fine; custom exception handlers must not echo `request.app.state` or config objects.

## 2. Input validation

**Requirement.** Every request boundary is validated by a Pydantic v2 model — this is mostly already true by construction (ADR-0014), but it only covers *shape*, not *content*.

- Shape validation (types, required fields, enums) — Pydantic handles this. Don't hand-roll manual `if` checks for things a Pydantic field constraint (`Field(min_length=...)`, `Annotated` types, custom validators) can express instead.
- Content validation Pydantic *doesn't* give you for free, and that still needs explicit code:
  - Query strings to `/query` — enforce a max length (protects against pathological prompt-assembly costs, not just "attacks").
  - Any free-text field that gets interpolated into a prompt template — treat it as untrusted even though it's your own data; a malformed or adversarial chunk shouldn't be able to break prompt structure.
- Validation errors return FastAPI's standard 422 with the Pydantic error detail — don't swallow validation errors into a generic 500, and don't leak internal file paths or stack traces in the error body.

## 3. File upload validation (`POST /ingest`)

**Requirement.** The README states "the server validates the file" — this section is what that needs to actually mean.

- **Extension is not validation.** Check the file's magic bytes / actual content type (e.g. `python-magic` or manual header sniffing for PDF `%PDF-` / EPUB's zip signature), not just the filename suffix or client-supplied `Content-Type` header, which is trivially spoofable.
- **Size cap enforced before the file is fully buffered into memory**, not after — stream and reject early, or use FastAPI's `UploadFile` with a size check against `Content-Length` up front. A single unbounded upload shouldn't be able to fill disk or memory on a dev machine.
- **Reject before the Document-Type Chunker runs**, not during it — validation failures should short-circuit at the `/ingest` route, producing a clear `failed` state (per the document state machine in `CONTEXT.md`), not surface as a chunker exception mid-pipeline.
- PDF/EPUB parsing libraries have real CVE history around malformed files (infinite loops, memory exhaustion, zip bombs for EPUB). If the ingestion pipeline doesn't already run parsing with a timeout or resource ceiling, that's a gap worth flagging even at MVP scale — a hostile or corrupted file shouldn't be able to hang the background task worker.

## 4. Parameterized queries

**Requirement.** No raw string interpolation into SQL, anywhere, ever — including inside f-strings that "just build a WHERE clause."

- SQLAlchemy ORM queries and Core `select()`/`text()` with bound parameters (`:param` + `.params()`) are both fine. String-formatted SQL is not, even for internal/trusted-looking values like a document ID.
- **Specifically audit the hybrid search path from ADR-0016** (tsvector full-text query + RRF fusion). Full-text search queries are the most common place hand-rolled SQL creeps in, because `to_tsquery()` syntax doesn't map cleanly onto ORM query builders. If that code constructs a `tsquery` string by concatenating user input, that's a real injection surface — use bound parameters or a query-sanitizing helper (e.g. `plainto_tsquery` with a bound param) instead of building the query string manually.
- Same audit applies to the pgvector HNSW cosine search path (ADR-0002) if any part of it constructs raw SQL for the vector similarity clause.

## 5. Cost guardrails on hosted API calls

**Requirement.** Not a classic "security" item, but the closest MVP equivalent to rate limiting given ADR-0018's local-single-user scope — the real risk right now is a bug burning API spend, not an attacker.

- `LLMClient` (ADR-0001) and the embedding client (ADR-0004) should have a request timeout and a bounded retry count — no unbounded retry loops against OpenAI/Cohere.
- `POST /query`'s top-k retrieval count should be bounded server-side (not just client-suppliable), so a malformed or malicious request body can't force a retrieval + generation call against an unreasonably large context.
- If `BackgroundTasks` ingestion (ADR-0006) doesn't already have a concurrency cap on outbound embedding calls, that's worth confirming — ADR-0006 mentions a semaphore-capped pipeline is implemented, so this may already be handled; verify rather than assume.

## 6. Database privilege hygiene

**Requirement.** Low-cost, do opportunistically when already touching migrations — not urgent given Postgres is only reachable inside the Docker network today, but cheap to get right now versus retrofitting later.

- Don't rely on Postgres's default `PUBLIC` grants. If the Alembic migrations don't already do it, the app's DB role should have only the privileges it needs (no superuser, no `CREATEDB`) even in local dev — matching prod behavior early avoids a "works locally, breaks in prod" surprise later.
- This is *not* row-level security (that's explicitly deferred per ADR-0018's graduation triggers) — this is instance-level privilege scoping, which is a different, cheaper concern.

---

## What NOT to build right now

Per ADR-0018, do not implement any of the following unless a human confirms a graduation trigger has fired:

- Login/session endpoints, JWT issuance or validation
- Per-route `Depends()` auth guards
- Row-level security policies
- Traffic-shaping rate limiting (e.g. `slowapi`, Redis-backed request counters)
- Multi-tenancy scoping on any table

If a task seems to require one of these to be "done properly," that's a signal to check in with a human before proceeding, not to build it defensively.

---

## Related docs

- [ADR-0018](./adr/0018-no-auth-access-control-for-mvp.md) — why auth/access control is out of scope, and what would change that
- [ADR-0001](./adr/0001-hosted-inference-api-for-mvp.md), [ADR-0004](./adr/0004-hosted-embedding-api.md) — hosted API clients this guide's cost guardrails apply to
- [ADR-0002](./adr/0002-pgvector-for-mvp.md), [ADR-0016](./adr/0016-parent-child-chunking-hybrid-search-reranker.md) — the query paths this guide's parameterization section applies to
- `docs/coding-standards.md` — general typing/testing/error-handling conventions this guide assumes