# 0018 — No Authentication or Access Control for MVP; Deferred to Its Own Milestone

## Status
Accepted

## Context
Learning Hub currently runs local-only via Docker, single user, with no network exposure beyond the local Docker network. Every route (`POST /ingest`, `GET /documents/{id}`, `POST /query`, `GET /health`) is open — no auth middleware, no login endpoint, no JWT, no API-key gating on requests. The only API keys in the system (OpenAI, Cohere) authenticate *outbound* calls to external services, not *inbound* requests. This gap is not currently recorded in any ADR.

Two options were considered:
- Build authentication, per-route access control, and row-level security now, matching what any internet-facing service would eventually need.
- Defer all of it, since there is currently no identity or tenancy boundary for any of these mechanisms to enforce — MVP has exactly one user who already has full filesystem and Docker access to the machine it runs on.

## Decision
MVP infra will be strictly needs-driven, per the same principle governing ADR-0001/0002/0004/0006 — the leanest stack that lets the current deployment function, not deliberately inclusive of controls it doesn't yet need. No authentication, per-route access control, row-level security, or request-level rate limiting will be built for MVP. Auth is not free — it touches every route, needs a session/token strategy, and would likely become its own concern inside `core/` or `api/` (ADR-0005). Building it before a real identity boundary exists is premature relative to the actual threat model, and would slow MVP velocity for a risk that doesn't yet exist.

This decision governs identity and access boundaries only. Baseline request hygiene — input validation, upload validation, parameterized queries, secret handling — is still required regardless of deployment topology and is not in scope here; see `docs/security-mvp-guide.md`.

**Graduation triggers** (any one ends this ADR's scope):
- The service becomes reachable from outside the local Docker network (bound to a non-loopback interface, tunneled, deployed to any host, or exposed via ngrok/Cloudflare Tunnel/etc.)
- A second user or account is introduced, in any form
- Any endpoint is exposed to the public internet, directly or indirectly

## Consequences
- Retrofitting auth later is a real, non-trivial cut-over: an auth/session strategy must be chosen and a dependency-injection guard added to every route in `api/`. This should be planned as an explicit milestone with its own scope, not a drop-in swap — and should be recorded as a new ADR that supersedes this one (matching the ADR-0007 → ADR-0015 pattern), not a silent edit to this document.
- Row-level security and per-route access control have distinct triggers, not a shared one: RLS's trigger is multi-tenancy (a second user), not mere internet exposure. A solo user reachable over the internet still doesn't need RLS — worth keeping these two triggers distinct when a trigger fires, since they don't imply the same scope of work.
- Until a trigger fires, this is a deliberate sequencing choice, not an oversight — worth remembering if a future reader wonders why an internet-adjacent API has no auth at all.