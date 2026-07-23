# Docker Compose and Dockerfile

This project does not provide Docker Compose files or production Dockerfiles for local development or deployment.

## Why this is out of scope

ADR-0014 explicitly states "no compose file" as a premise (L46). The project's CI/CD pipeline uses GitHub Actions for testing and relies on a managed PostgreSQL service (Supabase) with pgvector for the database layer. Docker Compose was never part of the schema/contract scope for either the ingestion or query tracer bullets.

Adding Docker Compose would introduce a secondary, unmanaged development path that:
- Duplicates the CI workflow's database setup (which uses service containers)
- Creates a maintenance burden for keeping Compose files in sync with the actual deployment target
- Implies a local-first development model that conflicts with the project's Supabase-hosted architecture

## Prior requests

- #19 — `docker-compose.yml` + `Dockerfile` present in the ingestion tracer bullet PR (PR #7), flagged as scope creep in code review
