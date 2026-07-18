# tech-stack.md
> The stack for the Learning Hub project, split into what MVP actually uses versus what's staged for later, explicitly infra-focused milestones. See `CONTEXT.md` for terminology (Harness A/B, Depth Dive, etc.) and `docs/adr/` for the reasoning behind each decision below.

## Guiding principle
MVP infra is **needs-driven** — the leanest stack that lets Harness A function — not deliberately inclusive of target-stack pieces it doesn't yet need (ADR-0001, ADR-0002, ADR-0004, ADR-0006). Every deferred piece below has an explicit graduation trigger, not an open-ended "eventually."

---

## MVP Stack (Harness A)

| Layer | Choice | Reasoning |
|---|---|---|
| Language | Python | Non-negotiable — ecosystem lingua franca |
| Dependency management | **uv** | Fast, first-class workspace support for the monorepo's multi-package structure (ADR-0005); same tooling family as ruff |
| Web framework | **FastAPI** | Async-native, fits I/O-bound calls to hosted inference/embeddings APIs; auto-generated OpenAPI docs |
| Validation | **Pydantic v2** | Pairs natively with FastAPI; fast, typed request/response models |
| Database | **PostgreSQL** | Single "boring," production-proven database for both app data and vectors |
| Vector store | **pgvector** (Postgres extension) | MVP choice over Qdrant — see ADR-0002. Migration to Qdrant is a staged, explicit later milestone, not open-ended |
| ORM / migrations | **SQLAlchemy + Alembic** | Standard, explicit schema control; fits hand-rolled philosophy (no hidden magic) |
| RAG orchestration | **Hand-rolled** (no LangChain/LlamaIndex) | See ADR-0003 — retrieval mechanics are the thing under test; a framework would hide exactly that |
| Chunking | **Document-Type Chunker** — structure-aware, per document type (paper/book/documentation) | Separation of concerns; see `CONTEXT.md` |
| Embeddings | **Hosted API** (Google `text-embedding-004` or OpenAI `text-embedding-3-small`) | See ADR-0004 — local `sentence-transformers` deferred due to 4GB VRAM constraint (3050 Ti) |
| Inference (generation) | **Hosted API** (Claude/OpenAI) | See ADR-0001 — self-hosted vLLM deferred to its own milestone |
| Background ingestion | **FastAPI `BackgroundTasks`** | See ADR-0006 — dedicated queue (arq/RQ + Redis) deferred until a concrete need (lost job, need for status visibility) appears |
| Testing | **pytest** | Standard, module-scoped test suites per ADR-0005 |
| Linting | **ruff** | Fast, combines linting + formatting in one tool |
| Type checking | **mypy** | Static typing enforcement, pairs with Pydantic v2's typed models |
| Containerization | **Docker** | Deployment artifact; also the release workflow's main build target |
| Repo structure | **Structured monorepo**, extractable module boundaries | See ADR-0005 |

---

## CI/CD (GitHub Actions)

| Workflow | Behavior |
|---|---|
| PR checks | Runs on every push to an open PR; lint (ruff), type check (mypy), tests (pytest) per module. Merge is blocked on any failing job. |
| `cd.yml` | Triggered on merge to `main` / on tag: builds and pushes a versioned Docker image to a registry (GHCR), and generates a changelog from commit history. |

---

## Post-MVP Target Stack (staged milestones, not MVP scope)

Each row below has a stated graduation trigger — a concrete reason to build it, not a default to reach for.

| Layer | Target choice | Graduation trigger |
|---|---|---|
| Inference serving | **vLLM** (self-hosted) | Once Harness A's retrieval quality is proven against real reading sessions (ADR-0001) |
| Vector store | **Qdrant** | Once pgvector's limits are concretely felt (vector-specific tuning, independent scaling needs) rather than assumed (ADR-0002) |
| Background processing | **arq or RQ + Redis** | First lost in-flight ingestion job to a restart, or first real need for external job-status visibility (ADR-0006) |
| Concept Linking | Explicit document-to-document relationship surfacing (see `CONTEXT.md`) | Once Harness A and B are both proven; needs its own architecture (entity/concept extraction, relation layer) |
| Retrieval Practice / Spaced Repetition | Quiz-based Depth Dives, scheduled resurfacing | Once dual-coding Depth Dives (MVP scope) are proven; needs durable per-concept state (long-term memory), unlike stateless dual coding |
| IaC | **Terraform** | Once deployment target is a real cloud environment, not local Docker |
| Orchestration | **Docker + Kubernetes** (k3s/kind locally, or GKE/EKS) | Once a single Docker container / Compose setup is no longer sufficient |
| GitOps | **ArgoCD or Flux** | Paired with the Kubernetes migration, not before |
| Autoscaling | **KEDA** | Once there's a real queue-depth or GPU-load signal to scale against (i.e., after vLLM and a real queue exist) |
| Observability | **Prometheus + Grafana**, **Langfuse**, **OpenTelemetry** | Once there's a running service worth instrumenting beyond local logs |
| Cloud provider | AWS or GCP (pick one, no split effort) | Decision deferred until the Terraform/Kubernetes milestone — premature to lock in before there's real infra to provision |
| Load testing | **Locust or k6** | Once there's a deployed endpoint worth load-testing, and a $/1000-requests metric becomes meaningful |

---

## Explicitly out of scope
- TensorFlow/PyTorch training loops — this project serves and operates models, it does not train them.
- Managed black-box-only services as the main story (e.g. pure hosted API + no infra depth anywhere) — the staged migrations above exist specifically so the project demonstrates real infra work over time, not just API integration.