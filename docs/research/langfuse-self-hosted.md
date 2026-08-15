# Running Langfuse self-hosted in the local docker-compose stack

> Research note resolving Wayfinder ticket **#267 — "Research: Langfuse
> self-hosted topology"**, part of the **"Deployed & Monitored Demo"** map. This
> is the "Deployed & Monitored Demo" → "Langfuse
> self-hosted via compose" standing decision: self-hosted Langfuse receives
> stage-level spans via the single OTel path established in the sibling note
> `docs/research/otel-instrumentation-plan.md` (ONE OTel path fans out through a
> standalone OTel Collector to BOTH Langfuse and Prometheus/Grafana). That lock
> is not re-litigated here.
>
> Survey question: what the self-hosted Langfuse stack requires in the local
> compose file — exact services (web/worker + backing stores), whether a
> Postgres-only ("core") mode exists and suffices at demo scale, resource
> footprint, how OTel spans reach it (OTL ingestion endpoint vs SDK), UI
> access/auth, version pinning, whether it shares the app's Postgres or gets its
> own, and how it folds into the existing two-service compose (postgres + api)
> alongside the standalone OTel Collector.

## 1. Scope and locked constraints

Per the map (not re-litigated here):

- **One instrumentation path, two backends.** The sibling note
  `docs/research/otel-instrumentation-plan.md` locks the design: the app exports
  OTLP/HTTP once to a **standalone OTel Collector** in compose, which fans out
  traces → Langfuse (standard `otlphttp` exporter at
  `http://langfuse:3000/api/public/otel`, Basic-Auth `pk-lf-…:sk-lf-…` headers +
  `x-langfuse-ingestion-version: 4`) and metrics → Prometheus/Grafana. **The app
  does NOT use the Langfuse SDK.**
- **Langfuse is self-hosted via docker-compose.** The repo already runs its
  local demo via docker compose (`docker-compose.yml` today runs `postgres` +
  `api`).
- **Repo stack:** Postgres 16 + pgvector (`pgvector/pgvector:pg16`), FastAPI +
  uvicorn, Python 3.12. Pinned deps `fastapi>=0.141.1`, `uvicorn>=0.52.1`,
  `sqlalchemy>=2`, `psycopg2-binary>=2.9`.
- **Map standing decisions:** local deployment now (compose), hybrid (VPS/k3s)
  later; no app-level auth (ADR-0018); `cd.yml` is the single build path (GHCR);
  showcase bar is dashboards + CI/CD + runbook, ~30h/week timebox; traffic
  source is the Locust load generator in `scripts/` replaying eval-corpus
  queries. **There is no monitoring code in the repo today.**

### Method note

Every claim below was traced to a **primary source** — official Langfuse docs at
langfuse.com/self-hosting and langfuse.com/docs, the official
`docker-compose.yml` reference in `langfuse/langfuse`, and Langfuse source —
read on 15 August 2026. No secondary write-ups were used as evidence. Versions
and status are current as of that date; Langfuse v4 is evolving quickly and
self-hosted version tags move.

## 2. The short answer

- **The self-hosted stack requires six services** in the official compose
  reference: `langfuse-web`, `langfuse-worker`, **Postgres**, **ClickHouse**,
  **Redis**, and **MinIO (S3-compatible object store)**. ClickHouse, Redis, and
  S3 are **not optional** in current Langfuse v4.
- **There is NO Postgres-only ("core") mode.** The Langfuse docs FAQ states
  explicitly: *"ClickHouse is currently a required component for self-hosting
  Langfuse… Langfuse cannot be self-hosted without using ClickHouse."* Postgres
  stores only transactional/state data (users, orgs, projects, API keys,
  settings); **all traces, observations, and scores live in ClickHouse**. At any
  scale — including this demo — a self-hosted instance needs ClickHouse.
- **Resource footprint:** the full official compose is heavier than the repo's
  current two-service stack. Langfuse's own VM guidance is "at least 4 cores and
  16 GiB" for a self-hosted instance; ClickHouse is the largest consumer. A
  docker-compose stack realistically consumes several GB of RAM across
  web + worker + ClickHouse + Postgres + Redis + MinIO.
- **OTel ingestion path:** self-hosted Langfuse consumes OTel spans directly on
  `/api/public/otel` (OTLP; trace endpoint `/api/public/otel/v1/traces`). **HTTP
  only — gRPC is not supported.** Auth is HTTP Basic Auth over a base64
  `pk-lf-…:sk-lf-…` keypair plus `x-langfuse-ingestion-version: 4` for real-time
  v4 ingestion. The collector reaches it with a **standard `otlphttp` exporter**;
  no dedicated "Langfuse exporter" exists and **no Langfuse SDK is required** —
  this aligns exactly with the locked OTel note.
- **UI/auth:** the web UI/API is exposed on host port **3000**. Default
  self-hosted auth is **email/password**, created on first run in the UI (or
  provisioned headlessly via `LANGFUSE_INIT_*` env vars). Admin/API keys live in
  the project settings in the UI.
- **Version pinning:** the official compose pins Langfuse images with the **major
  tag `:4`** (`langfuse/langfuse:4`, `langfuse/langfuse-worker:4`) — a floating
  major tag, not a full semver pin; backing stores are pinned to specific images
  (`clickhouse-server:25.12`, `redis:7`, `postgres:${POSTGRES_VERSION:-17}`,
  `cgr.dev/chainguard/minio`). Langfuse recommends keeping the server up to date
  and upgrading with `docker compose up --pull always`.
- **Postgres:** Langfuse should get **its own database and its own user**, not
  share the app's `learning_hub` DB. It can share the Postgres **container**,
  but must use a **separate database name and user** (the app's DB has pgvector
  and its own migration ownership; Langfuse owns its schema and uses the
  `public` schema of its selected database). This is the recommended, cleanest
  boundary.

## 3. Q1 — Service topology: what the self-hosted stack requires

Primary sources: [official `docker-compose.yml` in
langfuse/langfuse](https://github.com/langfuse/langfuse/blob/main/docker-compose.yml),
[Self-host Overview (Architecture)](https://langfuse.com/self-hosting),
[Docker Compose deployment
guide](https://langfuse.com/self-hosting/deployment/docker-compose).

The architecture docs name two **application containers** and three **storage
components** plus an optional external LLM gateway:

- **Application containers:** `langfuse-web` (the web app serving the UI and
  APIs) and `langfuse-worker` (async event processing).
- **Storage components:** Postgres (transactional/state), ClickHouse (OLAP for
  traces/observations/scores), Redis/Valkey (cache + queue), and an S3-compatible
  object store (event/media persistence). The official compose uses **MinIO** as
  the S3 store.

The official compose reference (`docker-compose.yml`) defines these services:

| Compose service | Image (as pinned in the reference) | Role |
|---|---|---|
| `langfuse-worker` | `docker.io/langfuse/langfuse-worker:4` | Async worker |
| `langfuse-web` | `docker.io/langfuse/langfuse:4` | Web UI + APIs + OTel endpoint |
| `clickhouse` | `docker.io/clickhouse/clickhouse-server:25.12` | OLAP (traces/observations/scores) |
| `minio` | `cgr.dev/chainguard/minio` | S3-compatible object store (events/media) |
| `redis` | `docker.io/redis:7` | Cache + queue |
| `postgres` | `docker.io/postgres:${POSTGRES_VERSION:-17}` | Transactional DB |

Optionality (from the docs):

- **ClickHouse — required.** "There is no alternative OLAP database supported…
  Langfuse cannot be self-hosted without using ClickHouse."
- **Redis — required** in the full reference (queue + API-key/prompt cache; the
  worker consumes events via Redis).
- **S3/MinIO — required** in the full reference: all incoming events are
  persisted to S3 first, then the worker ingests them into ClickHouse. The
  compose healthchecks for `langfuse-web`/`langfuse-worker` `depends_on` all of
  `postgres`, `minio`, `redis`, and `clickhouse` with `condition:
  service_healthy`.
- **Postgres — required.**
- The **external LLM gateway** is optional (only used by the playground and
  evals).

Versions current as of 15 Aug 2026: the reference pins Langfuse to the **major
`4` tag** for both app containers, ClickHouse `25.12`, Redis `7`, and Postgres
`17` (overridable via `POSTGRES_VERSION`). The compose reference is the v4
configuration; Langfuse v4 requires Postgres >= 15 (16 recommended) and
ClickHouse >= 25.12 (26.4 recommended).

## 4. Q2 — Postgres-only mode at demo scale

Primary source: [ClickHouse (self-hosted) FAQ — "Is ClickHouse required for
self-hosting Langfuse?"](https://langfuse.com/self-hosting/deployment/infrastructure/clickhouse).

**There is no Postgres-only mode.** The docs are unambiguous:

> **Is ClickHouse required for self-hosting Langfuse?**
> Yes, ClickHouse is currently a required component for self-hosting Langfuse.
> There is no alternative OLAP database supported at this time. Langfuse cannot
> be self-hosted without using ClickHouse as the main storage solution for
> traces, observations, and scores. All self-hosted deployments must include a
> ClickHouse instance.

The split of responsibilities is explicit:

- **Postgres** stores "all transactional data, including Users, Organizations,
  Projects, Datasets, Encrypted API keys, Settings" ([Postgres
  docs](https://langfuse.com/self-hosting/deployment/infrastructure/postgres)).
- **ClickHouse** "is the main OLAP storage solution within Langfuse for our
  Trace, Observation, and Score entities. It is optimized for high write
  throughput and fast analytical queries." ([ClickHouse
  docs](https://langfuse.com/self-hosting/deployment/infrastructure/clickhouse)).

Because the **observability data itself** (traces, observations, scores) is what
this demo is collecting, and that data lives **only in ClickHouse**, there is no
scale threshold below which Postgres-only becomes viable — ClickHouse is the
storage engine for the traces regardless of volume. The tradeoff framing in the
ticket (ClickHouse being "optional at low scale") does not apply to current
Langfuse v4: the FAQ explicitly rules it out. (In earlier v2/v3 times there were
experiments around reduced modes, but the v4 self-hosting story is
ClickHouse-required; the docs' own guidance is the FAQ quoted above.)

**At demo scale** (~load generator replaying eval-corpus queries, ~30h/week),
the practical lever is not dropping ClickHouse but running it as the single
development container the compose reference ships (`CLICKHOUSE_CLUSTER_ENABLED`
defaults to `false` in that file, i.e. single-node). The single-container
ClickHouse Docker mode is documented as "not recommended for production" but is
exactly the low-scale, single-instance path this demo needs.

## 5. Q3 — Resource footprint

Primary sources: [Docker Compose deployment
guide](https://langfuse.com/self-hosting/deployment/docker-compose) (VM sizing
recommendation), [ClickHouse (self-hosted)
docs](https://langfuse.com/self-hosting/deployment/infrastructure/clickhouse)
(ClickHouse is CPU/memory intensive).

- Langfuse's own **VM recommendation** for a self-hosted instance is **"at
  least 4 cores and 16 GiB of memory, e.g. a t3.xlarge on AWS"**, and "100GiB"
  storage (observability data is large). That is the official guidance for a
  single host running the whole stack.
- **ClickHouse** is called out as "CPU and memory intensive for analytical and
  highly concurrent requests." It is the dominant consumer in the stack.
- The six-service compose (2 app + 4 backing stores) collectively consumes on
  the order of **several GB of RAM** when idle-to-light: Postgres and Redis are
  modest (hundreds of MB), ClickHouse holds a large heap (GB-scale), MinIO
  lightweight, and the two Node-based app containers tens-to-hundreds of MB
  each. Exact per-service numbers are **not** published by Langfuse; the only
  hard number is the 4-core/16-GiB host recommendation.
- Implication for this repo: a local machine comfortably above 4 cores / 16 GiB
  (and with disk for the `langfuse_*` volumes) will run the full stack. This is
  a notable step up from the current two-service `postgres` + `api` compose, but
  is the documented minimum for self-hosting.

## 6. Q4 — OTel ingestion path

Primary source: [Langfuse "OpenTelemetry (OTEL) for LLM Observability"
docs](https://langfuse.com/docs/opentelemetry/get-started).

This section **confirms and aligns with** the locked sibling note
`otel-instrumentation-plan.md` (§6). Primary-source details:

- **Self-hosted Langfuse consumes OTel spans directly** on the OTLP endpoint
  `/api/public/otel` (trace endpoint `/api/public/otel/v1/traces` for
  signal-specific collectors). The local endpoint is
  `http://localhost:3000/api/public/otel` (>= v3.22.0); inside compose this is
  `http://langfuse:3000/api/public/otel`.
- **HTTP only.** "Langfuse currently supports OTLP over HTTP with both
  `HTTP/JSON` and `HTTP/protobuf`. `gRPC` is not supported yet." The collector
  therefore uses an **`otlphttp` exporter** — no gRPC anywhere on the Langfuse
  leg.
- **Auth headers:** Langfuse uses **HTTP Basic Auth**. The docs give the exact
  exporter config:
  ```bash
  OTEL_EXPORTER_OTLP_ENDPOINT="http://localhost:3000/api/public/otel"
  OTEL_EXPORTER_OTLP_HEADERS="Authorization=Basic ${AUTH_STRING},x-langfuse-ingestion-version=4"
  ```
  where `AUTH_STRING` is base64 of `pk-lf-1234567890:sk-lf-1234567890`, and
  `x-langfuse-ingestion-version: 4` enables **real-time v4 ingestion** (without
  it, directly-ingested OTel data can be delayed up to 10 minutes).
- **Collector YAML (primary source, verbatim shape):** the docs publish an
  `otlphttp/langfuse` exporter:
  ```yml
  exporters:
    otlphttp/langfuse:
      endpoint: "https://cloud.langfuse.com/api/public/otel"  # → local: http://langfuse:3000/api/public/otel
      headers:
        Authorization: "Basic ${AUTH_STRING}"
        x-langfuse-ingestion-version: "4"
  ```
  fed by a traces pipeline `receivers: [otlp] → processors: [memory_limiter, batch] → exporters: [otlphttp/langfuse]`. This is a **standard `otlphttp` exporter**; there is **no dedicated Langfuse collector exporter**.
- **No Langfuse SDK required** for this wiring. The docs note the OTel-native
  **Langfuse SDK v4** exists as a thin layer over the official OpenTelemetry
  client, and recommend it for Python/JS — but for a collector fan-out the app
  emits standard OTel and the collector reaches Langfuse via `otlphttp`,
  keeping the single-path lock. **The app does NOT use the Langfuse SDK.**
- **Version note:** the OTel endpoint landed in **v3.22.0** and improved since;
  on 4xx errors the docs advise upgrading the self-hosted server to latest. The
  real-time `x-langfuse-ingestion-version: 4` header is required for the v4 data
  model.
- **Attribute mapping** (for useful traces): Langfuse maps spans to its model
  from `gen_ai.*` plus the `langfuse.*` namespace; any span with a `model`
  attribute becomes a **generation**; token usage from `gen_ai.usage.*`, cost
  from `gen_ai.usage.cost`. Trace-level fields (userId/sessionId/metadata/tags)
  should be propagated to **all** spans via OTel **Baggage +
  BaggageSpanProcessor**. All consistent with the sibling note.

## 7. Q5 — UI access & auth

Primary sources: [Docker Compose deployment
guide](https://langfuse.com/self-hosting/deployment/docker-compose),
[Authentication & SSO (self-hosted)](https://langfuse.com/self-hosting/security/authentication-and-sso),
[Headless Initialization](https://langfuse.com/self-hosting/administration/headless-initialization),
official [`docker-compose.yml`](https://github.com/langfuse/langfuse/blob/main/docker-compose.yml).

- **Port:** the web container publishes `3000:3000`; the UI/API is at
  `http://localhost:3000` (compose network hostname `langfuse:3000`). Langfuse
  recommends restricting inbound host traffic to only `langfuse-web` (3000) and
  `minio` (9090); all other components bind to `127.0.0.1` in the reference.
- **Default auth — email/password.** "Email/password authentication is enabled
  by default. Users can sign up and log in using their email and password." On a
  fresh self-hosted instance the first user creates their account via the UI
  sign-up page (no SMTP needed for signup).
- **Admin/project/keys:** the first user creates an organization and project in
  the UI; **API keys are found in the project settings within the UI**. These
  `pk-lf-…` / `sk-lf-…` keys are exactly what the collector's Basic Auth uses.
- **Headless provisioning (recommended for compose/CI):** set
  `LANGFUSE_INIT_*` env vars on the web container to auto-create the org,
  project, user, and API keys on startup (no UI click-through). The reference
  file already exposes `LANGFUSE_INIT_ORG_ID`, `LANGFUSE_INIT_ORG_NAME`,
  `LANGFUSE_INIT_PROJECT_ID`, `LANGFUSE_INIT_PROJECT_NAME`,
  `LANGFUSE_INIT_PROJECT_PUBLIC_KEY`, `LANGFUSE_INIT_PROJECT_SECRET_KEY`,
  `LANGFUSE_INIT_USER_EMAIL`, `LANGFUSE_INIT_USER_NAME`,
  `LANGFUSE_INIT_USER_PASSWORD`. (Caveat in docs: do not double-quote these in
  compose.)
- **Other auth-relevant env:** `NEXTAUTH_URL`, `NEXTAUTH_SECRET`, and the
  `SALT`/`ENCRYPTION_KEY`/`DATABASE_URL` secrets marked `# CHANGEME` in the
  reference. `AUTH_DISABLE_SIGNUP=true` can later lock down new signups; for
  this local demo the default email/password path is fine.
- **No app-level auth in the repo** (ADR-0018) is unaffected — Langfuse's own
  login is a separate concern inside the Langfuse dashboard.

## 8. Q6 — Version pinning

Primary sources: official [`docker-compose.yml`](https://github.com/langfuse/langfuse/blob/main/docker-compose.yml),
[Versions & Compatibility](https://langfuse.com/self-hosting/upgrade/versioning),
[Docker Compose upgrade guide](https://langfuse.com/self-hosting/deployment/docker-compose#how-to-upgrade).

- The official compose pins the **Langfuse app images to the major tag**:
  `docker.io/langfuse/langfuse:4` and `docker.io/langfuse/langfuse-worker:4`.
  This is a floating **major-version tag** (`:4`), NOT a full semver pin — it
  tracks the latest 4.x patch/minor of the v4 line.
- Backing stores are pinned to concrete images: `clickhouse-server:25.12`,
  `redis:7`, `postgres:${POSTGRES_VERSION:-17}` (17 default, overridable), and
  `cgr.dev/chainguard/minio` (floating digest-tracked image).
- Langfuse uses **semantic versioning**: major bumps are reserved for
  infrastructure changes and removal/change of Public APIs; DB schemas and
  frontend APIs are internal. Each **server major supports the current and
  previous SDK major**; the v4 transition is a documented exception.
- **Langfuse recommends keeping the server up to date** ("We recommend keeping
  the Langfuse Server up to date to ensure access to all features and security
  updates"). The upgrade path for compose is `docker compose up --pull always`.
  The `:4` floating tag is how the reference itself is pinned; if the demo wants
  reproducibility, an explicit full tag (e.g. `4.x.y`) can be substituted, but
  the official reference uses `:4`.

## 9. Q7 — Postgres sharing: own DB/user vs shared

Primary sources: [Postgres (self-hosted)](https://langfuse.com/self-hosting/deployment/infrastructure/postgres),
[Versions & Compatibility](https://langfuse.com/self-hosting/upgrade/versioning),
official [`docker-compose.yml`](https://github.com/langfuse/langfuse/blob/main/docker-compose.yml).

- Langfuse **requires a persistent Postgres** for its state and "uses the
  `public` schema in the selected database." It needs its own **database name
  and user** at minimum. Langfuse v4 requires Postgres >= 15 (16 recommended).
- The app's Postgres (`pgvector/pgvector:pg16`) runs the `learning_hub` database
  with user `learning_hub`; the **pgvector extension is configured there** for
  the app's retrieval. Langfuse does **not** need pgvector and does **not** need
  the app's tables.
- **Recommendation: share the Postgres container, but give Langfuse its own
  database and its own user.** Concretely, add a separate DB (e.g.
  `langfuse`) and a separate user (e.g. `langfuse`) on the same Postgres
  instance. Reasons:
  - **Migration ownership:** Langfuse owns and runs its own schema migrations in
    its database's `public` schema; the app owns its own migrations. Mixing both
    in the `learning_hub` DB risks schema collisions and confuses each project's
    migration/rollback ownership.
  - **Extension isolation:** the app's DB has pgvector enabled; keeping Langfuse
    out of it avoids any interaction with the vector extension and the app's
    dense/sparse indexes.
  - **Clean deletion/downgrade:** the demo can drop the `langfuse` database and
    user wholesale without touching `learning_hub`, and vice versa.
  - **Version independence:** Langfuse wants Postgres 15+/16 recommended;
    `pgvector/pgvector:pg16` is Postgres 16, so sharing the container is
    compatible. (If we later wanted a different Postgres version for Langfuse,
    a separate container is trivial.)
- This means the existing `postgres` service gains a second database+user (via
  init or the official compose's own `postgres` being its own container). The
  **official reference runs its own separate `postgres` container**; either
  approach is valid, but the lowest-friction option that keeps the app's
  container untouched is a **second database + user on the existing Postgres**
  (or a small dedicated `langfuse-postgres` service). A **separate DB + user is
  mandatory** in every case; sharing the app's exact `learning_hub` DB/user is
  the one thing to avoid.

## 10. Q8 — Docker-compose placement

The existing `docker-compose.yml` runs two services: `postgres`
(`pgvector/pgvector:pg16`) and `api` (FastAPI, depends on postgres healthy).
Per the locked OTel note, the standalone **OTel Collector** also needs to be
added. The resulting composition adds, for Langfuse, the six-service official
stack plus the collector:

1. **`langfuse-web`** — `docker.io/langfuse/langfuse:4`, port `3000:3000`; the
   OTel endpoint target `langfuse:3000/api/public/otel`; set `NEXTAUTH_URL`,
   `NEXTAUTH_SECRET`, `SALT`, `ENCRYPTION_KEY`, `DATABASE_URL`, ClickHouse env,
   S3 env, and (for headless init) `LANGFUSE_INIT_*`.
2. **`langfuse-worker`** — `docker.io/langfuse/langfuse-worker:4`, shares the
   env; depends on postgres/minio/redis/clickhouse healthy.
3. **`clickhouse`** — `docker.io/clickhouse/clickhouse-server:25.12`,
   `CLICKHOUSE_CLUSTER_ENABLED=false`, UTC timezone.
4. **`redis`** — `docker.io/redis:7` with a password.
5. **`minio`** — `cgr.dev/chainguard/minio`, S3 store for events/media; exposes
   9090:9000 (and 9091:9001 console).
6. **Postgres** — either a dedicated `langfuse-postgres` service (official
   reference style) or a separate `langfuse` DB+user on the existing `postgres`
   container (see §9).
7. **`otel-collector`** — `otel/opentelemetry-collector-contrib` (from the OTel
   note): the app exports OTLP/HTTP to it; it fans out traces → Langfuse
   (`otlphttp` exporter → `http://langfuse:3000/api/public/otel`, Basic-Auth +
   `x-langfuse-ingestion-version: 4`) and metrics → Prometheus/Grafana.

Net effect: the compose file grows from 2 services to ~9 (api, postgres,
langfuse-web, langfuse-worker, clickhouse, redis, minio, otel-collector, plus
Prometheus/Grafana per the OTel note). For a local demo, bind the Langfuse and
backing services to `127.0.0.1` where practical (as the reference does) and only
expose `3000` to the host.

## 11. Findings at a glance

| Decision point | Finding |
|---|---|
| Services required | `langfuse-web` + `langfuse-worker`, plus Postgres, **ClickHouse (required)**, Redis, MinIO/S3 — six services total in the official compose. |
| Postgres-only mode | **Does not exist in v4.** ClickHouse is required for storing traces/observations/scores; FAQ rules out any Postgres-only deployment. |
| Resource footprint | Langfuse recommends ≥ 4 cores / 16 GiB for a self-hosted host; ClickHouse is the dominant consumer; stack uses several GB RAM. |
| OTel ingestion | Direct OTLP on `/api/public/otel` (HTTP only, no gRPC); Basic-Auth `pk-lf-…:sk-lf-…` + `x-langfuse-ingestion-version: 4`; standard `otlphttp` collector exporter; no Langfuse SDK needed. |
| UI access & auth | Port 3000; default email/password; keys in project settings; optional `LANGFUSE_INIT_*` headless provisioning. |
| Version pinning | Official compose uses floating major tag `:4` for Langfuse apps; concrete images for stores; Langfuse recommends keeping server current (`docker compose up --pull always`). |
| Postgres sharing | Langfuse gets **its own DB + user** (own schema/migration ownership; keeps pgvector app DB untouched). |
| Compose placement | Add the six-service Langfuse stack + the standalone OTel Collector to the existing 2-service compose. |

## 12. Cited primary sources

Langfuse docs:

- https://langfuse.com/self-hosting — Self-host Overview (Architecture: web/worker, Postgres/ClickHouse/Redis/S3)
- https://langfuse.com/self-hosting/deployment/docker-compose — Docker Compose deployment guide (ports, VM sizing, upgrade)
- https://langfuse.com/self-hosting/deployment/infrastructure/postgres — Postgres role + `public` schema + version reqs
- https://langfuse.com/self-hosting/deployment/infrastructure/clickhouse — ClickHouse role + "Is ClickHouse required?" FAQ
- https://langfuse.com/docs/opentelemetry/get-started — OTLP endpoint, HTTP-only, Basic-Auth headers, collector exporter YAML
- https://langfuse.com/self-hosting/security/authentication-and-sso — default email/password auth
- https://langfuse.com/self-hosting/administration/headless-initialization — `LANGFUSE_INIT_*` provisioning
- https://langfuse.com/self-hosting/upgrade/versioning — semver policy, `:4` major tag, upgrade guidance

Langfuse source / repo:

- https://github.com/langfuse/langfuse/blob/main/docker-compose.yml — official compose reference (service list, images, env, ports)

Sibling note (locked context, repo):

- `docs/research/otel-instrumentation-plan.md` — ONE OTel path → collector fan-out to Langfuse + Prometheus/Grafana; `otlphttp` exporter at `http://langfuse:3000/api/public/otel` with Basic-Auth + `x-langfuse-ingestion-version: 4`; no Langfuse SDK in app.

Repo (verified in-tree):

- `docker-compose.yml` — current `postgres` (pgvector:pg16) + `api` services.

## 13. Could not verify

- **Exact per-service memory/CPU of the Langfuse web/worker containers** — Langfuse publishes only the 4-core / 16-GiB host recommendation and qualitative statements ("ClickHouse is CPU and memory intensive"); no authoritative per-container RAM/CPU figures were found in the primary docs. The "several GB" figure is an inference from the stack composition, not a documented number.
- **Whether a Postgres-only / reduced "core" mode existed historically in v2/v3** — the current v4 FAQ is decisive that ClickHouse is required today, but I did not independently trace old v2/v3 changelogs for a previously-optional ClickHouse; the ticket's framing of "optional at low scale" appears to reflect an older state, not v4.
- **Whether the repo's `pgvector/pgvector:pg16` image can create a second database + user at container init** — the pgvector image inherits the official `postgres` init (`POSTGRES_DB`/`POSTGRES_USER` single-db behavior); adding a second DB+user may require an init script, which I did not verify against the pgvector image docs (external to Langfuse).
- **The exact current latest Langfuse v4 patch version** behind the `:4` floating tag — the reference pins `:4`; I did not enumerate the specific 4.x release as of the reading date.

## Recommendation

- **Add six services for self-hosted Langfuse** to the local compose: `langfuse-web`
  (`docker.io/langfuse/langfuse:4`, host port `3000`), `langfuse-worker`
  (`docker.io/langfuse/langfuse-worker:4`), `clickhouse`
  (`clickhouse/clickhouse-server:25.12`, `CLICKHOUSE_CLUSTER_ENABLED=false`, UTC),
  `redis` (`redis:7`), `minio` (`cgr.dev/chainguard/minio`), and a Postgres for
  Langfuse. Add the standalone **`otel-collector`**
  (`otel/opentelemetry-collector-contrib`) per the locked OTel note, with an
  `otlphttp` traces exporter → `http://langfuse:3000/api/public/otel` using
  Basic-Auth `pk-lf-…:sk-lf-…` + `x-langfuse-ingestion-version: 4`, and a
  metrics pipeline for Prometheus/Grafana.
- **Postgres-only is NOT enough — there is no such mode.** ClickHouse is a
  required component of self-hosting Langfuse (the FAQ says so directly); the
  traces/observations/scores the demo collects live only in ClickHouse. Use the
  single-node development ClickHouse the official compose ships. The demo stack
  is fine on a host above ~4 cores / 16 GiB.
- **Langfuse gets its own Postgres database and user, not the app's.** Share
  the Postgres container if convenient, but provision a separate `langfuse`
  database and `langfuse` user (own schema + migration ownership, keeps the
  pgvector-enabled `learning_hub` DB untouched). The app's `learning_hub`
  DB/user must not be reused by Langfuse.
- **Compose placement:** keep the existing `postgres` + `api`, and append the
  six Langfuse services plus `otel-collector` (and Prometheus/Grafana per the
  OTel note), binding non-UI services to `127.0.0.1` and exposing only `3000`
  for the Langfuse UI/dashboard.
