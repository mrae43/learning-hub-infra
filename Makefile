# Root Makefile: a thin, local-dev-only wrapper around the container lifecycle in
# docker-compose.yml (Postgres + pgvector). Not a general task runner - lint/test/
# format stay explicit uv run invocations - not a migration runner (Alembic), and
# never invoked by CI (workflows provision Postgres via their own services: block).

SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c
.DEFAULT_GOAL := help

# Scope `make logs` to a single service, e.g. `make logs SERVICE=postgres`.
SERVICE ?=

# Image tag pulled by `make deploy`. Defaults to `latest`; override for
# rollback, e.g. `make deploy IMAGE_TAG=1.2.3` (published semver tags carry no
# `v` prefix). Exported so the compose.deploy.yml `image:` interpolation sees it.
IMAGE_TAG ?= latest
export IMAGE_TAG

# Bounded /health poll used by `make deploy` (via scripts/health-poll.sh).
HEALTH_POLL_ATTEMPTS ?= 30
HEALTH_POLL_INTERVAL ?= 2

.PHONY: help
help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

.PHONY: check-docker
check-docker: ## Preflight: fail fast with a clear message if the Docker daemon is down
	@docker info > /dev/null 2>&1 || (echo "Docker daemon is not running. Start Docker and retry." && exit 1)

.PHONY: up
up: check-docker ## Start the local dev stack (Postgres + pgvector) in the background
	docker compose up -d

.PHONY: logs
logs: check-docker ## Tail logs (all services, or SERVICE=<name> for one)
	docker compose logs -f --tail=200 $(SERVICE)

.PHONY: down
down: check-docker ## Stop and remove containers/network (never deletes the pgvector data volume)
	docker compose down

.PHONY: load-up
load-up: check-docker ## Start the volume-load stack (mock upstream + api) in the background
	OPENAI_API_KEY=sk-load OPENAI_BASE_URL=http://mock-upstream:8080/v1 COHERE_API_KEY= docker compose --profile load up -d

.PHONY: load-down
load-down: check-docker ## Stop the volume-load stack (never deletes the pgvector data volume)
	docker compose --profile load down

# Load generator (decision #271, issue #290): one Locust locustfile driven by
# the LOAD_PROFILE env var. Volume runs sustained against the mock-upstream
# stack (`make load-up` first); smoke runs ~1 upstream user (plus a low-rate
# liveness user) against the real API with a capped upstream-backed budget.
# Each run evaluates the error-rate and p95 ceilings and exits non-zero when
# they're breached. LOAD_TARGET_URL overrides the api base URL; LOAD_RUN_TIME
# bounds a volume run (e.g. LOAD_RUN_TIME=5m).
LOAD_TARGET_URL ?= http://localhost:8000
LOAD_RUN_TIME ?=

.PHONY: load-run
load-run: ## Run a sustained volume load run against the mock-upstream stack
	LOAD_PROFILE=volume uv run --package scripts locust -f scripts/loadgen/locustfile.py \
		--host $(LOAD_TARGET_URL) --headless --only-summary \
		$(if $(LOAD_RUN_TIME),--run-time $(LOAD_RUN_TIME),)

.PHONY: smoke-run
smoke-run: ## Run a budgeted smoke run against the real API (~1 upstream user, <=50 upstream-backed requests)
	LOAD_PROFILE=smoke uv run --package scripts locust -f scripts/loadgen/locustfile.py \
		--host $(LOAD_TARGET_URL) --headless --only-summary

# Observability profile (issue #289, metrics path): OTel Collector + Prometheus +
# Grafana. `up-observability` starts the metrics trio and points the api's
# OTLP/HTTP export at the collector; `down-observability` stops the stack but
# never deletes the named data volumes (matching `down`). The Langfuse traces
# path is a sibling ticket (issue #291) and joins this profile later.
.PHONY: up-observability
up-observability: check-docker ## Start the metrics stack (OTel Collector + Prometheus + Grafana) in the background
	OTEL_EXPORTER_OTLP_ENDPOINT=http://otel-collector:4318 docker compose --profile observability up -d

.PHONY: down-observability
down-observability: check-docker ## Stop the observability stack (never deletes the named data volumes)
	docker compose --profile observability down

.PHONY: deploy
deploy: check-docker ## Deploy the demo stack from the published GHCR image (pull, no-build up, poll /health)
	docker compose -f docker-compose.yml -f compose.deploy.yml pull
	docker compose -f docker-compose.yml -f compose.deploy.yml up -d --no-build
	@bash scripts/health-poll.sh $(HEALTH_POLL_ATTEMPTS) $(HEALTH_POLL_INTERVAL) \
		|| (docker compose -f docker-compose.yml -f compose.deploy.yml logs api; exit 1)

# Edge gate (decision #269): Caddy reverse proxy + basic-auth + TLS in front of
# the demo stack. Explicitly opt-in only — never part of `up`/`deploy`, so the
# dev stack is unchanged by default. EDGE_GATE_AUTH is read from `.env`
# (required; compose fails fast without it). These targets are the LOCAL
# scaffold path (localhost + self-signed TLS); the public phase layers
# compose.deploy.yml too — see the README "edge gate" section for that command.
EDGE_GATE_DOMAIN ?= localhost
EDGE_GATE_TLS ?= tls internal
export EDGE_GATE_DOMAIN EDGE_GATE_TLS
# Dummy value for teardown: `stop`/`rm` don't recreate the container, so a
# placeholder satisfies the compose interpolation requirement for the
# EDGE_GATE_AUTH `:?` check without needing real credentials.
EDGE_GATE_AUTH_TEARDOWN := teardown-dummy

.PHONY: up-edge
up-edge: check-docker ## Start the edge gate (Caddy) in front of the local dev stack
	docker compose -f docker-compose.yml -f compose.edge.yml up -d edge-gate

.PHONY: down-edge
down-edge: check-docker ## Stop and remove only the edge gate (dev stack untouched)
	EDGE_GATE_AUTH="$(EDGE_GATE_AUTH_TEARDOWN)" \
	docker compose -f docker-compose.yml -f compose.edge.yml stop edge-gate
	EDGE_GATE_AUTH="$(EDGE_GATE_AUTH_TEARDOWN)" \
	docker compose -f docker-compose.yml -f compose.edge.yml rm -f edge-gate
