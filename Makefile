# Root Makefile: a thin, local-dev-only wrapper around the container lifecycle in
# docker-compose.yml (Postgres + pgvector). Not a general task runner - lint/test/
# format stay explicit uv run invocations - not a migration runner (Alembic), and
# never invoked by CI (workflows provision Postgres via their own services: block).

SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c
.DEFAULT_GOAL := help

# Scope `make logs` to a single service, e.g. `make logs SERVICE=postgres`.
SERVICE ?=

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
