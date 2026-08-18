# syntax=docker/dockerfile:1
FROM python:3.12-slim AS builder

WORKDIR /app

# Install uv.
COPY --from=ghcr.io/astral-sh/uv:0.11.24 /uv /uvx /bin/
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy

# Phase 1: dependency metadata only (root project, lockfile, and every
# workspace member's pyproject.toml), so this layer survives source edits.
COPY pyproject.toml uv.lock ./
COPY core/pyproject.toml ./core/
COPY retrieval_qa/pyproject.toml ./retrieval_qa/
COPY depth_dive/pyproject.toml ./depth_dive/
COPY api/pyproject.toml ./api/
COPY ingestion/pyproject.toml ./ingestion/
COPY scripts/pyproject.toml ./scripts/
COPY mock_upstream/pyproject.toml ./mock_upstream/

# Install third-party dependencies without the workspace members. The cache
# mount keeps wheels in the local uv cache even when a lockfile change
# invalidates this layer.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --all-packages --no-dev --no-editable --frozen --no-install-workspace

# Phase 2: source trees, migration configuration, and scripts.
COPY core/src ./core/src
COPY retrieval_qa/src ./retrieval_qa/src
COPY depth_dive/src ./depth_dive/src
COPY api/src ./api/src
COPY ingestion/src ./ingestion/src
COPY alembic.ini ./
COPY scripts/*.py ./scripts/

# Install the workspace members on top of the third-party dependencies. The
# mock-upstream package is deliberately excluded: it runs as its own compose
# service (mock_upstream/Dockerfile) and must never enter the api image.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --all-packages --no-dev --no-editable --frozen --no-install-package mock-upstream

FROM python:3.12-slim
WORKDIR /app

# Run as a non-root user.
RUN groupadd -r app && useradd -r -g app app

COPY --from=builder --chown=app:app /app/.venv ./.venv
COPY --from=builder --chown=app:app /app/alembic.ini ./
COPY --from=builder --chown=app:app /app/core/src/core/database/migrations ./core/src/core/database/migrations
COPY --chown=app:app docker-entrypoint.sh /docker-entrypoint.sh
ENV PATH="/app/.venv/bin:$PATH"

USER app

ENTRYPOINT ["/docker-entrypoint.sh"]
