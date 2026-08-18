# syntax=docker/dockerfile:1
# Standalone image for the mock upstream service. Built with the repo root as
# the build context (it needs the root pyproject.toml + uv.lock for the
# workspace), via compose's `build.dockerfile: mock_upstream/Dockerfile`. The
# api image deliberately excludes this package, so the mock never ships there.
FROM python:3.12-slim AS builder

WORKDIR /app

# Install uv.
COPY --from=ghcr.io/astral-sh/uv:0.11.24 /uv /uvx /bin/
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy

# Phase 1: dependency metadata only (root workspace + the mock-upstream
# member's pyproject), so this layer survives source edits.
COPY pyproject.toml uv.lock ./
COPY mock_upstream/pyproject.toml ./mock_upstream/

# Install third-party dependencies without the workspace members. The cache
# mount keeps wheels in the local uv cache even when a lockfile change
# invalidates this layer.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-workspace

# Phase 2: the mock-upstream source, then install only that member.
COPY mock_upstream/src ./mock_upstream/src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-editable --package mock-upstream

FROM python:3.12-slim
WORKDIR /app

# Run as a non-root user.
RUN groupadd -r app && useradd -r -g app app

COPY --from=builder --chown=app:app /app/.venv ./.venv
ENV PATH="/app/.venv/bin:$PATH"

USER app

CMD ["uvicorn", "mock_upstream.app:app", "--host", "0.0.0.0", "--port", "8080"]
