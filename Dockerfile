# syntax=docker/dockerfile:1
FROM python:3.12-slim AS builder

WORKDIR /app

# Install uv.
COPY --from=ghcr.io/astral-sh/uv:0.11.24 /uv /uvx /bin/
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy

# Copy workspace metadata.
COPY pyproject.toml uv.lock ./

# Copy package metadata and source code (tests excluded from production image).
COPY core/pyproject.toml ./core/
COPY core/src ./core/src
COPY retrieval_qa/pyproject.toml ./retrieval_qa/
COPY retrieval_qa/src ./retrieval_qa/src
COPY depth_dive/pyproject.toml ./depth_dive/
COPY depth_dive/src ./depth_dive/src
COPY api/pyproject.toml ./api/
COPY api/src ./api/src
COPY ingestion/pyproject.toml ./ingestion/
COPY ingestion/src ./ingestion/src

# Install production dependencies and workspace packages.
RUN uv sync --no-dev --frozen

FROM python:3.12-slim
WORKDIR /app

# Run as a non-root user.
RUN groupadd -r app && useradd -r -g app app

COPY --from=builder --chown=app:app /app/.venv ./.venv
ENV PATH="/app/.venv/bin:$PATH"

USER app

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
