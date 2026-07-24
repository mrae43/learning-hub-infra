#!/bin/sh
set -e

# Apply pending database migrations before serving traffic.
alembic upgrade head

# Start the API server.
exec uvicorn api.main:app --host 0.0.0.0 --port 8000
