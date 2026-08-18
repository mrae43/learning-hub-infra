#!/usr/bin/env bash
# Poll the api /health endpoint until it returns ok, or fail after a bounded
# number of attempts. Single source of truth for the deploy health poll used by
# `make deploy` (host-side) and the cd.yml smoke-test gate (CI runner).
set -eu

attempts="${1:-30}"
interval="${2:-2}"

for i in $(seq 1 "${attempts}"); do
  if curl -fsS http://localhost:8000/health | grep -q '"status":"ok"'; then
    echo "api /health OK"
    exit 0
  fi
  sleep "${interval}"
done

echo "error: api /health not OK after ${attempts} polls" >&2
exit 1
