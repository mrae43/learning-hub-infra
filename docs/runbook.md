# Runbook

> Operator-facing incident response for the Learning Hub observability stack.
> Every alert in [`observability/prometheus/alerts.yml`](../observability/prometheus/alerts.yml)
> has **exactly one** entry below, keyed by the alert name (issue #292). Each
> entry covers the symptom that fires the alert, the diagnosis steps that
> localise it, and the recovery actions that restore service. The "Induce it"
> block in each entry describes how to demonstrate the alert firing under a
> load run (`make load-up` + `make load-run` for volume traffic, `make
> up-observability` for the monitoring stack).

## Overview

- **Where alerts appear.** Firing rules are visible in the Prometheus UI
  (`http://localhost:9090/alerts`) and in Alertmanager
  (`http://localhost:9093/#/alerts`).
- **Delivery.** Alertmanager fans every alert to a single webhook sink. The
  sink URL comes from the `ALERTMANAGER_WEBHOOK_URL` env var (in `.env`).
  **Default is UI-only**: with the var unset, the sink points at a dead
  loopback port, so alerts are visible in the Alertmanager UI but nothing is
  delivered. Set `ALERTMANAGER_WEBHOOK_URL` to a real webhook (e.g. a
  Discord/Slack-compatible endpoint) to enable delivery, then
  `make down-observability && make up-observability` to restart Alertmanager.
- **Signal sources.** The RED rules read the api's span-derived metrics from
  the OTel Collector (`traces_span_metrics_*`). Because the api maps both
  upstream failures and its own database failures to 502/503 (ADR-0014), the
  rules distinguish the two by **stage**: upstream-backed stages
  (`embed`/`generate`/`web-search`/`rerank`) drive the Upstream502503 alert,
  while the api's own retrieval stage (`retrieve`, its pgvector database)
  drives Internal5xx. The ServiceDown rule reads a blackbox availability
  probe of the api's `/health`; the IngestionFailed rule reads the
  postgres-exporter custom query's `learning_hub_ingestion_failed` metric.
- **Thresholds are first-pass** (decision #271/#272): they key on the volume
  run's ceilings and get sharpened against smoke baselines.

---

## ServiceDown

**Severity:** critical — the api is unreachable.

**Symptom.** The api process is down, unresponsive, or unreachable on the
internal network: the blackbox probe of `http://api:8000/health` has failed
for over a minute (`probe_success{job="blackbox-api"} == 0`). `/query`,
`/dive`, and `/health` all fail for clients.

**Diagnosis.**

1. Check the probe target in Prometheus
   (`Status > Targets`, job `blackbox-api`) and the probe metric
   `probe_success{instance="http://api:8000/health"}`.
2. Confirm the api container state and recent logs:
   `docker compose ps api` and `make logs SERVICE=api`.
3. Check the host's container resources (`node_memory_Active_bytes`,
   `node_cpu_seconds_total`) — an OOM-killed or resource-starved container
   looks identical to a crash.

**Recovery.**

1. Restart the api: `docker compose restart api`.
2. If it restarts in a crash loop, inspect `docker compose logs api` and
   `docker inspect api` for the exit reason (OOM, panic, config error).
3. Confirm `/health` returns 200 and the probe is green again before resuming
   load.

**Induce it.** With the observability stack up and a volume load run in
flight, stop the api: `docker compose stop api`. The probe fails on the next
scrape interval; the alert fires within ~1 minute.

---

## Upstream502503

**Severity:** warning — the api is up but its upstream is failing.

**Symptom.** The api's upstream-backed stages failed in the last 5 minutes
(`sum(rate(traces_span_metrics_calls_total{span_name=~"embed|generate|
web-search|rerank", status_code="STATUS_CODE_ERROR"}[5m])) > 0`). The
embeddings / inference / web-search upstream (hosted APIs, or the mock
upstream during volume runs) is returning errors, is unreachable, or timed
out; the api surfaces the failures to clients as 502/503 (ADR-0014).

**Diagnosis.**

1. Confirm which stage fails: in Grafana, the Error-rate-by-stage panel, or
   `sum by (span_name) (rate(traces_span_metrics_calls_total{
   status_code="STATUS_CODE_ERROR"}[5m]))` — `embed`/`generate`/`web-search`
   point at the upstream-backed stages.
2. Confirm the api is returning 502/503 to clients: `sum by (
   http_response_status_code) (rate(traces_span_metrics_calls_total{
   span_name=~".* /.*"}[5m]))`.
3. For a mock-upstream run, check the mock's logs (`make logs
   SERVICE=mock-upstream`) and whether `MOCK_ERROR_RATE`/`MOCK_ERROR_STATUS`
   are set. For real APIs, check the provider's status page and the api logs
   for the underlying error.

**Recovery.**

1. If fault injection was left on, unset `MOCK_ERROR_RATE` (restore to 0) and
   restart the mock: `docker compose up -d mock-upstream`.
2. If the upstream is genuinely down, wait for it to recover; the alert
   clears once the stage errors stop for the 5-minute window.
3. Verify with a smoke run (`make smoke-run`) that `/query` returns 200.

**Induce it.** Start a volume run against the mock upstream with fault
injection: `MOCK_ERROR_RATE=0.5 MOCK_ERROR_STATUS=502 docker compose up -d
mock-upstream` (or set the same vars in `.env`). The upstream-backed stages
error and the api returns 502s; the alert fires within ~1 minute of sustained
errors.

---

## P95LatencyHigh

**Severity:** warning — the api is serving but slow.

**Symptom.** A client-facing endpoint's p95 latency exceeded 2000ms over the
last 5 minutes
(`histogram_quantile(0.95, sum by (le, span_name) (rate(
traces_span_metrics_duration_milliseconds_bucket{
span_name=~".* /.*"}[5m]))) > 2000`). The ceiling matches the volume run's
`/query` p95 limit (decision #271).

**Diagnosis.**

1. Identify the slow span: Grafana's p95-by-stage panel, or `topk(5,
   histogram_quantile(0.95, sum by (span_name, le) (rate(
   traces_span_metrics_duration_milliseconds_bucket[5m]))))`.
2. Compare against the load run's own p95 report
   (`make load-run`'s report, or `LOAD_RUN_TIME=5m make load-run`) to confirm
   the client-visible number matches the alert.
3. Check upstream latency: for a mock run, the `MOCK_*_LATENCY_MAX_MS` knobs;
   for real APIs, the provider dashboards/Langfuse spans
   (`http://localhost:3000`).

**Recovery.**

1. If the mock's simulated latency was raised for a demo, restore the default
   ranges and `docker compose up -d mock-upstream`.
2. If latency is genuinely high, look for upstream slowness, a slow database,
   or resource exhaustion (`node_*` metrics); reduce load or scale up.
3. Confirm p95 drops back under 2s for a 5-minute window.

**Induce it.** Raise the mock's latency before/while running a volume run:
`MOCK_CHAT_LATENCY_MAX_MS=4000 MOCK_EMBEDDINGS_LATENCY_MAX_MS=1000 docker
compose up -d mock-upstream`. `/query` and `/dive` p95 climb past 2s; the
alert fires within ~1 minute.

---

## Internal5xx

**Severity:** critical — the api's own internal component is failing.

**Symptom.** The api's internal retrieval stage errored, or the api returned
an internal 5xx (in practice 500) to clients, in the last 5 minutes
(`(sum(rate(traces_span_metrics_calls_total{span_name="retrieve",
status_code="STATUS_CODE_ERROR"}[5m])) > 0) or (sum(rate(
traces_span_metrics_calls_total{span_name=~".* /.*",
http_response_status_code=~"5..", http_response_status_code!~"502|503"}[5m])) > 0)`).
The retrieve stage is the api's own pgvector database — when it fails the api
surfaces 502/503 to clients (ADR-0014), so this alert watches the stage
directly rather than the client-visible code. The second clause catches true
internal 500s from an application bug or unexpected exception.

**Diagnosis.**

1. Confirm it is the retrieve stage: `sum by (span_name) (rate(
   traces_span_metrics_calls_total{status_code="STATUS_CODE_ERROR"}[5m]))`
   shows `retrieve` failing while the upstream stages stay green — that
   isolates the failure to the api's own database component.
2. Check the database: the postgres-exporter `pg_up`/`pg_stat_*` metrics, and
   whether Postgres is healthy (`docker compose ps postgres`).
3. If the failing signal is instead a client-visible 500, check the api logs
   for the traceback (`make logs SERVICE=api`) — the catch-all handler
   returns 500 with `{"detail": <error>}`.

**Recovery.**

1. If Postgres is down, restore it (`docker compose up -d postgres`) and wait
   for its healthcheck; retrieve stops erroring once the database is back.
2. If it's an application bug, reproduce against the failing input, fix, and
   redeploy (`make deploy` after the fix lands on `main`).
3. Confirm the retrieve stage is green and 5xx responses stop for a 5-minute
   window.

**Induce it.** With the observability stack up and load in flight, stop the
database: `docker compose stop postgres`. `/query` and `/dive` fail at the
retrieve stage (surfaced to clients as 503 per ADR-0014) while `/health`
returns its own 503 (ignored by both this alert and Upstream502503, whose
upstream stages stay green); the alert fires within ~1 minute.

---

## IngestionFailed

**Severity:** warning — background ingestion is failing documents.

**Symptom.** At least one document in the app's `learning_hub` database is
stuck in the `failed` ingestion state
(`learning_hub_ingestion_failed > 0`, a gauge from the postgres-exporter
custom query `learning_hub_ingestion`).

**Diagnosis.**

1. Check the gauge and which documents are affected:
   `select id, title, status, updated_at from documents where status =
   'failed';` against the `learning_hub` database.
2. Inspect the ingestion worker's failure in the api logs
   (`make logs SERVICE=api`) around the failing document's `updated_at`.
3. If failures cluster around one document type, check the chunker/embedding
   path for that type.

**Recovery.**

1. Re-ingest the failed documents after fixing the underlying cause (e.g.
   replace the bad file, or fix a transient upstream error) —
   `POST /ingest` again.
2. If the documents are unrecoverable, delete them
   (`DELETE FROM documents WHERE status = 'failed';` or via the app) so the
   gauge returns to 0.
3. Confirm the alert clears on the next evaluation once no failed documents
   remain.

**Induce it.** Mark a document failed directly in the database to observe the
alert (the load generator's `/ingest` path is out of scope per decision
#271, so a manual `UPDATE` is the quickest induction):
`docker compose exec postgres psql -U learning_hub -d learning_hub -c
"UPDATE documents SET status = 'failed' WHERE id = '<some-document-id>'";`
The gauge goes to 1 on the next postgres-exporter scrape; the alert fires
within ~1 minute. Restore with the same statement setting `status` back
(e.g. `'ready'`) or deleting the row.
