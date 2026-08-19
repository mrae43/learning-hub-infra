"""Run-level success gates: error-rate and p95-latency ceilings (decision #271).

Pure computation over a per-endpoint stats snapshot, so the gates are
unit-testable without a running Locust process. The locustfile adapts Locust's
stats objects into :class:`EndpointStats` and evaluates the verdict at
shutdown, using it to set the process exit code.

Error classification (decision #271):

- **volume** — any 4xx/5xx (including 502/503) is an error. The mock upstream
  is deterministic, so anything non-2xx is a genuine failure.
- **smoke** — 5xx is an error and 429 (real-API rate limiting) is exempt and
  logged, not failed; other 4xx still count (a malformed payload is a run
  bug). A transport failure (no usable status) is always an error.

Gates (decision #271): the run-level error rate must be <= 1%, and each
endpoint's measured p95 must be within its profile ceiling. An endpoint with
no samples is flagged (``p95_ok=None``) and reported as a warning rather than
failing the run; a run that made no requests at all cannot pass.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from scripts.loadgen.config import (
    ENDPOINTS,
    MAX_ERROR_RATE,
    P95_CEILINGS_MS,
    Endpoint,
    LoadProfile,
)

_SMOKE_EXEMPT_STATUS = 429


class LocustStatsEntry(Protocol):
    """The Locust ``StatsEntry`` surface the snapshot adapter needs."""

    name: str
    method: str
    num_requests: int
    num_failures: int

    def get_response_time_percentile(self, percent: float) -> int: ...


class LocustStats(Protocol):
    """The Locust ``RequestStats`` surface the snapshot adapter needs."""

    entries: dict[tuple[str, str], LocustStatsEntry]


@dataclass(frozen=True)
class EndpointStats:
    """Per-endpoint request statistics snapshot, decoupled from Locust."""

    name: Endpoint
    num_requests: int
    num_failures: int
    p95_response_time_ms: float | None


@dataclass(frozen=True)
class GateResult:
    """The p95 verdict for one endpoint against its ceiling."""

    endpoint: Endpoint
    num_requests: int
    num_failures: int
    error_rate: float
    p95_ms: float | None
    p95_ceiling_ms: int
    p95_ok: bool | None
    ok: bool


@dataclass(frozen=True)
class RunVerdict:
    """Aggregate verdict for a whole load run."""

    profile: LoadProfile
    total_requests: int
    total_failures: int
    error_rate: float
    error_rate_ok: bool
    gates: tuple[GateResult, ...]
    overall_ok: bool


def is_rate_limit_exempt(status_code: int | None, profile: LoadProfile) -> bool:
    """Check whether a response is an exempted smoke-run rate limit.

    Args:
        status_code: The HTTP status of the response (``None``/``0`` for a
            transport failure, which is never exempt).
        profile: The run profile.

    Returns:
        ``True`` only for HTTP 429 in a smoke run — real-API rate limits that
        are logged, not counted as errors (decision #271).
    """
    return profile is LoadProfile.SMOKE and status_code == _SMOKE_EXEMPT_STATUS


def is_error_status(status_code: int | None, profile: LoadProfile) -> bool:
    """Classify a response status as an error for the given profile.

    Args:
        status_code: The HTTP status of the response, or ``None``/``0`` for a
            transport failure that produced no usable response.
        profile: The run profile controlling the classification.

    Returns:
        ``True`` when the response counts against the run's error rate.
    """
    if not status_code:
        return True
    if status_code < 400:
        return False
    return not is_rate_limit_exempt(status_code, profile)


def snapshot_run_stats(stats: LocustStats) -> list[EndpointStats]:
    """Adapt Locust stats entries into plain :class:`EndpointStats` objects.

    Args:
        stats: A Locust ``RequestStats`` instance (``environment.runner.stats``).

    Returns:
        One :class:`EndpointStats` per recorded request name, in no particular
        order. Entries with zero requests carry ``None`` for the p95 value.
    """
    snapshot: list[EndpointStats] = []
    for entry in stats.entries.values():
        if entry.name not in ENDPOINTS:
            raise ValueError(f"unexpected endpoint in load run stats: {entry.name!r}")
        p95 = entry.get_response_time_percentile(0.95) if entry.num_requests else None
        snapshot.append(
            EndpointStats(
                name=entry.name,
                num_requests=entry.num_requests,
                num_failures=entry.num_failures,
                p95_response_time_ms=p95,
            )
        )
    return snapshot


def _error_rate(failures: int, requests: int) -> float:
    if requests == 0:
        return 0.0
    return failures / requests


def _evaluate_endpoint(stats: EndpointStats, profile: LoadProfile) -> GateResult:
    ceiling = P95_CEILINGS_MS[profile][stats.name]
    p95_ms = stats.p95_response_time_ms
    p95_ok = None if p95_ms is None else p95_ms <= ceiling
    return GateResult(
        endpoint=stats.name,
        num_requests=stats.num_requests,
        num_failures=stats.num_failures,
        error_rate=_error_rate(stats.num_failures, stats.num_requests),
        p95_ms=p95_ms,
        p95_ceiling_ms=ceiling,
        p95_ok=p95_ok,
        ok=p95_ok is not False,
    )


def evaluate_run(stats: list[EndpointStats], profile: LoadProfile) -> RunVerdict:
    """Evaluate a load run against the profile's gates.

    Args:
        stats: Per-endpoint statistics for the finished run.
        profile: The profile whose ceilings apply.

    Returns:
        A :class:`RunVerdict`. The error-rate gate is run-level only (decision
        #271); per-endpoint error rates are reported for information but do not
        independently fail the run. A run that made no requests at all fails.
    """
    total_requests = sum(s.num_requests for s in stats)
    total_failures = sum(s.num_failures for s in stats)
    error_rate = _error_rate(total_failures, total_requests)
    error_rate_ok = error_rate <= MAX_ERROR_RATE
    gates = tuple(_evaluate_endpoint(s, profile) for s in stats)
    overall_ok = bool(total_requests) and error_rate_ok and all(g.ok for g in gates)
    return RunVerdict(
        profile=profile,
        total_requests=total_requests,
        total_failures=total_failures,
        error_rate=error_rate,
        error_rate_ok=error_rate_ok,
        gates=gates,
        overall_ok=overall_ok,
    )
