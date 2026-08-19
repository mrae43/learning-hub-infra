"""Tests for scripts/loadgen/gates.py."""

from __future__ import annotations

import pytest

from scripts.loadgen.config import Endpoint, LoadProfile
from scripts.loadgen.gates import (
    EndpointStats,
    LocustStatsEntry,
    evaluate_run,
    is_error_status,
    is_rate_limit_exempt,
    snapshot_run_stats,
)


class _FakeEntry:
    """Minimal stand-in for Locust's ``StatsEntry``."""

    def __init__(
        self,
        name: str,
        method: str,
        num_requests: int,
        num_failures: int,
        p95: int,
    ) -> None:
        self.name = name
        self.method = method
        self.num_requests = num_requests
        self.num_failures = num_failures
        self._p95 = p95

    def get_response_time_percentile(self, percent: float) -> int:
        return self._p95


class _FakeStats:
    """Minimal stand-in for Locust's ``RequestStats``."""

    entries: dict[tuple[str, str], LocustStatsEntry]

    def __init__(self, entries: list[_FakeEntry]) -> None:
        self.entries = {(e.name, e.method): e for e in entries}


def _entry(name: str, requests: int, failures: int = 0, p95: int = 50) -> _FakeEntry:
    return _FakeEntry(name, "POST", requests, failures, p95)


def _stats_of(
    name: Endpoint,
    requests: int,
    failures: int = 0,
    p95: float | None = 50.0,
) -> EndpointStats:
    return EndpointStats(
        name=name,
        num_requests=requests,
        num_failures=failures,
        p95_response_time_ms=p95,
    )


class TestIsErrorStatus:
    def test_volume_any_4xx_or_5xx_is_error(self) -> None:
        for status in (400, 404, 422, 429, 500, 502, 503):
            assert is_error_status(status, LoadProfile.VOLUME) is True

    def test_volume_2xx_and_3xx_are_not_errors(self) -> None:
        for status in (200, 201, 204, 301, 302):
            assert is_error_status(status, LoadProfile.VOLUME) is False

    def test_smoke_5xx_is_error(self) -> None:
        for status in (500, 502, 503):
            assert is_error_status(status, LoadProfile.SMOKE) is True

    def test_smoke_other_4xx_is_still_error(self) -> None:
        for status in (400, 404, 422):
            assert is_error_status(status, LoadProfile.SMOKE) is True

    def test_transport_failure_is_always_error(self) -> None:
        for status in (None, 0):
            assert is_error_status(status, LoadProfile.VOLUME) is True
            assert is_error_status(status, LoadProfile.SMOKE) is True


class TestIsRateLimitExempt:
    def test_smoke_exempts_429_only(self) -> None:
        assert is_rate_limit_exempt(429, LoadProfile.SMOKE) is True
        for status in (200, 404, 422, 500):
            assert is_rate_limit_exempt(status, LoadProfile.SMOKE) is False

    def test_volume_never_exempts(self) -> None:
        assert is_rate_limit_exempt(429, LoadProfile.VOLUME) is False

    def test_transport_failure_never_exempt(self) -> None:
        assert is_rate_limit_exempt(None, LoadProfile.SMOKE) is False


class TestSnapshotRunStats:
    def test_adapts_entries(self) -> None:
        stats = _FakeStats(
            [
                _entry("/health", 10, failures=1, p95=90),
                _entry("/query", 20, p95=1500),
            ]
        )
        snapshot = snapshot_run_stats(stats)
        by_name = {s.name: s for s in snapshot}
        assert by_name["/health"].num_requests == 10
        assert by_name["/health"].num_failures == 1
        assert by_name["/health"].p95_response_time_ms == 90
        assert by_name["/query"].p95_response_time_ms == 1500

    def test_zero_request_entry_has_none_p95(self) -> None:
        stats = _FakeStats([_entry("/dive", 0)])
        snapshot = snapshot_run_stats(stats)
        assert snapshot[0].p95_response_time_ms is None

    def test_unknown_endpoint_raises(self) -> None:
        stats = _FakeStats([_entry("/ingest", 5)])
        with pytest.raises(ValueError, match="unexpected endpoint"):
            snapshot_run_stats(stats)


class TestEvaluateRun:
    def test_all_gates_pass(self) -> None:
        stats = [
            _stats_of("/health", 100, p95=50),
            _stats_of("/query", 300, p95=1500),
            _stats_of("/dive", 100, p95=3000),
        ]
        verdict = evaluate_run(stats, LoadProfile.VOLUME)
        assert verdict.overall_ok is True
        assert verdict.error_rate == 0.0
        assert all(gate.ok for gate in verdict.gates)

    def test_run_level_error_rate_at_ceiling_still_ok(self) -> None:
        stats = [
            _stats_of("/health", 100, failures=1),
            _stats_of("/query", 100, failures=1, p95=1500),
        ]
        verdict = evaluate_run(stats, LoadProfile.VOLUME)
        assert verdict.total_requests == 200
        assert verdict.error_rate == 0.01  # exactly at the run ceiling is ok
        assert verdict.error_rate_ok is True
        assert verdict.overall_ok is True

    def test_error_rate_just_over_ceiling_fails(self) -> None:
        stats = [
            _stats_of("/health", 100, failures=2),
            _stats_of("/query", 99, failures=0, p95=1500),
        ]
        verdict = evaluate_run(stats, LoadProfile.VOLUME)
        assert verdict.total_requests == 199
        assert verdict.error_rate > 0.01
        assert verdict.error_rate_ok is False
        assert verdict.overall_ok is False

    def test_error_rate_is_run_level_not_per_endpoint(self) -> None:
        # /dive carries 2% errors on its own, but the run-level rate stays
        # under the ceiling — the gate is run-level only (decision #271).
        stats = [
            _stats_of("/health", 100),
            _stats_of("/dive", 100, failures=2, p95=3000),
        ]
        verdict = evaluate_run(stats, LoadProfile.VOLUME)
        assert verdict.total_failures == 2
        assert verdict.error_rate == 0.01
        assert verdict.error_rate_ok is True
        assert verdict.overall_ok is True

    def test_p95_over_ceiling_fails(self) -> None:
        stats = [_stats_of("/query", 100, p95=2500)]
        verdict = evaluate_run(stats, LoadProfile.VOLUME)
        gate = verdict.gates[0]
        assert gate.p95_ok is False
        assert gate.ok is False
        assert verdict.overall_ok is False

    def test_smoke_p95_ceiling_is_looser(self) -> None:
        stats = [_stats_of("/dive", 100, p95=20000)]
        smoke = evaluate_run(stats, LoadProfile.SMOKE)
        assert smoke.gates[0].p95_ok is True
        volume = evaluate_run(stats, LoadProfile.VOLUME)
        assert volume.gates[0].p95_ok is False

    def test_no_samples_endpoint_is_warning_not_failure(self) -> None:
        stats = [
            _stats_of("/health", 100, p95=50),
            _stats_of("/dive", 0, p95=None),
        ]
        verdict = evaluate_run(stats, LoadProfile.VOLUME)
        dive = next(g for g in verdict.gates if g.endpoint == "/dive")
        assert dive.p95_ok is None
        assert dive.ok is True
        assert verdict.overall_ok is True

    def test_run_with_zero_total_requests_fails(self) -> None:
        verdict = evaluate_run([], LoadProfile.VOLUME)
        assert verdict.overall_ok is False

    def test_verdict_returns_gate_results_in_input_order(self) -> None:
        stats = [_stats_of("/query", 10), _stats_of("/health", 10)]
        verdict = evaluate_run(stats, LoadProfile.VOLUME)
        assert [g.endpoint for g in verdict.gates] == ["/query", "/health"]
