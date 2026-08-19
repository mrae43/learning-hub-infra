"""Tests for scripts/loadgen/report.py."""

from __future__ import annotations

from scripts.loadgen.config import Endpoint, LoadProfile
from scripts.loadgen.gates import EndpointStats, RunVerdict, evaluate_run
from scripts.loadgen.report import format_report


def _stats(
    name: Endpoint, requests: int, failures: int = 0, p95: float | None = 50.0
) -> EndpointStats:
    return EndpointStats(
        name=name,
        num_requests=requests,
        num_failures=failures,
        p95_response_time_ms=p95,
    )


def _verdict(profile: LoadProfile, *stats: EndpointStats) -> RunVerdict:
    return evaluate_run(list(stats), profile)


class TestFormatReport:
    def test_passing_run_reports_all_sections(self) -> None:
        verdict = _verdict(
            LoadProfile.VOLUME,
            _stats("/health", 100, p95=50),
            _stats("/query", 300, p95=1500),
            _stats("/dive", 100, p95=3000),
        )
        report = format_report(verdict)
        assert "profile: volume" in report
        assert "Total requests: 500" in report
        assert "error rate: 0.00%  (ceiling 1%)" in report
        assert "Error-rate gate: PASS" in report
        for endpoint in ("/health", "/query", "/dive"):
            assert endpoint in report
        assert "Overall: PASS" in report

    def test_failing_error_rate_reports_fail(self) -> None:
        verdict = _verdict(
            LoadProfile.SMOKE,
            _stats("/health", 50),
            _stats("/query", 50, failures=5, p95=5000),
        )
        report = format_report(verdict)
        assert "Error-rate gate: FAIL" in report
        assert "Overall: FAIL" in report

    def test_p95_over_ceiling_reports_fail(self) -> None:
        verdict = _verdict(
            LoadProfile.VOLUME,
            _stats("/query", 100, p95=2500),
        )
        report = format_report(verdict)
        assert "Overall: FAIL" in report

    def test_no_samples_endpoint_flagged_as_warning(self) -> None:
        verdict = _verdict(
            LoadProfile.VOLUME,
            _stats("/health", 100),
            _stats("/dive", 0, p95=None),
        )
        report = format_report(verdict)
        assert "WARN (no samples)" in report
        assert "n/a" in report
        assert "Overall: PASS" in report

    def test_smoke_header_and_ceiling(self) -> None:
        verdict = _verdict(LoadProfile.SMOKE, _stats("/dive", 100, p95=20000))
        report = format_report(verdict)
        assert "profile: smoke" in report
        assert "30000 ms" in report
        assert "Overall: PASS" in report
