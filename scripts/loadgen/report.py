"""Plain-text run report for the load generator.

Formats a :class:`RunVerdict` into a human-readable table showing, per
endpoint, the request/error counts and measured p95 against its ceiling, plus
the run-level error rate. Printed by the locustfile at shutdown and usable as
the single "did the run pass" artifact the runbook keys on.
"""

from __future__ import annotations

from scripts.loadgen.config import ENDPOINTS, MAX_ERROR_RATE
from scripts.loadgen.gates import GateResult, RunVerdict

_SEPARATOR = "=" * 74


def _gate_row(gate: GateResult) -> str:
    p95 = "n/a" if gate.p95_ms is None else f"{gate.p95_ms:.0f} ms"
    p95_verdict = {
        None: "WARN (no samples)",
        True: "PASS",
        False: "FAIL",
    }[gate.p95_ok]
    return (
        f"{gate.endpoint:<10} {gate.num_requests:>8} {gate.num_failures:>7} "
        f"{gate.error_rate * 100:>7.2f}% {p95:>10} {gate.p95_ceiling_ms:>10} ms "
        f"{p95_verdict:>10}"
    )


def format_report(verdict: RunVerdict) -> str:
    """Render the run report.

    Args:
        verdict: The evaluated run.

    Returns:
        The full report text, ending with a newline.
    """
    by_endpoint = {gate.endpoint: gate for gate in verdict.gates}
    lines = [
        _SEPARATOR,
        f"Learning Hub load run - profile: {verdict.profile.value}",
        _SEPARATOR,
        f"Total requests: {verdict.total_requests}   errors: {verdict.total_failures}   "
        f"error rate: {verdict.error_rate * 100:.2f}%  (ceiling {MAX_ERROR_RATE * 100:.0f}%)",
        f"Error-rate gate: {'PASS' if verdict.error_rate_ok else 'FAIL'}",
        "",
        f"{'Endpoint':<10} {'Requests':>8} {'Errors':>7} {'Error%':>9} {'p95':>11} "
        f"{'p95 ceiling':>12} {'p95 gate':>10}",
    ]
    for endpoint in ENDPOINTS:
        if endpoint in by_endpoint:
            lines.append(_gate_row(by_endpoint[endpoint]))
    lines.extend(
        [
            "",
            f"Overall: {'PASS' if verdict.overall_ok else 'FAIL'}",
            _SEPARATOR,
        ]
    )
    return "\n".join(lines) + "\n"
