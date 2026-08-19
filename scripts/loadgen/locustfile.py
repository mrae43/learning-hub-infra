"""Locust load generator entrypoint (decision #271, issue #290).

One locustfile drives both run profiles, selected by the ``LOAD_PROFILE`` env
var (default ``volume``):

- **volume** — sustained run against the mock upstream (``make load-up``);
  the default 10 users / 1 per second, runs until stopped.
- **smoke** — ~1 upstream user (plus a low-rate liveness user), no ramp-up,
  against the real API; stops after the upstream-backed request cap
  regardless of any ``--run-time``.

At shutdown the run is evaluated against the profile's error-rate and p95
ceilings, a plain-text report is printed, and the process exit code is set
(0 = pass, 1 = fail) so CI and the runbook can gate on it.

Run with e.g.::

    uv run --package scripts locust -f scripts/loadgen/locustfile.py \\
        --host http://localhost:8000 --headless
"""

from __future__ import annotations

import logging
from typing import Any

from locust import events

from scripts.loadgen.config import DEFAULT_SPAWN_RATE, DEFAULT_USERS
from scripts.loadgen.gates import evaluate_run, snapshot_run_stats
from scripts.loadgen.report import format_report
from scripts.loadgen.scenarios import PROFILE, HealthUser, LoadUser

logger = logging.getLogger(__name__)

__all__ = ["HealthUser", "LoadUser"]


def _apply_profile_defaults(environment: Any, **kwargs: Any) -> None:
    """Enforce the profile's user count and spawn rate.

    Smoke is ~1 upstream user with no ramp-up (decision #271); volume defaults
    to a small sustained pool. Locust's CLI parser stores the ``--users``
    option as ``num_users``; explicit ``--users``/``--spawn-rate`` flags are
    overridden so the ``LOAD_PROFILE`` knob alone determines the run shape.
    """
    if environment.parsed_options is None:
        return
    environment.parsed_options.num_users = DEFAULT_USERS[PROFILE]
    environment.parsed_options.spawn_rate = DEFAULT_SPAWN_RATE[PROFILE]


def _evaluate_run(environment: Any, **kwargs: Any) -> None:
    """Evaluate the run's gates, print the report, and set the exit code."""
    stats = snapshot_run_stats(environment.runner.stats)
    verdict = evaluate_run(stats, PROFILE)
    logger.info("Load run report:\n%s", format_report(verdict))
    environment.process_exit_code = 0 if verdict.overall_ok else 1


# Locust is an untyped third-party library, so attach listeners via its plain
# EventHook API (calling an Any method) instead of using the untyped
# ``@events.*.add_listener`` decorators. The two ``no-untyped-call`` ignores
# below are the same trade-off: locust ships no type stubs, so its EventHook
# is intentionally untyped.
events.init.add_listener(_apply_profile_defaults)  # type: ignore[no-untyped-call]
events.quitting.add_listener(_evaluate_run)  # type: ignore[no-untyped-call]
