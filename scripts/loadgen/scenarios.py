"""Locust user scenarios for the load generator (decision #271).

``/query`` (75%) and ``/dive`` (25%) are weighted tasks on :class:`LoadUser`;
``/health`` is a separate low-rate liveness user (:class:`HealthUser`) that
keeps a steady ~0.2-1 rps health stream without diluting the 75/25 mix. The
``LOAD_PROFILE`` env var is read once at import so both the error
classification and the smoke budget are stable for the run.
"""

from __future__ import annotations

import logging
from typing import Any, Protocol

import gevent
from locust import HttpUser, between, task

from scripts.loadgen.budget import RequestBudget
from scripts.loadgen.config import (
    SMOKE_UPSTREAM_REQUEST_CAP,
    LoadProfile,
    load_config,
)
from scripts.loadgen.gates import is_error_status, is_rate_limit_exempt
from scripts.loadgen.workload import RoundRobin, load_corpus_queries, load_dive_passages

logger = logging.getLogger(__name__)

PROFILE = load_config()
"""The active run profile, read once at import (default ``volume``)."""

_QUERIES = RoundRobin(load_corpus_queries())
"""Round-robin over all corpus queries, shared by every :class:`LoadUser`."""

_PASSAGES = RoundRobin(load_dive_passages())
"""Round-robin over the curated dive fixtures, shared by every :class:`LoadUser`."""

_REQUEST_TIMEOUT_SECONDS = 60
"""Per-request timeout; dive can legitimately take tens of seconds in smoke."""


class LocustResponse(Protocol):
    """The ``catch_response=True`` surface scenarios mark responses with."""

    status_code: int

    def failure(self, exc: str) -> None: ...

    def success(self) -> None: ...


def _record_response(resp: LocustResponse, profile: LoadProfile) -> None:
    """Mark a response as success/failure against the profile's error rules.

    Args:
        resp: The Locust ``ResponseContextManager`` from ``catch_response=True``.
        profile: The run profile controlling error classification.
    """
    status = resp.status_code
    if is_error_status(status, profile):
        reason = f"status {status}" if status else "transport error"
        resp.failure(reason)
    else:
        if is_rate_limit_exempt(status, profile):
            logger.warning("smoke run: 429 rate limit exempted (not counted as an error)")
        resp.success()


class LoadUser(HttpUser):
    """Weighted ``/query`` (75%) / ``/dive`` (25%) user scenario.

    ``weight`` makes this class dominate user spawning (Locust's deterministic
    KL dispatcher gives ~75% of spawned users to the weight-3 class), so a
    single-user smoke run always spawns the upstream-backed user rather than
    only the liveness user.
    """

    wait_time = between(1, 3)
    weight = 3
    smoke_budget: RequestBudget = RequestBudget(
        SMOKE_UPSTREAM_REQUEST_CAP if PROFILE is LoadProfile.SMOKE else None
    )

    @task(3)
    def query(self) -> None:
        """Send a ``POST /query`` with the next round-robin corpus query."""
        self._send_upstream_request("/query", {"query": _QUERIES.next()})

    @task(1)
    def dive(self) -> None:
        """Send a ``POST /dive`` with the next round-robin passage fixture."""
        payload = {
            "captured_passage": _PASSAGES.next(),
            "requested_output_type": "interactive_animation",
        }
        self._send_upstream_request("/dive", payload)

    def _send_upstream_request(self, path: str, body: dict[str, Any]) -> None:
        with self.client.post(
            path, json=body, catch_response=True, timeout=_REQUEST_TIMEOUT_SECONDS
        ) as resp:
            _record_response(resp, PROFILE)
        if self.smoke_budget.spend():
            # Defer the quit to a fresh greenlet: a direct runner.quit() from
            # inside this user's task can deadlock while Locust stops the user
            # greenlets (it would be joining on itself), leaving the run up.
            gevent.spawn_later(0, self.environment.runner.quit)


class HealthUser(HttpUser):
    """Low-rate ``/health`` liveness user (~0.2-1 rps per user).

    A small ``weight`` keeps health traffic at a constant low rate without
    diluting the 75/25 ``/query``-``/dive`` mix; it is separate from the
    weighted user scenarios (decision #271).
    """

    wait_time = between(1, 5)
    weight = 1

    @task
    def health(self) -> None:
        """Send a ``GET /health`` liveness probe."""
        with self.client.get(
            "/health", catch_response=True, timeout=_REQUEST_TIMEOUT_SECONDS
        ) as resp:
            _record_response(resp, PROFILE)
