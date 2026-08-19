"""Run-profile configuration for the Locust load generator.

One locustfile, two run profiles selected by the ``LOAD_PROFILE`` env var
(decision #271): ``volume`` (mock upstream, sustained) and ``smoke`` (real
API, ~1 upstream user, budgeted). The env var is the generator's run-profile
knob only; the mock-vs-real upstream switch is the ``load`` compose profile's
``OPENAI_BASE_URL`` (decision #265).

The ceilings and caps below are the first-pass values from decision #271 —
shape what the alert rules key on, and get sharpened against smoke baselines.
"""

from __future__ import annotations

import os
from enum import StrEnum
from typing import Literal

Endpoint = Literal["/health", "/query", "/dive"]
"""The endpoints the load generator exercises (report order in ``ENDPOINTS``)."""


class LoadProfile(StrEnum):
    """Run profiles for the load generator (decision #271)."""

    VOLUME = "volume"
    SMOKE = "smoke"


DEFAULT_TARGET_URL = "http://localhost:8000"
"""Default api base URL when the user does not pass Locust's ``--host``."""

MAX_ERROR_RATE = 0.01
"""Run-level error-rate ceiling: a run fails if errors exceed 1% (decision #271)."""

SMOKE_UPSTREAM_REQUEST_CAP = 50
"""Smoke-run cap on total upstream-backed requests (``/query`` + ``/dive``).

Smoke runs hit the real hosted APIs and spend real budget, so the run stops
after this many upstream-backed requests regardless of any ``--run-time``
(decision #271). ``/health`` is not upstream-backed and does not count.
"""

P95_CEILINGS_MS: dict[LoadProfile, dict[Endpoint, int]] = {
    LoadProfile.VOLUME: {"/health": 100, "/query": 2000, "/dive": 5000},
    LoadProfile.SMOKE: {"/health": 100, "/query": 10_000, "/dive": 30_000},
}
"""p95 latency ceilings in milliseconds, per endpoint and profile (decision #271)."""

DEFAULT_USERS: dict[LoadProfile, int] = {
    LoadProfile.VOLUME: 10,
    # One upstream-backed user (the ~1 user of decision #271) plus one
    # low-rate liveness user so the smoke run also exercises ``/health``.
    LoadProfile.SMOKE: 2,
}
"""User count per profile (decision #271)."""

DEFAULT_SPAWN_RATE: dict[LoadProfile, int] = {LoadProfile.VOLUME: 1, LoadProfile.SMOKE: 1}
"""Spawn rate per profile: smoke has no ramp-up (decision #271)."""

ENDPOINTS: tuple[Endpoint, ...] = ("/health", "/query", "/dive")
"""Endpoints the load generator exercises, in report order."""


def load_config() -> LoadProfile:
    """Read the ``LOAD_PROFILE`` env var.

    Returns:
        The selected :class:`LoadProfile`. Unknown values raise ``ValueError``
        so a typo fails fast instead of silently running the default.

    Raises:
        ValueError: If ``LOAD_PROFILE`` is set to anything but ``volume`` or
            ``smoke``.
    """
    raw = os.environ.get("LOAD_PROFILE", LoadProfile.VOLUME.value)
    try:
        return LoadProfile(raw.lower())
    except ValueError as exc:
        known = ", ".join(p.value for p in LoadProfile)
        raise ValueError(f"LOAD_PROFILE must be one of {known}; got {raw!r}") from exc
