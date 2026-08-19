"""Shared request budget for a load run (decision #271).

``/query`` and ``/dive`` requests spend real API budget in smoke runs, so the
run stops once the upstream-backed cap is spent. ``/health`` never spends.
"""

from __future__ import annotations

import threading


class RequestBudget:
    """Shared budget of upstream-backed requests for a run.

    Args:
        cap: Maximum upstream-backed requests, or ``None`` for no cap (volume).
    """

    def __init__(self, cap: int | None) -> None:
        self._cap = cap
        self._count = 0
        self._lock = threading.Lock()

    def spend(self) -> bool:
        """Record one upstream-backed request.

        Returns:
            ``True`` when the cap has just been reached (caller should stop
            the run); ``False`` otherwise or when uncapped.
        """
        if self._cap is None:
            return False
        with self._lock:
            self._count += 1
            return self._count >= self._cap
