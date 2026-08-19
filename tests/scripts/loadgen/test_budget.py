"""Tests for scripts/loadgen/budget.py."""

from __future__ import annotations

from scripts.loadgen.budget import RequestBudget


class TestRequestBudget:
    def test_uncapped_budget_never_triggers(self) -> None:
        budget = RequestBudget(None)
        for _ in range(10_000):
            assert budget.spend() is False

    def test_capped_budget_triggers_at_cap(self) -> None:
        budget = RequestBudget(3)
        assert budget.spend() is False
        assert budget.spend() is False
        assert budget.spend() is True
        assert budget.spend() is True  # stays exhausted

    def test_zero_cap_triggers_immediately(self) -> None:
        budget = RequestBudget(0)
        assert budget.spend() is True
