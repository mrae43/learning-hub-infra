"""Guard the 1:1 alert-rules-to-runbook mapping (issue #292).

docs/runbook.md must have exactly one entry per Prometheus alert rule in
observability/prometheus/alerts.yml, keyed by the alert name. Drift in either
direction — a rule without a runbook entry, or a runbook entry without a rule
— fails here instead of surfacing during an incident.
"""

import re
from pathlib import Path

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
_ALERTS_FILE = _REPO_ROOT / "observability" / "prometheus" / "alerts.yml"
_RUNBOOK_FILE = _REPO_ROOT / "docs" / "runbook.md"

EXPECTED_ALERTS = {
    "ServiceDown",
    "Upstream502503",
    "P95LatencyHigh",
    "Internal5xx",
    "IngestionFailed",
}


def _alert_names() -> list[str]:
    """Return the rule names in ``alerts.yml``, in file order."""
    payload = yaml.safe_load(_ALERTS_FILE.read_text())
    return [rule["alert"] for group in payload["groups"] for rule in group["rules"]]


def _runbook_entries() -> list[str]:
    """Return the alert entries in ``runbook.md``, in file order."""
    text = _RUNBOOK_FILE.read_text()
    return re.findall(r"^## ([A-Za-z0-9]+)\s*$", text, flags=re.MULTILINE)


def test_alerts_file_has_exactly_the_five_expected_rules() -> None:
    """The rule set is exactly the five alerts decided in issue #272."""
    names = _alert_names()
    assert len(names) == 5
    assert set(names) == EXPECTED_ALERTS


def test_every_alert_rule_carries_an_expr_and_severity() -> None:
    """Each rule is queryable and routed (every alert needs a severity)."""
    payload = yaml.safe_load(_ALERTS_FILE.read_text())
    for group in payload["groups"]:
        for rule in group["rules"]:
            assert rule["expr"].strip()
            assert rule.get("labels", {}).get("severity")
            assert rule.get("annotations", {}).get("summary")


def test_runbook_has_exactly_one_entry_per_alert() -> None:
    """The runbook maps each alert one-to-one to an incident response.

    Structural headings (``## Overview``) are ignored; every alert must appear
    as a heading exactly once.
    """
    names = _alert_names()
    entries = _runbook_entries()
    alert_entries = [entry for entry in entries if entry in EXPECTED_ALERTS]
    assert set(alert_entries) == set(names), "runbook headings must match alert names"
    assert len(alert_entries) == len(set(names)) == 5, "exactly one runbook entry per alert"
