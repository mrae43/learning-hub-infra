"""Workload sources for the load generator.

``/query`` round-robins all corpus queries from ``eval_corpus/eval_set_tuning.yaml``;
``/dive`` round-robins a curated list of ``TextPassage`` fixtures shipped in
``scripts/loadgen/fixtures/dive_passages.json`` (decision #271). Both sources
are wrapped in a lock-protected round-robin iterator so concurrent Locust
users never send the same payload at the same instant.

The dive fixtures are deliberately plain data shaped like a ``TextPassage``
request body rather than a Pydantic model import: the load generator is a
black-box HTTP client that sends JSON over the wire and must not import from
the application packages (decision #271).
"""

from __future__ import annotations

import json
import threading
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Literal, TypedDict

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
"""Repo root — ``scripts/loadgen/workload.py`` sits three levels below it."""

_CORPUS_QUERIES_YAML = _REPO_ROOT / "eval_corpus" / "eval_set_tuning.yaml"
_DIVE_FIXTURES_JSON = Path(__file__).resolve().parent / "fixtures" / "dive_passages.json"


class TextPassageFixture(TypedDict):
    """Shape of a curated ``/dive`` ``captured_passage`` payload."""

    passage_type: Literal["text"]
    content: str
    source: str


class RoundRobin[T]:
    """Thread-safe round-robin over a fixed item list.

    Args:
        items: The items to cycle through. Must be non-empty.

    Raises:
        ValueError: If ``items`` is empty — a cyclic workload source with
            nothing to emit is a configuration bug, not a runtime case.
    """

    def __init__(self, items: Iterable[T]) -> None:
        self._items = list(items)
        if not self._items:
            raise ValueError("RoundRobin requires at least one item")
        self._index = 0
        self._lock = threading.Lock()

    def next(self) -> T:
        """Return the next item, advancing the cursor.

        Returns:
            The item at the current cursor position.
        """
        with self._lock:
            item = self._items[self._index]
            self._index = (self._index + 1) % len(self._items)
            return item


def load_corpus_queries(path: Path = _CORPUS_QUERIES_YAML) -> list[str]:
    """Load the tuning corpus query strings.

    Args:
        path: The eval tuning YAML file (``eval_set_tuning.yaml`` by default).

    Returns:
        The list of ``query`` strings, in file order.

    Raises:
        ValueError: If the file has no ``queries`` list or it is empty.
    """
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    entries = data.get("queries") if isinstance(data, dict) else None
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"no queries found in {path}")
    queries = [
        entry["query"] for entry in entries if isinstance(entry, dict) and entry.get("query")
    ]
    if not queries:
        raise ValueError(f"no query strings found in {path}")
    return queries


def load_dive_passages(path: Path = _DIVE_FIXTURES_JSON) -> list[TextPassageFixture]:
    """Load the curated ``TextPassage`` dive fixtures.

    Args:
        path: The fixtures JSON file (``dive_passages.json`` by default).

    Returns:
        A list of passage fixtures, each already shaped as the
        ``captured_passage`` payload of a ``POST /dive`` request.

    Raises:
        ValueError: If the file is not a non-empty JSON list of objects, or an
            entry is not a well-formed text-passage fixture.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not data:
        raise ValueError(f"no dive passage fixtures found in {path}")
    passages = [entry for entry in data if isinstance(entry, dict)]
    if not passages:
        raise ValueError(f"no dive passage objects found in {path}")
    return [_as_text_passage_fixture(passage) for passage in passages]


def _as_text_passage_fixture(passage: dict[str, Any]) -> TextPassageFixture:
    """Validate one fixture entry into a :class:`TextPassageFixture`.

    Args:
        passage: A raw fixture entry from the fixtures JSON.

    Returns:
        The validated fixture.

    Raises:
        ValueError: If the entry is not a ``text`` passage with ``content``
            and ``source`` string fields.
    """
    if passage.get("passage_type") != "text":
        raise ValueError(f"fixture must be a text passage, got {passage.get('passage_type')!r}")
    content = passage.get("content")
    source = passage.get("source")
    if not isinstance(content, str) or not isinstance(source, str):
        raise ValueError("text passage fixture needs string 'content' and 'source'")
    return TextPassageFixture(passage_type="text", content=content, source=source)
