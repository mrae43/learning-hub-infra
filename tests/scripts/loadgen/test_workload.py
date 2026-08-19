"""Tests for scripts/loadgen/workload.py."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from scripts.loadgen.workload import RoundRobin, load_corpus_queries, load_dive_passages


class TestRoundRobin:
    def test_cycles_through_items(self) -> None:
        rr = RoundRobin(["a", "b", "c"])
        assert [rr.next() for _ in range(5)] == ["a", "b", "c", "a", "b"]

    def test_single_item_repeats(self) -> None:
        rr = RoundRobin(["only"])
        assert [rr.next() for _ in range(3)] == ["only", "only", "only"]

    def test_empty_items_raise(self) -> None:
        with pytest.raises(ValueError, match="at least one item"):
            RoundRobin([])


class TestLoadCorpusQueries:
    def test_loads_query_strings_in_order(self, tmp_path: Path) -> None:
        path = tmp_path / "eval_set.yaml"
        path.write_text(
            yaml.safe_dump(
                {
                    "queries": [
                        {"query": "first", "stratum": "concept_lookup"},
                        {"query": "second", "stratum": "exact_match"},
                    ]
                }
            )
        )
        assert load_corpus_queries(path) == ["first", "second"]

    def test_missing_queries_key_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "eval_set.yaml"
        path.write_text(yaml.safe_dump({"other": []}))
        with pytest.raises(ValueError, match="no queries"):
            load_corpus_queries(path)

    def test_empty_queries_raise(self, tmp_path: Path) -> None:
        path = tmp_path / "eval_set.yaml"
        path.write_text(yaml.safe_dump({"queries": []}))
        with pytest.raises(ValueError, match="no queries"):
            load_corpus_queries(path)

    def test_entries_without_query_strings_raise(self, tmp_path: Path) -> None:
        path = tmp_path / "eval_set.yaml"
        path.write_text(yaml.safe_dump({"queries": [{"stratum": "exact_match"}]}))
        with pytest.raises(ValueError, match="no query strings"):
            load_corpus_queries(path)

    def test_real_corpus_has_queries(self) -> None:
        queries = load_corpus_queries()
        assert len(queries) >= 50


class TestLoadDivePassages:
    def test_loads_passage_dicts(self, tmp_path: Path) -> None:
        path = tmp_path / "passages.json"
        path.write_text(
            json.dumps(
                [
                    {"passage_type": "text", "content": "one", "source": "docs/a.md"},
                    {"passage_type": "text", "content": "two", "source": "docs/b.md"},
                ]
            )
        )
        passages = load_dive_passages(path)
        assert [p["content"] for p in passages] == ["one", "two"]
        assert all(p["passage_type"] == "text" for p in passages)

    def test_non_text_passage_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "passages.json"
        path.write_text(
            json.dumps([{"passage_type": "code", "content": "x", "source": "docs/a.md"}])
        )
        with pytest.raises(ValueError, match="must be a text passage"):
            load_dive_passages(path)

    def test_empty_list_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "passages.json"
        path.write_text("[]")
        with pytest.raises(ValueError, match="no dive passage fixtures"):
            load_dive_passages(path)

    def test_non_list_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "passages.json"
        path.write_text('{"passage_type": "text"}')
        with pytest.raises(ValueError, match="no dive passage fixtures"):
            load_dive_passages(path)

    def test_fixtures_are_text_passages(self) -> None:
        passages = load_dive_passages()
        assert len(passages) >= 5
        for passage in passages:
            assert passage["passage_type"] == "text"
            assert passage["content"]
