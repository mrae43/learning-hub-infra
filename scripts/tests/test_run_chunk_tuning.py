"""Tests for scripts/run_chunk_tuning.py."""

from __future__ import annotations

import json
from pathlib import Path
from uuid import UUID, uuid4

import pytest

from core.types.responses import RetrievalResult, ScoredChunk
from scripts.run_chunk_tuning import (
    RETRIEVAL_MODES,
    ConfigResult,
    QueryEval,
    build_results_json,
    check_sidecars,
    compute_config_metrics,
    determine_winner,
    format_results_table,
    score_retrieval_result,
)

# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_chunk(text: str, score: float = 1.0, chunk_id: UUID | None = None) -> ScoredChunk:
    return ScoredChunk(
        chunk_id=chunk_id or uuid4(),
        text=text,
        score=score,
    )


def _make_retrieval_result(
    dense_texts: list[str] | None = None,
    sparse_texts: list[str] | None = None,
    fused_texts: list[str] | None = None,
) -> RetrievalResult:
    return RetrievalResult(
        dense=[_make_chunk(t) for t in (dense_texts or [])],
        sparse=[_make_chunk(t) for t in (sparse_texts or [])],
        fused=[_make_chunk(t) for t in (fused_texts or [])],
    )


# ── check_sidecars ───────────────────────────────────────────────────────────


class TestCheckSidecars:
    def test_all_present(self, tmp_path: Path) -> None:
        for name in ("256_10", "512_15", "1024_20"):
            (tmp_path / f"eval_vectors_{name}.json").write_text("{}")
        missing = check_sidecars(tmp_path, frozenset({"256_10", "512_15", "1024_20"}))
        assert missing == []

    def test_one_missing(self, tmp_path: Path) -> None:
        (tmp_path / "eval_vectors_256_10.json").write_text("{}")
        (tmp_path / "eval_vectors_1024_20.json").write_text("{}")
        missing = check_sidecars(tmp_path, frozenset({"256_10", "512_15", "1024_20"}))
        assert missing == ["512_15"]

    def test_all_missing(self, tmp_path: Path) -> None:
        missing = check_sidecars(tmp_path, frozenset({"256_10", "512_15", "1024_20"}))
        assert sorted(missing) == ["1024_20", "256_10", "512_15"]

    def test_empty_config_set(self, tmp_path: Path) -> None:
        missing = check_sidecars(tmp_path, frozenset())
        assert missing == []


# ── score_retrieval_result ────────────────────────────────────────────────────


class TestScoreRetrievalResult:
    def test_all_modes_scored(self) -> None:
        result = _make_retrieval_result(
            dense_texts=["alpha", "beta", "gamma"],
            sparse_texts=["delta", "epsilon"],
            fused_texts=["zeta", "eta", "theta", "iota"],
        )
        hits = score_retrieval_result(result, "eta", top_k=3)
        assert set(hits.keys()) == {"dense", "sparse", "fused"}
        assert len(hits["dense"]) == 3
        assert len(hits["sparse"]) == 3
        assert len(hits["fused"]) == 3

    def test_hit_in_fused(self) -> None:
        result = _make_retrieval_result(
            dense_texts=["not relevant"],
            sparse_texts=["also not"],
            fused_texts=["miss", "eta is here", "also miss"],
        )
        hits = score_retrieval_result(result, "eta is here", top_k=3)
        assert hits["fused"] == [False, True, False]
        assert hits["dense"] == [False, False, False]
        assert hits["sparse"] == [False, False, False]

    def test_hit_in_dense(self) -> None:
        result = _make_retrieval_result(
            dense_texts=["target text", "other"],
            sparse_texts=["wrong"],
            fused_texts=["wrong"],
        )
        hits = score_retrieval_result(result, "target text", top_k=2)
        assert hits["dense"] == [True, False]

    def test_hit_in_sparse(self) -> None:
        result = _make_retrieval_result(
            dense_texts=["x"],
            sparse_texts=["found it here"],
            fused_texts=["y"],
        )
        hits = score_retrieval_result(result, "found it here", top_k=2)
        assert hits["sparse"] == [True, False]

    def test_partial_hits(self) -> None:
        result = _make_retrieval_result(
            dense_texts=["a", "b", "c", "d", "e"],
            sparse_texts=[],
            fused_texts=["a", "b", "c", "d", "e"],
        )
        hits = score_retrieval_result(result, "c", top_k=5)
        assert hits["dense"] == [False, False, True, False, False]

    def test_empty_results(self) -> None:
        result = _make_retrieval_result()
        hits = score_retrieval_result(result, "anything", top_k=3)
        assert hits["dense"] == [False, False, False]
        assert hits["sparse"] == [False, False, False]
        assert hits["fused"] == [False, False, False]

    def test_fewer_results_than_top_k_padded_with_false(self) -> None:
        result = _make_retrieval_result(
            dense_texts=["hit"],
            fused_texts=["hit"],
        )
        hits = score_retrieval_result(result, "hit", top_k=5)
        assert hits["dense"] == [True, False, False, False, False]
        assert hits["fused"] == [True, False, False, False, False]


# ── compute_config_metrics ────────────────────────────────────────────────────


def _qe(
    query: str = "q",
    stratum: str = "concept_lookup",
    hits: dict[str, list[bool]] | None = None,
) -> QueryEval:
    return QueryEval(query=query, stratum=stratum, hits=hits or {})


class TestComputeConfigMetrics:
    def test_single_query_all_hits(self) -> None:
        evals = [
            _qe(hits={"dense": [True, True], "sparse": [True, False], "fused": [True, True]}),
        ]
        result = compute_config_metrics("256_10", evals, top_k=2)
        assert result.config_key == "256_10"
        assert result.query_count == 1
        assert result.overall["dense"]["recall@2"] == 1.0
        assert result.overall["dense"]["mrr"] == 1.0
        assert result.overall["fused"]["recall@2"] == 1.0
        assert result.overall["sparse"]["recall@2"] == 0.5

    def test_single_query_no_hits(self) -> None:
        evals = [
            _qe(hits={"dense": [False, False], "sparse": [False, False], "fused": [False, False]}),
        ]
        result = compute_config_metrics("512_15", evals, top_k=2)
        assert result.overall["fused"]["recall@2"] == 0.0
        assert result.overall["fused"]["mrr"] == 0.0

    def test_multiple_queries_averages(self) -> None:
        evals = [
            _qe(hits={"dense": [True, True], "sparse": [True, True], "fused": [True, True]}),
            _qe(hits={"dense": [False, False], "sparse": [False, False], "fused": [False, False]}),
        ]
        result = compute_config_metrics("1024_20", evals, top_k=2)
        assert result.overall["fused"]["recall@2"] == 0.5
        assert result.overall["dense"]["recall@2"] == 0.5
        assert result.overall["sparse"]["recall@2"] == 0.5

    def test_mrr_first_hit_early_ranks_higher(self) -> None:
        mrr1 = 1.0 / 1  # first hit at rank 1
        mrr2 = 1.0 / 3  # first hit at rank 3
        misses = [False, False, False]
        evals = [
            _qe(hits={"dense": misses, "sparse": misses, "fused": [True, False, False]}),
            _qe(hits={"dense": misses, "sparse": misses, "fused": [False, False, True]}),
        ]
        result = compute_config_metrics("256_10", evals, top_k=3)
        expected_mrr = (mrr1 + mrr2) / 2
        assert result.overall["fused"]["mrr"] == pytest.approx(expected_mrr)

    def test_per_stratum_breakdown(self) -> None:
        misses = [False, False]
        evals = [
            _qe(
                stratum="concept_lookup",
                hits={"dense": misses, "sparse": misses, "fused": [True, False]},
            ),
            _qe(
                stratum="concept_lookup",
                hits={"dense": misses, "sparse": misses, "fused": [True, True]},
            ),
            _qe(
                stratum="exact_match",
                hits={"dense": misses, "sparse": misses, "fused": [False, False]},
            ),
        ]
        result = compute_config_metrics("256_10", evals, top_k=2)
        assert result.per_stratum["concept_lookup"]["fused"]["recall@2"] == 0.75
        assert result.per_stratum["exact_match"]["fused"]["recall@2"] == 0.0
        assert len(result.per_stratum) == 2

    def test_stratum_all_modes_present(self) -> None:
        evals = [
            _qe(
                stratum="multi_hop",
                hits={
                    "dense": [True, False],
                    "sparse": [False, False],
                    "fused": [True, False],
                },
            ),
        ]
        result = compute_config_metrics("256_10", evals, top_k=2)
        for mode in RETRIEVAL_MODES:
            assert mode in result.per_stratum["multi_hop"]

    def test_empty_query_list(self) -> None:
        result = compute_config_metrics("256_10", [], top_k=2)
        assert result.query_count == 0
        for mode in RETRIEVAL_MODES:
            assert result.overall[mode]["recall@2"] == 0.0
            assert result.overall[mode]["mrr"] == 0.0
        assert result.per_stratum == {}


# ── determine_winner ─────────────────────────────────────────────────────────


def _config_result(
    key: str,
    fused_recall: float,
    fused_mrr: float = 0.0,
) -> ConfigResult:
    recall_key = "recall@10"
    overall: dict[str, dict[str, float]] = {
        mode: {recall_key: 0.0, "mrr": 0.0} for mode in RETRIEVAL_MODES
    }
    overall["fused"][recall_key] = fused_recall
    overall["fused"]["mrr"] = fused_mrr
    return ConfigResult(
        config_key=key,
        query_count=1,
        overall=overall,
        per_stratum={},
        per_query_hits=[],
    )


class TestDetermineWinner:
    def test_clear_winner(self) -> None:
        results = [
            _config_result("256_10", fused_recall=0.80),
            _config_result("512_15", fused_recall=0.92),
            _config_result("1024_20", fused_recall=0.88),
        ]
        winner, runner_up = determine_winner(results, top_k=10)
        assert winner == "512_15"
        # 0.92 - 0.88 = 0.04 > 0.02, so no runner-up
        assert runner_up is None

    def test_tie_break_by_mrr(self) -> None:
        """When configs are within 0.02 of max recall, MRR decides."""
        results = [
            _config_result("256_10", fused_recall=0.90, fused_mrr=0.70),
            _config_result("512_15", fused_recall=0.91, fused_mrr=0.85),
            _config_result("1024_20", fused_recall=0.90, fused_mrr=0.80),
        ]
        winner, runner_up = determine_winner(results, top_k=10)
        assert winner == "512_15"
        # All three are within 0.02 of max (0.91). MRR ranking: 512_15 > 1024_20 > 256_10
        assert runner_up == "1024_20"

    def test_within_threshold_tie_break(self) -> None:
        """Both configs within 0.02 of max recall; higher MRR wins."""
        results = [
            _config_result("256_10", fused_recall=0.90, fused_mrr=0.70),
            _config_result("512_15", fused_recall=0.89, fused_mrr=0.95),
        ]
        winner, runner_up = determine_winner(results, top_k=10)
        # Max recall=0.90. Both within 0.02. 512_15 has higher MRR.
        assert winner == "512_15"
        assert runner_up == "256_10"

    def test_outside_threshold_no_runner_up(self) -> None:
        results = [
            _config_result("256_10", fused_recall=0.95, fused_mrr=0.70),
            _config_result("512_15", fused_recall=0.80, fused_mrr=0.95),
        ]
        winner, runner_up = determine_winner(results, top_k=10)
        assert winner == "256_10"
        assert runner_up is None

    def test_tie_winner_top_recall(self) -> None:
        results = [
            _config_result("256_10", fused_recall=0.88),
            _config_result("512_15", fused_recall=0.88, fused_mrr=0.9),
            _config_result("1024_20", fused_recall=0.88, fused_mrr=0.8),
        ]
        winner, runner_up = determine_winner(results, top_k=10)
        assert winner == "512_15"  # highest MRR among ties
        assert runner_up == "1024_20"

    def test_single_config(self) -> None:
        results = [_config_result("256_10", fused_recall=0.75)]
        winner, runner_up = determine_winner(results, top_k=10)
        assert winner == "256_10"
        assert runner_up is None

    def test_empty_results(self) -> None:
        winner, runner_up = determine_winner([], top_k=10)
        assert winner == ""
        assert runner_up is None


# ── format_results_table ──────────────────────────────────────────────────────


class TestFormatResultsTable:
    def test_output_contains_all_configs(self) -> None:
        results = [
            _config_result("256_10", fused_recall=0.80, fused_mrr=0.70),
            _config_result("512_15", fused_recall=0.90, fused_mrr=0.85),
        ]
        output = format_results_table(results, "512_15", top_k=10)
        assert "256_10" in output
        assert "512_15" in output
        assert "Chunk-Size Tuning Results" in output

    def test_winner_marked(self) -> None:
        results = [
            _config_result("256_10", fused_recall=0.80),
            _config_result("512_15", fused_recall=0.90),
        ]
        output = format_results_table(results, "512_15", top_k=10)
        lines = output.split("\n")
        # Find the line with 512_15 and check it has the checkmark
        for line in lines:
            if line.strip().startswith("512_15"):
                assert "✓" in line

    def test_no_winner_shows_message(self) -> None:
        results = [
            _config_result("256_10", fused_recall=0.80),
        ]
        output = format_results_table(results, None, top_k=10)
        assert "No winner could be determined" in output

    def test_per_stratum_section_present(self) -> None:
        recall_key = "recall@10"
        overall: dict[str, dict[str, float]] = {
            m: {recall_key: 0.0, "mrr": 0.0} for m in RETRIEVAL_MODES
        }
        result = ConfigResult(
            config_key="256_10",
            query_count=1,
            overall=overall,
            per_stratum={
                "concept_lookup": {"fused": {recall_key: 0.5, "mrr": 0.3}},
                "exact_match": {"fused": {recall_key: 0.7, "mrr": 0.4}},
            },
            per_query_hits=[],
        )
        output = format_results_table([result], "256_10", top_k=10)
        assert "Per-Stratum Fused" in output
        assert "concept_lookup" in output
        assert "exact_match" in output


# ── build_results_json ────────────────────────────────────────────────────────


class TestBuildResultsJson:
    def test_structure(self) -> None:
        results = [
            _config_result("256_10", fused_recall=0.80, fused_mrr=0.70),
            _config_result("512_15", fused_recall=0.90, fused_mrr=0.85),
        ]
        data = build_results_json(results, "512_15", top_k=10)
        assert data["top_k"] == 10
        assert data["winner"] == "512_15"
        assert len(data["configs"]) == 2
        assert data["configs"][0]["config_key"] == "256_10"
        assert data["configs"][1]["config_key"] == "512_15"

    def test_empty_winner(self) -> None:
        data = build_results_json([], None, top_k=5)
        assert data["winner"] == ""
        assert data["configs"] == []

    def test_serializable(self) -> None:
        results = [
            _config_result("256_10", fused_recall=0.80, fused_mrr=0.70),
        ]
        data = build_results_json(results, "256_10", top_k=10)
        json.dumps(data)  # should not raise
