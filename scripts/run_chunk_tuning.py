#!/usr/bin/env python3
"""Top-level orchestrator for the chunk-size tuning harness.

Checks sidecar availability, seeds each config, runs all eval queries,
collects hit flags, computes fused metrics, determines the best config,
and writes results.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from core.clients.embeddings_client import Embedder, EmbeddingsClient, InMemoryEmbedder
from core.config.settings import Settings
from core.database.connection import get_engine
from core.types.responses import RetrievalResult
from core.types.retrieval_config import RetrievalConfig
from retrieval_qa.retrieval.query import retrieve_relevant_chunks
from scripts.eval_metrics import MRRMetric, RecallAtKMetric, is_hit
from scripts.seed_schema import CONFIG_NAMES, seed_schema, teardown_schema

_REPO_ROOT = Path(__file__).resolve().parent.parent
_TUNING_YAML = _REPO_ROOT / "eval_corpus" / "eval_set_tuning.yaml"
_EVAL_CORPUS_DIR = _REPO_ROOT / "eval_corpus"

DEFAULT_TOP_K = 10

RETRIEVAL_MODES = ("dense", "sparse", "fused")


@dataclass
class QueryEval:
    """Results of evaluating a single query for a single config."""

    query: str
    stratum: str
    hits: dict[str, list[bool]]


@dataclass
class ConfigResult:
    """Aggregated metrics for one chunk-size config."""

    config_key: str
    query_count: int
    overall: dict[str, dict[str, float]]
    per_stratum: dict[str, dict[str, dict[str, float]]]
    per_query_hits: list[QueryEval] = field(repr=False)


# ── Pure computation helpers (testable without I/O) ───────────────────────────


def check_sidecars(corpus_dir: Path, config_names: frozenset[str]) -> list[str]:
    """Check which sidecar files are missing.

    Args:
        corpus_dir: Directory containing sidecar JSON files.
        config_names: Set of config keys to check (e.g. ``256_10``).

    Returns:
        A sorted list of missing config keys. Empty list means all present.
    """
    missing: list[str] = []
    for c in sorted(config_names):
        path = corpus_dir / f"eval_vectors_{c}.json"
        if not path.exists():
            missing.append(c)
    return missing


def score_retrieval_result(
    result: RetrievalResult,
    expected_signature: str,
    top_k: int,
) -> dict[str, list[bool]]:
    """Score each retrieval mode against the expected signature.

    The result is padded with ``False`` if a mode returns fewer than ``top_k``
    results.

    Args:
        result: The retrieval result with dense, sparse, and fused paths.
        expected_signature: Substring to search for in each chunk text.
        top_k: Number of ranked results to consider.

    Returns:
        A dict mapping mode name to a list of bools (hit per rank).
    """
    hits: dict[str, list[bool]] = {}
    for mode in RETRIEVAL_MODES:
        chunks = getattr(result, mode)
        mode_hits = [is_hit(c.text, expected_signature) for c in chunks[:top_k]]
        while len(mode_hits) < top_k:
            mode_hits.append(False)
        hits[mode] = mode_hits
    return hits


def compute_config_metrics(
    config_key: str,
    query_evals: list[QueryEval],
    top_k: int,
) -> ConfigResult:
    """Aggregate per-query results into per-config metrics.

    For each retrieval mode computes mean recall@k and mean MRR across
    all queries, with per-stratum breakdowns.

    Args:
        config_key: The chunk-size config key (e.g. ``256_10``).
        query_evals: Per-query evaluation results.
        top_k: The rank cutoff used during evaluation.

    Returns:
        A ``ConfigResult`` with overall and per-stratum metrics.
    """
    recall_metric = RecallAtKMetric(top_k)

    strata: set[str] = {qe.stratum for qe in query_evals}
    per_stratum_hits: dict[str, dict[str, list[list[bool]]]] = {
        s: {m: [] for m in RETRIEVAL_MODES} for s in strata
    }
    all_hits: dict[str, list[list[bool]]] = {m: [] for m in RETRIEVAL_MODES}

    for qe in query_evals:
        for mode in RETRIEVAL_MODES:
            all_hits[mode].append(qe.hits[mode])
            per_stratum_hits[qe.stratum][mode].append(qe.hits[mode])

    def _mean(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    recall_key = f"recall@{top_k}"

    overall: dict[str, dict[str, float]] = {}
    for mode in RETRIEVAL_MODES:
        mode_all_hits = all_hits[mode]
        overall[mode] = {
            recall_key: _mean([recall_metric.compute(h) for h in mode_all_hits]),
            "mrr": _mean([MRRMetric.compute(h) for h in mode_all_hits]),
        }

    per_stratum_metrics: dict[str, dict[str, dict[str, float]]] = {}
    for stratum in sorted(strata):
        per_stratum_metrics[stratum] = {}
        for mode in RETRIEVAL_MODES:
            mode_hits = per_stratum_hits[stratum][mode]
            if not mode_hits:
                per_stratum_metrics[stratum][mode] = {recall_key: 0.0, "mrr": 0.0}
            else:
                per_stratum_metrics[stratum][mode] = {
                    recall_key: _mean([recall_metric.compute(h) for h in mode_hits]),
                    "mrr": _mean([MRRMetric.compute(h) for h in mode_hits]),
                }

    return ConfigResult(
        config_key=config_key,
        query_count=len(query_evals),
        overall=overall,
        per_stratum=per_stratum_metrics,
        per_query_hits=query_evals,
    )


def determine_winner(
    config_results: list[ConfigResult],
    top_k: int,
) -> tuple[str, str | None]:
    """Determine the winning chunk-size config.

    Primary: all configs within 0.02 of the highest fused recall@k are
    candidates. Among those, the config with the highest fused MRR wins.
    Runner-up is the second-best candidate by MRR (or ``None``).

    Args:
        config_results: List of aggregated results per config.
        top_k: The rank cutoff used during evaluation.

    Returns:
        A tuple of ``(winner_config_key, runner_up_config_key_or_None)``.
    """
    if not config_results:
        return ("", None)

    recall_key = f"recall@{top_k}"

    max_recall = max(cr.overall["fused"][recall_key] for cr in config_results)

    candidates: list[tuple[str, float]] = [
        (cr.config_key, cr.overall["fused"]["mrr"])
        for cr in config_results
        if max_recall - cr.overall["fused"][recall_key] <= 0.02
    ]

    candidates.sort(key=lambda x: -x[1])

    winner_key = candidates[0][0]
    runner_up: str | None = candidates[1][0] if len(candidates) > 1 else None

    return (winner_key, runner_up)


def format_results_table(
    config_results: list[ConfigResult],
    winner_key: str | None,
    top_k: int,
) -> str:
    """Format the tuning results as a human-readable table.

    Includes a config x retrieval-mode x metric matrix, per-stratum fused
    recall@k breakdown, and winner recommendation.

    Args:
        config_results: List of aggregated results per config.
        winner_key: The winning config key, or ``None``.
        top_k: The rank cutoff used during evaluation.

    Returns:
        A formatted string suitable for stdout.
    """
    lines: list[str] = []
    lines.append("Chunk-Size Tuning Results")
    lines.append("=" * 60)
    lines.append("")

    mode_label = {"dense": "Dense", "sparse": "Sparse", "fused": "Fused"}
    recall_key = f"recall@{top_k}"

    header = f"{'Config':<12}"
    for mode in RETRIEVAL_MODES:
        header += f"  {mode_label[mode]:<14}"
    header += f"  {'Winner':<8}"
    lines.append(header)
    lines.append("-" * len(header))

    for cr in config_results:
        row = f"{cr.config_key:<12}"
        for mode in RETRIEVAL_MODES:
            r = cr.overall[mode][recall_key]
            m = cr.overall[mode]["mrr"]
            row += f"  {r:.4f}/{m:.4f}  "
        is_winner = "\u2713" if cr.config_key == winner_key else ""
        row += f"  {is_winner:<8}"
        lines.append(row)

    lines.append("")
    strata = sorted({s for cr in config_results for s in cr.per_stratum})
    if strata:
        lines.append("Per-Stratum Fused Recall@k")
        lines.append("-" * 40)
        lines.append("")

        s_header = f"{'Config':<12}"
        for s in strata:
            s_header += f"  {s:<22}"
        lines.append(s_header)
        lines.append("-" * len(s_header))

        for cr in config_results:
            s_row = f"{cr.config_key:<12}"
            for s in strata:
                val = cr.per_stratum.get(s, {}).get("fused", {}).get(recall_key, 0.0)
                s_row += f"  {val:.4f}            "
            lines.append(s_row)

    lines.append("")
    if winner_key:
        lines.append(f"Recommended config: {winner_key}")
    else:
        lines.append("No winner could be determined.")

    return "\n".join(lines)


def build_results_json(
    config_results: list[ConfigResult],
    winner_key: str | None,
    top_k: int,
) -> dict[str, Any]:
    """Build the full results dict for JSON serialisation.

    Args:
        config_results: List of aggregated results per config.
        winner_key: The winning config key, or ``None``.
        top_k: The rank cutoff used during evaluation.

    Returns:
        A dict ready for ``json.dumps``.
    """
    return {
        "top_k": top_k,
        "winner": winner_key or "",
        "configs": [
            {
                "config_key": cr.config_key,
                "query_count": cr.query_count,
                "overall": cr.overall,
                "per_stratum": cr.per_stratum,
            }
            for cr in config_results
        ],
    }


# ── Orchestration (I/O-bound, database-backed) ────────────────────────────────


def run_tuning(args: argparse.Namespace) -> int:
    """Full tuning harness: check, seed, evaluate, aggregate, report.

    Args:
        args: Parsed CLI arguments (``.in_memory``, ``.top_k``).

    Returns:
        Exit code (0 = success).
    """
    top_k = getattr(args, "top_k", DEFAULT_TOP_K)

    # 1. Check sidecars
    missing = check_sidecars(_EVAL_CORPUS_DIR, CONFIG_NAMES)
    if missing:
        print(
            f"Error: missing sidecar files for config(s): {', '.join(missing)}. "
            f"Run `uv run python scripts/generate_eval_vectors.py` first.",
            file=sys.stderr,
        )
        return 1

    # 2. Load eval set
    if not _TUNING_YAML.exists():
        print(f"Error: tuning eval set not found: {_TUNING_YAML}", file=sys.stderr)
        return 1

    with open(_TUNING_YAML) as f:
        data = yaml.safe_load(f)
    queries: list[dict[str, Any]] = data.get("queries", [])
    if not queries:
        print("Error: no queries found in tuning eval set.", file=sys.stderr)
        return 1

    # 3. Set up embedder and retrieval config
    embedder: Embedder = (
        InMemoryEmbedder() if getattr(args, "in_memory", False) else EmbeddingsClient()
    )
    settings = Settings()
    retrieval_config = RetrievalConfig(
        model_name=settings.embedding_model,
        ef_search=settings.hnsw_ef_search,
        top_k=top_k,
        hybrid_search=True,
        reranker=False,
    )

    from sqlalchemy import text
    from sqlalchemy.orm import Session, sessionmaker

    engine = get_engine()
    config_results: list[ConfigResult] = []

    # 4. For each config, seed → evaluate → tear down
    for config_key in sorted(CONFIG_NAMES):
        print(f"\n--- Config: {config_key} ---")

        exit_code = seed_schema(config_key, engine=engine)
        if exit_code != 0:
            print(f"Error: failed to seed schema for {config_key}", file=sys.stderr)
            return exit_code

        schema_name = f"chunks_{config_key}"
        engine_with_schema = engine.execution_options(schema_translate_map={None: schema_name})
        SessionClass = sessionmaker(bind=engine_with_schema)

        query_evals: list[QueryEval] = []

        for i, q in enumerate(queries):
            query_text: str = q["query"]
            expected_signature: str = q["expected_signature"]
            stratum: str = q["stratum"]

            query_vectors = embedder.embed([query_text])
            query_vector = query_vectors[0]

            session: Session = SessionClass()
            try:
                # The retrieval queries (retrieval_qa.retrieval.query) use raw
                # text() SQL with unqualified table names. schema_translate_map
                # only rewrites ORM/Core Table SQL, so point search_path at the
                # seeded schema to route the raw queries to the right tables.
                # public must stay on the path for the pgvector `vector` type.
                session.execute(text(f'SET LOCAL search_path TO "{schema_name}", public'))
                result = retrieve_relevant_chunks(
                    query_vector=query_vector,
                    session=session,
                    config=retrieval_config,
                    query_text=query_text,
                )
            finally:
                session.close()

            hits = score_retrieval_result(result, expected_signature, top_k)
            query_evals.append(QueryEval(query=query_text, stratum=stratum, hits=hits))

            if (i + 1) % 10 == 0:
                print(f"  evaluated {i + 1}/{len(queries)} queries")

        config_result = compute_config_metrics(config_key, query_evals, top_k)
        config_results.append(config_result)
        print(f"  done — recall@{top_k}={config_result.overall['fused'][f'recall@{top_k}']:.4f}")

        teardown_schema(config_key, engine=engine)

    # 5. Determine winner
    winner_key, runner_up = determine_winner(config_results, top_k)
    if runner_up:
        print(f"\nRunner-up: {runner_up}")

    # 6. Output
    report = format_results_table(config_results, winner_key, top_k)
    print("\n" + report)

    json_path = _REPO_ROOT / "tuning_results.json"
    json_data = build_results_json(config_results, winner_key, top_k)
    json_path.write_text(json.dumps(json_data, indent=2))
    print(f"\nWrote full results to {json_path}")

    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]``).

    Returns:
        Exit code (0 = success).
    """
    parser = argparse.ArgumentParser(
        description="Run the chunk-size tuning harness end to end.",
    )
    parser.add_argument(
        "--in-memory",
        action="store_true",
        help="Use InMemoryEmbedder instead of the real API (for testing).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help=f"Number of results to consider per query (default: {DEFAULT_TOP_K}).",
    )
    args = parser.parse_args(argv)
    return run_tuning(args)


if __name__ == "__main__":
    sys.exit(main())
