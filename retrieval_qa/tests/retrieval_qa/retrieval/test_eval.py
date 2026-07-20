"""Recall@k retrieval eval using DeepEval.

Each parametrized query exercises the real pgvector retrieval pipeline against
a pre-seeded corpus (``eval_corpus`` session fixture).  A custom
``RecallAtKMetric`` computes set-intersection recall against expected chunks.

Queries marked ``known_borderline: true`` in the eval set produce a warning
instead of a test failure, keeping their score and mismatch details visible.
"""

import json
import warnings
from hashlib import sha256
from pathlib import Path
from typing import Any, Self

import pytest
import yaml
from deepeval import assert_test  # type: ignore[attr-defined]
from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase
from pydantic import BaseModel, model_validator
from sqlalchemy.orm import Session

from core.config.settings import Settings
from core.types.retrieval_config import RetrievalConfig
from retrieval_qa.retrieval.query import retrieve_relevant_chunks

_EVAL_SET_PATH = Path(__file__).parent / "eval_set.yaml"
_EVAL_VECTORS_PATH = _EVAL_SET_PATH.with_name("eval_vectors.json")


class EvalQuery(BaseModel):
    """A single query entry from the eval YAML set.

    Fields mirror the YAML schema.  ``reason`` is required (non-``None``)
    when ``known_borderline`` is ``True``, enforced by ``model_validator``.
    """

    query: str
    content_sha256: str
    expected_chunk_contents: list[str]
    known_borderline: bool = False
    reason: str | None = None

    @model_validator(mode="after")
    def reason_required_when_borderline(self) -> Self:
        if self.known_borderline and self.reason is None:
            raise ValueError("reason is required when known_borderline is true")
        return self


def _load_eval_data() -> dict[str, Any]:
    with open(_EVAL_SET_PATH) as f:
        return yaml.safe_load(f)  # type: ignore[no-any-return]


def _sha256(text: str) -> str:
    return sha256(text.encode()).hexdigest()


def _validate_eval_integrity(
    data: dict[str, Any],
    vectors: dict[str, list[float]],
    sidecar_model: str | None = None,
) -> None:
    """Validate SHA-256 hashes and vector cross-references.

    Checks:
    - Every chunk content and query string hash matches its stored hash.
    - Every YAML content hash has a corresponding entry in the vector JSON.
    - Every vector in JSON has a corresponding YAML entry (no orphans).
    - Model metadata matches ``settings.embedding_model`` (if provided).
    """
    yaml_hashes: set[str] = set()

    expected_model = Settings().embedding_model
    if sidecar_model is not None:
        assert sidecar_model == expected_model, (
            f"Sidecar JSON model '{sidecar_model}' does not match "
            f"settings.embedding_model '{expected_model}'. "
            f"Re-run scripts/generate_eval_vectors.py."
        )

    for doc in data.get("documents", []):
        for chunk in doc.get("chunks", []):
            actual = _sha256(chunk["content"])
            expected = chunk["content_sha256"]
            assert actual == expected, (
                f"SHA-256 mismatch for chunk content {chunk['content'][:60]!r}...: "
                f"got {actual}, expected {expected}. "
                f"Re-run scripts/generate_eval_vectors.py to update."
            )
            yaml_hashes.add(expected)

    for query in data.get("queries", []):
        actual = _sha256(query["query"])
        expected = query["content_sha256"]
        assert actual == expected, (
            f"SHA-256 mismatch for query {query['query']!r}: "
            f"got {actual}, expected {expected}. "
            f"Re-run scripts/generate_eval_vectors.py to update."
        )
        yaml_hashes.add(expected)

    for yaml_hash in yaml_hashes:
        assert yaml_hash in vectors, (
            f"Vector missing for entry with hash {yaml_hash}. "
            f"Re-run `uv run python scripts/generate_eval_vectors.py`"
        )

    for vec_hash in vectors:
        assert vec_hash in yaml_hashes, (
            f"Orphan vector in sidecar JSON for hash {vec_hash} — no matching YAML entry found."
        )


def _load_vectors() -> dict[str, list[float]]:
    with open(_EVAL_VECTORS_PATH) as f:
        sidecar = json.load(f)
    with open(_EVAL_SET_PATH) as f:
        eval_data = yaml.safe_load(f)
    vectors: dict[str, list[float]] = sidecar["vectors"]
    _validate_eval_integrity(eval_data, vectors, sidecar_model=sidecar.get("model"))
    return vectors


_EVAL_DATA = _load_eval_data()
_EVAL_VECTORS = _load_vectors()
_EVAL_QUERIES: list[EvalQuery] = [EvalQuery.model_validate(q) for q in _EVAL_DATA["queries"]]

_QUERY_VECTORS: dict[str, list[float]] = {
    q["content_sha256"]: _EVAL_VECTORS[q["content_sha256"]] for q in _EVAL_DATA["queries"]
}


class RecallAtKMetric(BaseMetric):  # type: ignore[no-untyped-call]
    """Set-intersection recall@k against top-k retrieved chunk texts.

    Args:
        expected_chunks: Ground-truth chunk content strings for the query.
        threshold: Minimum recall to consider the test successful.
    """

    def __init__(self, expected_chunks: list[str], threshold: float = 0.5) -> None:
        self.expected_chunks = set(expected_chunks)
        self.threshold = threshold
        self.score: float | None = None

    def measure(self, test_case: LLMTestCase, *args: object, **kwargs: object) -> float:
        retrieved = set(test_case.retrieval_context or [])
        expected = self.expected_chunks
        if not expected:
            self.score = 1.0
        else:
            intersection = len(retrieved & expected)
            self.score = intersection / len(expected)
        return self.score

    async def a_measure(self, test_case: LLMTestCase, *args: object, **kwargs: object) -> float:
        return self.measure(test_case, *args, **kwargs)

    def is_successful(self) -> bool:
        return self.score is not None and self.score >= self.threshold

    @property
    def __name__(self) -> str:
        return "Recall@K"


@pytest.mark.parametrize(
    "query_data",
    _EVAL_QUERIES,
    ids=[q.query[:40] for q in _EVAL_QUERIES],
)
def test_recall_at_k_retrieves_expected_passages(
    query_data: EvalQuery, eval_session: Session
) -> None:
    """Retrieve chunks for the query and assert recall@k >= threshold.

    Queries with ``known_borderline: true`` emit a warning carrying the score
    and mismatch details instead of calling ``assert_test`` (which would fail).
    """
    top_k = Settings().query_top_k
    query_vector = _QUERY_VECTORS[query_data.content_sha256]
    results = retrieve_relevant_chunks(
        query_vector=query_vector,
        session=eval_session,
        config=RetrievalConfig(
            model_name="text-embedding-3-small",
            ef_search=Settings().hnsw_ef_search,
            top_k=top_k,
        ),
    )

    retrieved_texts = [r.text for r in results]
    test_case = LLMTestCase(
        input=query_data.query,
        actual_output="",
        retrieval_context=retrieved_texts,  # type: ignore[arg-type]
    )
    metric = RecallAtKMetric(
        expected_chunks=query_data.expected_chunk_contents,
    )

    if query_data.known_borderline:
        metric.measure(test_case)
        expected_items = "\n".join(f"  - {c[:80]}" for c in query_data.expected_chunk_contents)
        retrieved_items = "\n".join(f"  - {t[:80]}" for t in retrieved_texts)
        warnings.warn(
            f"Borderline query: {query_data.query[:60]!r}...\n"
            f"  score: {metric.score:.2f}  (threshold: {metric.threshold})\n"
            f"  reason: {query_data.reason}\n"
            f"  expected:\n{expected_items}\n"
            f"  retrieved:\n{retrieved_items}",
            stacklevel=2,
        )
    else:
        assert_test(test_case, [metric])
