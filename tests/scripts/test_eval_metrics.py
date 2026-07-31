"""Tests for scripts/eval_metrics.py.

Covers the three pure-computation evaluation metric components:
``is_hit``, ``RecallAtKMetric``, and ``MRRMetric``.
"""

from scripts.eval_metrics import MRRMetric, RecallAtKMetric, is_hit


def test_is_hit_returns_true_when_signature_present() -> None:
    assert is_hit("The quick brown fox jumps over the lazy dog.", "fox")


def test_is_hit_returns_false_when_signature_absent() -> None:
    assert not is_hit("The quick brown fox jumps over the lazy dog.", "cat")


def test_is_hit_case_sensitive() -> None:
    assert not is_hit("The quick brown FOX jumps.", "fox")


def test_is_hit_empty_chunk_text() -> None:
    assert not is_hit("", "fox")


def test_is_hit_empty_signature() -> None:
    assert is_hit("any text", "")


def test_is_hit_exact_match() -> None:
    assert is_hit("fox", "fox")


def test_recall_at_k_all_hits() -> None:
    assert RecallAtKMetric(k=10).compute([True] * 10) == 1.0


def test_recall_at_k_no_hits() -> None:
    assert RecallAtKMetric(k=10).compute([False] * 10) == 0.0


def test_recall_at_k_partial_hits() -> None:
    assert RecallAtKMetric(k=10).compute([True] * 3 + [False] * 7) == 0.3


def test_recall_at_k_ignores_results_beyond_k() -> None:
    assert RecallAtKMetric(k=3).compute([True, False, True, True, True]) == 2 / 3


def test_recall_at_k_fewer_results_than_k() -> None:
    assert RecallAtKMetric(k=10).compute([True, True, True]) == 0.3


def test_recall_at_k_empty_results() -> None:
    assert RecallAtKMetric(k=10).compute([]) == 0.0


def test_recall_at_k_k_of_one() -> None:
    assert RecallAtKMetric(k=1).compute([True, False, False]) == 1.0
    assert RecallAtKMetric(k=1).compute([False, True, False]) == 0.0


def test_recall_at_k_accepts_generator() -> None:
    gen = (b for b in [True, False, True])
    assert RecallAtKMetric(k=3).compute(gen) == 2 / 3


def test_mrr_first_hit_at_position_one() -> None:
    assert MRRMetric.compute([True, False, False]) == 1.0


def test_mrr_first_hit_at_position_three() -> None:
    assert MRRMetric.compute([False, False, True, False]) == 1 / 3


def test_mrr_no_hits() -> None:
    assert MRRMetric.compute([False, False, False]) == 0.0


def test_mrr_empty_results() -> None:
    assert MRRMetric.compute([]) == 0.0


def test_mrr_first_hit_at_position_one_among_many() -> None:
    assert MRRMetric.compute([True, True, True, True]) == 1.0


def test_mrr_accepts_generator() -> None:
    gen = (b for b in [False, True])
    assert MRRMetric.compute(gen) == 0.5
