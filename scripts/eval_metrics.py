"""Pure-computation evaluation metrics for retrieval tuning.

Provides substring-based hit detection (``is_hit``) and rank-based metrics
(``RecallAtKMetric``, ``MRRMetric``) used by the chunk-size tuning harness.

All functions and classes are pure computation — no I/O, no database, no API
calls.  Each metric accepts an iterable of booleans (hit or miss per rank
position) and returns a float in [0, 1].
"""

import itertools
from collections.abc import Iterable


def is_hit(chunk_text: str, expected_signature: str) -> bool:
    """Check whether ``expected_signature`` is a substring of ``chunk_text``.

    Args:
        chunk_text: The content of a retrieved chunk.
        expected_signature: The expected text pattern to search for.

    Returns:
        ``True`` if ``expected_signature`` is found as a substring
        of ``chunk_text``, ``False`` otherwise.
    """
    return expected_signature in chunk_text


class RecallAtKMetric:
    """Recall@k computed from a ranked list of hit/miss results.

    Recall@k is the fraction of the top-``k`` results that are hits.  When
    the result list has fewer than ``k`` items, the missing positions are
    treated as misses (the denominator is always ``k``).

    Args:
        k: The rank cutoff.  Must be >= 1.
    """

    def __init__(self, k: int) -> None:
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        self.k = k

    def compute(self, results: Iterable[bool]) -> float:
        """Compute recall@k from ranked hit/miss results.

        Args:
            results: Iterable of booleans ordered by rank (position 1 first).
                     ``True`` indicates a hit; ``False`` indicates a miss.

        Returns:
            A float in [0, 1] — the fraction of top-``k`` results that
            are hits.
        """
        hits = sum(1 for hit in itertools.islice(results, self.k) if hit)
        return hits / self.k


class MRRMetric:
    """Mean Reciprocal Rank (MRR) computed from a ranked list of hit/miss results.

    For a single ranked result list this is the reciprocal rank of the first
    hit (1 / rank).  Returns 0.0 when no hit is found.  Average multiple
    queries at the call site to obtain a true *mean* reciprocal rank.
    """

    @staticmethod
    def compute(results: Iterable[bool]) -> float:
        """Compute the reciprocal rank of the first hit.

        Args:
            results: Iterable of booleans ordered by rank (position 1 first).
                     ``True`` indicates a hit; ``False`` indicates a miss.

        Returns:
            A float in [0, 1] — the reciprocal rank of the first hit,
            or 0.0 if no hit is found.
        """
        for rank, hit in enumerate(results, start=1):
            if hit:
                return 1.0 / rank
        return 0.0
