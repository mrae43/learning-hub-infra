"""Internal shared utilities for the retrieval_qa package."""

from hashlib import sha256


def count_tokens(text: str) -> int:
    """Approximate token count for chunk sizing.

    Uses a simple whitespace split; this is sufficient for MVP chunk ordering
    and sanity checks. More precise counting can be swapped in later without
    changing the chunker interface.
    """
    return max(1, len(text.split()))


def _sha256(text: str) -> str:
    return sha256(text.encode()).hexdigest()
