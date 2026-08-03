"""Shared text utilities for the Learning Hub monorepo."""


def count_tokens(text: str) -> int:
    """Approximate token count for chunk sizing.

    Uses a simple whitespace split; this is sufficient for MVP chunk ordering
    and sanity checks. More precise counting can be swapped in later without
    changing the chunker interface.

    Args:
        text: The text to count tokens for.

    Returns:
        The number of whitespace-separated words, with a minimum of one.
    """
    return max(1, len(text.split()))
