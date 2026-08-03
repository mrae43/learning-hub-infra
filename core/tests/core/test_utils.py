"""Tests for shared core text utilities."""

from core.utils import count_tokens


def test_count_tokens_counts_whitespace_separated_words() -> None:
    """A sentence counts as the number of whitespace-separated words."""
    assert count_tokens("the quick brown fox") == 4


def test_count_tokens_empty_text_counts_one() -> None:
    """Empty text never reports zero, matching the chunker convention."""
    assert count_tokens("") == 1


def test_count_tokens_single_word_counts_one() -> None:
    """A single word counts as one token."""
    assert count_tokens("hello") == 1


def test_count_tokens_collapses_whitespace() -> None:
    """Multiple consecutive whitespace characters count as one separator."""
    assert count_tokens("a   b\n\t c") == 3
