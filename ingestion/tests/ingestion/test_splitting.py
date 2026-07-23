"""Tests for the recursive fixed-size text splitter."""

from ingestion.splitting import recursive_fixed_size_split


def test_identity_split_when_under_limit() -> None:
    """A parent ≤512 tokens produces exactly one child (identity split)."""
    content = "small chunk"
    parent_token_count = 2
    result = recursive_fixed_size_split(content, parent_token_count)
    assert len(result) == 1
    assert result[0].content == content
    assert result[0].token_count == 2
    assert result[0].position == 0


def test_identity_split_at_exact_limit() -> None:
    """A parent at exactly 512 tokens produces exactly one child."""
    tokens = ["token"] * 512
    content = " ".join(tokens)
    result = recursive_fixed_size_split(content, 512)
    assert len(result) == 1
    assert result[0].position == 0


def test_splits_into_multiple_children() -> None:
    """A parent >512 tokens produces multiple child chunks."""
    tokens = ["token"] * 1200
    content = " ".join(tokens)
    result = recursive_fixed_size_split(content, 1200)
    assert len(result) >= 2


def test_each_child_within_token_limit() -> None:
    """Each child chunk is ≤512 tokens."""
    tokens = ["token"] * 2000
    content = " ".join(tokens)
    result = recursive_fixed_size_split(content, 2000)
    for child in result:
        assert child.token_count <= 512, f"Child has {child.token_count} tokens, exceeds 512"


def test_overlapping_boundary_content() -> None:
    """Adjacent children share overlapping content."""
    tokens = [f"word{i}" for i in range(1500)]
    content = " ".join(tokens)
    result = recursive_fixed_size_split(content, 1500)

    # Adjacent children should share some tokens
    for i in range(len(result) - 1):
        current_tokens = set(result[i].content.split())
        next_tokens = set(result[i + 1].content.split())
        overlap = current_tokens & next_tokens
        assert len(overlap) > 0, f"Children {i} and {i + 1} have no overlapping content"


def test_position_enumerates_within_parent() -> None:
    """Child position enumerates from 0 within the parent."""
    tokens = ["token"] * 2000
    content = " ".join(tokens)
    result = recursive_fixed_size_split(content, 2000)
    for i, child in enumerate(result):
        assert child.position == i, f"Expected position {i}, got {child.position}"


def test_children_ordered_correctly() -> None:
    """Children are ordered by position, first child starts with parent content."""
    words = [f"unique_word_{i}" for i in range(600)]
    content = " ".join(words)
    result = recursive_fixed_size_split(content, 600)
    assert len(result) >= 2
    assert result[0].content.startswith("unique_word_0")
    assert result[-1].content.endswith("unique_word_599")


def test_small_chunk_size() -> None:
    """Works correctly with very small chunk sizes."""
    tokens = ["a"] * 50
    content = " ".join(tokens)
    result = recursive_fixed_size_split(content, 50, chunk_size=10, overlap_ratio=0.0)
    assert len(result) == 5
    for child in result:
        assert child.token_count <= 10


def test_zero_overlap() -> None:
    """With 0% overlap, children partition the content without sharing."""
    tokens = [f"w{i}" for i in range(600)]
    content = " ".join(tokens)
    result = recursive_fixed_size_split(content, 600, overlap_ratio=0.0)
    assert len(result) >= 2


def test_empty_content_raises() -> None:
    """Empty content raises ValueError."""
    import pytest

    with pytest.raises(ValueError, match="cannot be empty"):
        recursive_fixed_size_split("", 0)


def test_single_token() -> None:
    """A single token produces exactly one child."""
    result = recursive_fixed_size_split("hello", 1)
    assert len(result) == 1
    assert result[0].content == "hello"
    assert result[0].token_count == 1
