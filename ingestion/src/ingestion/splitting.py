"""Recursive fixed-size text splitter for parent-child chunking.

Per ADR-0016, structure-aware chunks (parents) are split into fixed-size
child chunks of 512 tokens with 15% contextual overlap. Only child chunks
are embedded and indexed for retrieval.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ChildSplit:
    """A single child chunk produced by the recursive splitter.

    Attributes:
        content: The text content of the child chunk.
        token_count: Approximate token count (whitespace-based).
        position: Zero-based position of this child within its parent.
    """

    content: str
    token_count: int
    position: int


_DEFAULT_CHUNK_SIZE = 512
_DEFAULT_OVERLAP_RATIO = 0.15


def _count_tokens(text: str) -> int:
    """Approximate token count matching the convention in retrieval_qa._utils."""
    return max(1, len(text.split()))


def recursive_fixed_size_split(
    text: str,
    text_token_count: int | None = None,
    *,
    chunk_size: int = _DEFAULT_CHUNK_SIZE,
    overlap_ratio: float = _DEFAULT_OVERLAP_RATIO,
) -> list[ChildSplit]:
    """Split ``text`` into fixed-size child chunks with contextual overlap.

    If the text is at or below ``chunk_size`` tokens, a single child is
    returned (identity split). Otherwise, the text is split into sliding
    windows of ``chunk_size`` tokens with ``overlap_ratio`` fractional
    overlap between adjacent windows.

    Args:
        text: The parent chunk content to split.
        text_token_count: Precomputed token count. If ``None``, computed
            via whitespace split.
        chunk_size: Maximum tokens per child chunk.
        overlap_ratio: Fractional overlap between adjacent children
            (e.g. 0.15 = 15%).

    Returns:
        A list of ``ChildSplit`` instances ordered by position.

    Raises:
        ValueError: If ``text`` is empty.
    """
    if not text:
        raise ValueError("text cannot be empty")

    token_count = text_token_count if text_token_count is not None else _count_tokens(text)

    if token_count <= chunk_size:
        return [ChildSplit(content=text, token_count=token_count, position=0)]

    tokens = text.split()
    overlap_tokens = round(chunk_size * overlap_ratio)
    step = max(1, chunk_size - overlap_tokens)

    children: list[ChildSplit] = []
    start = 0
    position = 0

    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        child_tokens = tokens[start:end]
        child_content = " ".join(child_tokens)
        child_token_count = len(child_tokens)
        children.append(
            ChildSplit(content=child_content, token_count=child_token_count, position=position)
        )
        position += 1
        if end >= len(tokens):
            break
        start += step

    return children


__all__ = ["ChildSplit", "recursive_fixed_size_split"]
