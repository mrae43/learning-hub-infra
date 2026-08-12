"""Passage Transform stage (depth-dive spec §5).

Runs before the framing agent: validates the carried content against the
declared ``passage_type``, normalizes, and emits model-ready content blocks.
For the MVP tracer bullet only ``text`` passes are transformable; other
passage types raise ``PassageTransformError`` until their transforms land.

Requests exceeding declared bounds are rejected (422 via the route), never
silently cropped.
"""

from core.types.captured_passage import CapturedPassage, TextPassage

TEXT_PASSAGE_MAX_CHARS = 100_000
"""MVP size bound for text passages (spec §5: bounded by model token budget)."""


class PassageTransformError(Exception):
    """A captured passage violated transform validation or is unsupported.

    Raised for size-bound violations and for passage types the transform
    cannot yet process. The route layer maps this to a 422 response.
    """


def transform_passage(passage: CapturedPassage) -> str:
    """Validate a captured passage and return the model-ready text block.

    Args:
        passage: The captured passage carried in the request.

    Returns:
        The normalized (stripped) text content for ``text`` passages.

    Raises:
        PassageTransformError: The passage content violates the declared
            bounds, or the passage type is not supported by this transform.
    """
    if isinstance(passage, TextPassage):
        return _transform_text(passage.content)
    raise PassageTransformError(
        f"passage_type {passage.passage_type!r} is not supported by the MVP tracer bullet"
    )


def _transform_text(content: str) -> str:
    """Validate and normalize a text passage.

    Args:
        content: The raw captured text content.

    Returns:
        The stripped text content.

    Raises:
        PassageTransformError: The content is empty/whitespace-only or
            exceeds ``TEXT_PASSAGE_MAX_CHARS`` characters.
    """
    stripped = content.strip()
    if not stripped:
        raise PassageTransformError("text passage content must be non-empty")
    if len(stripped) > TEXT_PASSAGE_MAX_CHARS:
        raise PassageTransformError(
            f"text passage content exceeds the {TEXT_PASSAGE_MAX_CHARS} character limit"
        )
    return stripped


__all__ = ["TEXT_PASSAGE_MAX_CHARS", "PassageTransformError", "transform_passage"]
