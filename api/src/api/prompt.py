"""Prompt assembly for Harness A."""

from collections.abc import Sequence

from core.types.chat import ChatMessage
from core.types.responses import CitedPassage

SYSTEM_PROMPT = (
    "You answer the user's question using only the provided passages. "
    "If no passages are provided, or if the answer is not contained in them, "
    "say you cannot answer from the current corpus. Do not invent information."
)


def build_messages(
    query: str,
    chunks: Sequence[CitedPassage],
) -> list[ChatMessage]:
    """Assemble the chat-completions message list for the inference call.

    Args:
        query: The user's raw query string.
        chunks: Retrieved chunks (objects with a ``text`` attribute). Empty
            for the not-found branch, in which case the user message carries
            the query alone (no fake injected context) so the model emits a
            refusal.

    Returns:
        A two-message list: a system message setting the grounding contract,
        and a user message containing the injected context (when present) plus
        the query.
    """
    user_parts: list[str] = []
    if chunks:
        context = "\n\n".join(f"[{i}] {chunk.text}" for i, chunk in enumerate(chunks, start=1))
        user_parts.append("Injected Context:\n")
        user_parts.append(context)
    user_parts.append(f"Question: {query}")
    user_message = "\n\n".join(p for p in user_parts if p)
    return [
        ChatMessage(role="system", content=SYSTEM_PROMPT),
        ChatMessage(role="user", content=user_message),
    ]


__all__ = ["SYSTEM_PROMPT", "build_messages"]
