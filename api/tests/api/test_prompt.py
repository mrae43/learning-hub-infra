"""Tests for Harness A prompt assembly."""

from uuid import uuid4

from api.prompt import SYSTEM_PROMPT, build_messages
from core.types.chat import ChatMessage
from core.types.responses import CitedPassage


def test_build_messages_includes_system_prompt() -> None:
    """The system message sets the grounding contract."""
    messages = build_messages("q", [])
    system = next(m for m in messages if m.role == "system")
    assert system.content == SYSTEM_PROMPT


def test_build_messages_includes_context_when_present() -> None:
    """The user message includes enumerated context when chunks are provided."""
    chunks = [
        CitedPassage(chunk_id=uuid4(), text="first passage"),
        CitedPassage(chunk_id=uuid4(), text="second passage"),
    ]
    messages = build_messages("what is RAG?", chunks)
    user = next(m for m in messages if m.role == "user")
    assert "first passage" in user.content
    assert "second passage" in user.content
    assert "[1] first passage" in user.content
    assert "[2] second passage" in user.content
    assert "what is RAG?" in user.content


def test_build_messages_omits_context_when_empty() -> None:
    """The user message omits the Injected Context block when no chunks are retrieved."""
    messages = build_messages("what is RAG?", [])
    user = next(m for m in messages if m.role == "user")
    assert "what is RAG?" in user.content
    assert "Injected Context:" not in user.content


def test_build_messages_returns_two_messages() -> None:
    """Prompt assembly returns exactly a system and a user message."""
    messages = build_messages("q", [])
    assert len(messages) == 2
    assert messages[0].role == "system"
    assert messages[1].role == "user"


def test_build_messages_returns_chat_message_models() -> None:
    """All returned messages are typed ChatMessage instances."""
    messages = build_messages("q", [])
    assert all(isinstance(m, ChatMessage) for m in messages)
