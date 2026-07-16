"""Boundary types for chat messages across the inference pipeline."""

from pydantic import BaseModel


class ChatMessage(BaseModel):
    """A single chat-completions message with role and content."""

    role: str
    content: str
