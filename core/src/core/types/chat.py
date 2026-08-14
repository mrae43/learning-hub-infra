"""Boundary types for chat messages across the inference pipeline."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field


class ChatContentImageURL(BaseModel):
    """An image reference for a multimodal content part.

    ``url`` is either a public URL or a base64 data URI
    (``data:<media_type>;base64,<payload>``), matching the OpenAI vision shape.
    """

    model_config = ConfigDict(extra="forbid")

    url: str


class ChatContentTextPart(BaseModel):
    """A plain-text content part of a chat message."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["text"] = "text"
    text: str


class ChatContentImagePart(BaseModel):
    """An image content part of a chat message."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["image_url"] = "image_url"
    image_url: ChatContentImageURL


ChatContentPart = Annotated[
    ChatContentTextPart | ChatContentImagePart,
    Field(discriminator="type"),
]
"""A content part of a multimodal chat message: text or an image reference."""


class ChatMessage(BaseModel):
    """A single chat-completions message with role and content.

    ``content`` is a plain string for text-only messages (Harness A and the
    Depth Dive generation turn's system message), or a list of content parts
    when the message carries image content (Depth Dive's model-ready image
    carriers).
    """

    role: str
    content: str | list[ChatContentPart]


__all__ = [
    "ChatContentImagePart",
    "ChatContentImageURL",
    "ChatContentPart",
    "ChatContentTextPart",
    "ChatMessage",
]
