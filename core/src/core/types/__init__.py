"""Shared Pydantic boundary types."""

from core.types.captured_passage import (
    CapturedPassage,
    CodePassage,
    DiagramPassage,
    ImageMediaType,
    ImagePassage,
    TablePassage,
    TextPassage,
)
from core.types.chat import ChatMessage
from core.types.chunk import (
    BookChunkMetadata,
    Chunk,
    DocumentationChunkMetadata,
    PaperChunkMetadata,
)
from core.types.depth_dive import (
    AnimationStep,
    ElementState,
    ElementStyle,
    HarnessBRequest,
    HarnessBResponse,
    InteractionHints,
    InteractiveAnimation,
    SceneElement,
    Treatment,
    Viewport,
)
from core.types.document import (
    DocumentStatus,
    DocumentStatusResponse,
    DocumentType,
)
from core.types.responses import (
    CitedPassage,
    HarnessARequest,
    HarnessAResponse,
    RetrievalResult,
    ScoredChunk,
)
from core.types.retrieval_config import RetrievalConfig

__all__ = [
    "AnimationStep",
    "BookChunkMetadata",
    "CapturedPassage",
    "ChatMessage",
    "Chunk",
    "CitedPassage",
    "CodePassage",
    "DiagramPassage",
    "DocumentStatus",
    "DocumentStatusResponse",
    "DocumentType",
    "DocumentationChunkMetadata",
    "ElementState",
    "ElementStyle",
    "HarnessARequest",
    "HarnessAResponse",
    "HarnessBRequest",
    "HarnessBResponse",
    "ImageMediaType",
    "ImagePassage",
    "InteractionHints",
    "InteractiveAnimation",
    "PaperChunkMetadata",
    "RetrievalConfig",
    "RetrievalResult",
    "SceneElement",
    "ScoredChunk",
    "TablePassage",
    "TextPassage",
    "Treatment",
    "Viewport",
]
