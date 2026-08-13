"""Web-search tool wrapper for Depth Dive (ADR-0012, ADR-0013).

The framing agent decides per-request whether external grounding would help
(``search_intent``); the assembly agent calls this wrapper only when the brief
carries one. The wrapper retries once on failure and falls back with a
user-facing note if the retry also fails (ADR-0013). Lives inside Depth Dive,
not a generic tool (ai-system-tree.md).
"""

from depth_dive.web_search.client import (
    OpenAIWebSearchClient,
    StubWebSearchClient,
    WebSearchClient,
    WebSearchError,
    WebSearchResult,
)
from depth_dive.web_search.wrapper import FALLBACK_NOTE, SearchOutcome, run_web_search

__all__ = [
    "FALLBACK_NOTE",
    "OpenAIWebSearchClient",
    "SearchOutcome",
    "StubWebSearchClient",
    "WebSearchClient",
    "WebSearchError",
    "WebSearchResult",
    "run_web_search",
]
