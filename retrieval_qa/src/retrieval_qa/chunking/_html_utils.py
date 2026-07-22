"""Shared HTML utilities for chunkers.

Provides ``_BaseHTMLTextExtractor``, a foundation for HTML tag-stripping
with structural newlines.  Chunkers extend or compose from it for their
specific heading-awareness behaviour.
"""

import re
from html.parser import HTMLParser


class _BaseHTMLTextExtractor(HTMLParser):
    """Strip HTML tags while preserving structural newlines.

    Skips content inside ``<script>`` and ``<style>`` elements.  Inserts a
    single ``\\n`` for block-level structural tags (``<p>``, ``<div>``,
    ``<li>``, ``<br>``) and heading tags (``<h1>``--``<h4>``) so the
    remaining text retains basic paragraph / list structure.
    """

    _SKIP_TAGS = frozenset({"script", "style"})
    _UNSKIP_TAGS = frozenset({"script", "style"})
    _STRUCTURAL_START_TAGS = frozenset({"p", "div", "li", "br", "h1", "h2", "h3", "h4"})
    _STRUCTURAL_END_TAGS = frozenset({"p", "div", "li", "h1", "h2", "h3", "h4"})

    def __init__(self) -> None:
        super().__init__()
        self._parts: list[str] = []
        self._skip = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in self._SKIP_TAGS:
            self._skip = True
        elif tag in self._STRUCTURAL_START_TAGS:
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in self._UNSKIP_TAGS:
            self._skip = False
        elif tag in self._STRUCTURAL_END_TAGS:
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._skip:
            self._parts.append(data)

    def get_text(self) -> str:
        """Join extracted parts and normalize whitespace."""
        text = "".join(self._parts)
        return re.sub(r"\n\s*\n+", "\n\n", text).strip()


__all__ = ["_BaseHTMLTextExtractor"]
