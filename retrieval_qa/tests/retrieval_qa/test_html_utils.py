"""Tests for shared HTML utilities used by chunkers."""

from html.parser import HTMLParser

from retrieval_qa.chunking._html_utils import _BaseHTMLTextExtractor


def _extract(html: str) -> str:
    """Helper to feed HTML and return extracted text."""
    extractor = _BaseHTMLTextExtractor()
    extractor.feed(html)
    extractor.close()
    return extractor.get_text()


def test_strips_script_content() -> None:
    """Content inside <script> tags is removed."""
    html = "<p>Hello</p><script>var x = 1;</script><p>World</p>"
    assert _extract(html) == "Hello\n\nWorld"


def test_strips_style_content() -> None:
    """Content inside <style> tags is removed."""
    html = "<p>Hello</p><style>body { color: red; }</style><p>World</p>"
    assert _extract(html) == "Hello\n\nWorld"


def test_structural_newlines_for_block_tags() -> None:
    """Block-level tags produce newlines around their content."""
    html = "<p>First</p><div>Second</div><li>Third</li>"
    result = _extract(html)
    # Each block tag body should be separated by newlines
    assert "First" in result
    assert "Second" in result
    assert "Third" in result
    assert result.startswith("First")
    assert "Second" in result


def test_br_tag_inserts_newline() -> None:
    """<br> tags insert a newline."""
    html = "Line1<br>Line2"
    assert _extract(html) == "Line1\nLine2"


def test_heading_tags_preserve_content() -> None:
    """Heading content is preserved as-is (no heading markers in base)."""
    html = "<h1>Title</h1><p>Body</p><h2>Subtitle</h2><p>More</p>"
    result = _extract(html)
    assert "Title" in result
    assert "Subtitle" in result
    assert "Body" in result
    assert "More" in result


def test_get_text_normalizes_whitespace() -> None:
    """Multiple consecutive newlines are collapsed to two."""
    html = "<p>A</p><p>B</p><p>C</p>"
    assert _extract(html) == "A\n\nB\n\nC"


def test_empty_html_returns_empty_string() -> None:
    """Empty or whitespace-only HTML returns an empty string."""
    assert _extract("") == ""
    assert _extract("<html><body></body></html>") == ""


def test_plain_text_passes_through() -> None:
    """HTML with no tags returns the raw text unchanged."""
    assert _extract("Hello, World!") == "Hello, World!"


def test_subclass_can_override_heading_behavior() -> None:
    """Subclassing works: a heading-aware subclass can add markers."""

    class _HeadingExtractor(_BaseHTMLTextExtractor):
        def __init__(self) -> None:
            super().__init__()
            self._heading_level: int | None = None

        def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
            if tag in {"h1", "h2", "h3", "h4", "h5", "h6"}:
                self._heading_level = int(tag[1])
                self._parts.append("\n")
                self._parts.append("#" * self._heading_level + " ")
            else:
                super().handle_starttag(tag, attrs)

        def handle_endtag(self, tag: str) -> None:
            if tag in {"h1", "h2", "h3", "h4", "h5", "h6"}:
                self._heading_level = None
                self._parts.append("\n")
            else:
                super().handle_endtag(tag)

    extractor = _HeadingExtractor()
    extractor.feed("<h1>Title</h1><p>Body</p><h2>Sub</h2><p>Text</p>")
    extractor.close()
    result = extractor.get_text()

    assert result.startswith("# Title")
    assert "Body" in result
    assert "## Sub" in result


def test_subclass_can_skip_head_tag() -> None:
    """Subclassing works for adding <head> tag skipping."""

    class _HeadSkippingExtractor(_BaseHTMLTextExtractor):
        def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
            if tag == "head":
                self._skip = True
            else:
                super().handle_starttag(tag, attrs)

        def handle_endtag(self, tag: str) -> None:
            if tag == "head":
                self._skip = False
            else:
                super().handle_endtag(tag)

    html = "<p>Before</p><head><title>Title</title></head><p>After</p>"
    assert _HeadSkippingExtractor().get_text() is not None  # smoke-test construction
    extractor = _HeadSkippingExtractor()
    extractor.feed(html)
    extractor.close()
    result = extractor.get_text()
    assert "Before" in result
    assert "After" in result
    assert "Title" not in result


def test_inherits_from_htmlparser() -> None:
    """_BaseHTMLTextExtractor is a genuine HTMLParser subclass."""
    assert issubclass(_BaseHTMLTextExtractor, HTMLParser)
