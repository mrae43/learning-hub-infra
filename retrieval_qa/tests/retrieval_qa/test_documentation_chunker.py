"""Tests for the documentation chunker."""

import io

import pytest

from core.exceptions import IngestionError
from core.types.chunk import DocumentationChunkMetadata
from retrieval_qa.chunking.documentation_chunker import chunk_documentation


def _make_pdf_from_text(text: str) -> bytes:
    """Build a minimal PDF containing the given text using reportlab."""
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas

    buffer = io.BytesIO()
    canva = canvas.Canvas(buffer, pagesize=letter)
    y = 720
    for line in text.splitlines():
        canva.drawString(72, y, line)
        y -= 20
    canva.showPage()
    canva.save()
    buffer.seek(0)
    return buffer.read()


def test_documentation_chunker_splits_markdown_by_headings() -> None:
    """Markdown headings create page/section boundaries."""
    markdown = (
        "# Installation\n\n"
        "Install the package with pip.\n\n"
        "## Quick start\n\n"
        "Run the init command.\n\n"
        "# API Reference\n\n"
        "Endpoints are described below.\n\n"
        "## Users\n\n"
        "Manage users.\n"
    )
    chunks = chunk_documentation(markdown.encode("utf-8"))

    assert len(chunks) >= 2
    pages = {chunk.metadata.page for chunk in chunks}
    assert "Installation" in pages
    assert "API Reference" in pages


def test_documentation_chunker_detects_api_entries() -> None:
    """API endpoint lines become section boundaries."""
    markdown = (
        "# API Reference\n\n"
        "## Users\n\n"
        "GET /api/v1/users\n\n"
        "Returns a list of users.\n\n"
        "POST /api/v1/users\n\n"
        "Creates a new user.\n"
    )
    chunks = chunk_documentation(markdown.encode("utf-8"))

    sections = [chunk.metadata.section for chunk in chunks]
    assert any(section == "GET /api/v1/users" for section in sections)
    assert any(section == "POST /api/v1/users" for section in sections)


def test_documentation_chunker_parses_html_headings() -> None:
    """HTML ``<h1>`` / ``<h2>`` tags are converted to page/section boundaries."""
    html = (
        "<!DOCTYPE html><html><body>"
        "<h1>Installation</h1><p>Install with pip.</p>"
        "<h2>Quick start</h2><p>Run init.</p>"
        "<h1>API Reference</h1><p>Endpoints below.</p>"
        "</body></html>"
    )
    chunks = chunk_documentation(html.encode("utf-8"))

    assert len(chunks) >= 2
    pages = {chunk.metadata.page for chunk in chunks}
    assert "Installation" in pages
    assert "API Reference" in pages


def test_documentation_chunker_parses_pdf_pages() -> None:
    """PDF pages are treated as documentation pages."""
    pdf_bytes = _make_pdf_from_text(
        "# Installation\nInstall the package.\n\n## Quick start\nRun init."
    )
    chunks = chunk_documentation(pdf_bytes)

    assert len(chunks) >= 1
    assert all(chunk.metadata.page for chunk in chunks)


def test_documentation_chunker_rejects_empty_bytes() -> None:
    """Empty byte streams raise IngestionError."""
    with pytest.raises(IngestionError):
        chunk_documentation(b"")


def test_documentation_chunker_rejects_invalid_bytes() -> None:
    """Byte streams that are not valid text raise IngestionError."""
    with pytest.raises(IngestionError):
        chunk_documentation(b"\xff\xfe not valid utf-8")


def test_documentation_chunk_metadata_validates_at_boundary() -> None:
    """Chunks emitted by the documentation chunker validate against the metadata model."""
    markdown = "# Installation\n\nInstall with pip.\n"
    chunks = chunk_documentation(markdown.encode("utf-8"))

    for chunk in chunks:
        assert DocumentationChunkMetadata.model_validate(chunk.metadata.model_dump())
