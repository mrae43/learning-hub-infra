"""Tests for the documentation chunker."""

from collections.abc import Callable
from pathlib import Path

import pytest

from core.exceptions import IngestionError
from core.types.chunk import DocumentationChunkMetadata
from retrieval_qa.chunking.documentation_chunker import chunk_documentation


def _make_pdf_from_text(text: str, pdf_path: Path) -> Path:
    """Build a minimal PDF containing the given text using reportlab."""
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas

    canva = canvas.Canvas(str(pdf_path), pagesize=letter)
    y = 720
    for line in text.splitlines():
        canva.drawString(72, y, line)
        y -= 20
    canva.showPage()
    canva.save()
    return pdf_path


def test_documentation_chunker_splits_markdown_by_headings(temp_file: Callable[..., Path]) -> None:
    """Markdown headings create page/section boundaries."""
    md_path = temp_file(
        b"# Installation\n\n"
        b"Install the package with pip.\n\n"
        b"## Quick start\n\n"
        b"Run the init command.\n\n"
        b"# API Reference\n\n"
        b"Endpoints are described below.\n\n"
        b"## Users\n\n"
        b"Manage users.\n",
        "docs.md",
    )
    chunks = chunk_documentation(md_path)

    assert len(chunks) >= 2
    pages = {chunk.metadata.page for chunk in chunks}
    assert "Installation" in pages
    assert "API Reference" in pages


def test_documentation_chunker_detects_api_entries(temp_file: Callable[..., Path]) -> None:
    """API endpoint lines become section boundaries."""
    md_path = temp_file(
        b"# API Reference\n\n"
        b"## Users\n\n"
        b"GET /api/v1/users\n\n"
        b"Returns a list of users.\n\n"
        b"POST /api/v1/users\n\n"
        b"Creates a new user.\n",
        "api.md",
    )
    chunks = chunk_documentation(md_path)

    sections = [chunk.metadata.section for chunk in chunks]
    assert any(section == "GET /api/v1/users" for section in sections)
    assert any(section == "POST /api/v1/users" for section in sections)


def test_documentation_chunker_parses_html_headings(temp_file: Callable[..., Path]) -> None:
    """HTML ``<h1>`` / ``<h2>`` tags are converted to page/section boundaries."""
    html_path = temp_file(
        b"<!DOCTYPE html><html><body>"
        b"<h1>Installation</h1><p>Install with pip.</p>"
        b"<h2>Quick start</h2><p>Run init.</p>"
        b"<h1>API Reference</h1><p>Endpoints below.</p>"
        b"</body></html>",
        "docs.html",
    )
    chunks = chunk_documentation(html_path)

    assert len(chunks) >= 2
    pages = {chunk.metadata.page for chunk in chunks}
    assert "Installation" in pages
    assert "API Reference" in pages


def test_documentation_chunker_parses_pdf_pages(tmp_path: Path) -> None:
    """PDF pages are treated as documentation pages."""
    pdf_path = _make_pdf_from_text(
        "# Installation\nInstall the package.\n\n## Quick start\nRun init.",
        pdf_path=tmp_path / "docs.pdf",
    )
    chunks = chunk_documentation(pdf_path)

    assert len(chunks) >= 1
    assert all(chunk.metadata.page for chunk in chunks)


def test_documentation_chunker_rejects_empty_file(temp_file: Callable[..., Path]) -> None:
    """Empty files raise IngestionError."""
    empty_path = temp_file(b"", "empty.md")
    with pytest.raises(IngestionError):
        chunk_documentation(empty_path)


def test_documentation_chunker_rejects_invalid_utf8(temp_file: Callable[..., Path]) -> None:
    """Files that are not valid UTF-8 raise IngestionError."""
    bad_path = temp_file(b"\xff\xfe not valid utf-8", "bad.bin")
    with pytest.raises(IngestionError):
        chunk_documentation(bad_path)


def test_documentation_chunk_metadata_validates_at_boundary(temp_file: Callable[..., Path]) -> None:
    """Chunks emitted by the documentation chunker validate against the metadata model."""
    md_path = temp_file(b"# Installation\n\nInstall with pip.\n", "docs.md")
    chunks = chunk_documentation(md_path)

    for chunk in chunks:
        assert DocumentationChunkMetadata.model_validate(chunk.metadata.model_dump())
