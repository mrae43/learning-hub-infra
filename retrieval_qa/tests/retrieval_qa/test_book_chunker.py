"""Tests for the book chunker."""

import io
import zipfile

import pytest

from core.exceptions import IngestionError
from core.types.chunk import BookChunkMetadata
from retrieval_qa.chunking.book_chunker import chunk_book


def _make_epub(chapters: list[tuple[str, str, str]]) -> bytes:
    """Build an in-memory EPUB from chapter specs.

    Each tuple is ``(file_name, xhtml_body, spine_idref)``.

    The *xhtml_body* is placed directly inside ``<body>...</body>``.
    All chapters share the same OPF manifest/spine structure.
    """
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("mimetype", "application/epub+zip", compress_type=zipfile.ZIP_STORED)
        zf.writestr(
            "META-INF/container.xml",
            '<?xml version="1.0" encoding="UTF-8"?>'
            '<container version="1.0"'
            ' xmlns="urn:oasis:names:tc:opendocument:xmlns:container">'
            "<rootfiles>"
            '<rootfile full-path="OEBPS/content.opf"'
            ' media-type="application/oebps-package+xml"/>'
            "</rootfiles>"
            "</container>",
        )
        items_xml = ""
        spine_xml = ""
        for file_name, _body, spine_idref in chapters:
            items_xml += (
                f'<item id="{spine_idref}" href="{file_name}" media-type="application/xhtml+xml"/>'
            )
            spine_xml += f'<itemref idref="{spine_idref}"/>'
        zf.writestr(
            "OEBPS/content.opf",
            '<?xml version="1.0" encoding="UTF-8"?>'
            '<package xmlns="http://www.idpf.org/2007/opf"'
            ' unique-identifier="bookid" version="2.0">'
            "<metadata>"
            '<dc:title xmlns:dc="http://purl.org/dc/elements/1.1/">Test Book</dc:title>'
            '<dc:identifier xmlns:dc="http://purl.org/dc/elements/1.1/"'
            ' id="bookid">test-001</dc:identifier>'
            '<dc:language xmlns:dc="http://purl.org/dc/elements/1.1/">en</dc:language>'
            "</metadata>"
            f"<manifest>{items_xml}</manifest>"
            f"<spine>{spine_xml}</spine>"
            "</package>",
        )
        for file_name, body, _spine_idref in chapters:
            zf.writestr(
                f"OEBPS/{file_name}",
                '<?xml version="1.0" encoding="utf-8"?>'
                "<!DOCTYPE html>"
                '<html xmlns="http://www.w3.org/1999/xhtml">'
                "<head><title>Test</title></head>"
                f"<body>{body}</body></html>",
            )
    buffer.seek(0)
    return buffer.read()


def test_book_chunker_emits_chapters(sample_book_pdf: bytes) -> None:
    """chunk_book extracts text and splits it into ordered chunks."""
    chunks = chunk_book(sample_book_pdf)

    assert len(chunks) >= 1
    for chunk in chunks:
        assert chunk.content.strip()
        assert chunk.token_count > 0
        assert isinstance(chunk.metadata.chapter, int)
        assert chunk.metadata.chapter >= 1
        assert chunk.metadata.heading is None or isinstance(chunk.metadata.heading, str)


def test_book_chunker_detects_chapter_boundaries(sample_book_pdf: bytes) -> None:
    """Detected chapter boundaries are reflected in chunk metadata."""
    chunks = chunk_book(sample_book_pdf)

    chapters = [chunk.metadata.chapter for chunk in chunks]
    assert 1 in chapters
    assert 2 in chapters


def test_book_chunker_epub(sample_book_epub: bytes) -> None:
    """chunk_book can parse an EPUB with chapter structure."""
    chunks = chunk_book(sample_book_epub)

    assert len(chunks) >= 1
    chapters = {chunk.metadata.chapter for chunk in chunks}
    assert 1 in chapters
    assert all(isinstance(chunk.metadata.chapter, int) for chunk in chunks)


def test_book_chunker_rejects_invalid_bytes() -> None:
    """A non-PDF/non-EPUB byte stream raises IngestionError."""
    with pytest.raises(IngestionError):
        chunk_book(b"not a book")


def test_book_chunk_metadata_validates_at_boundary(sample_book_pdf: bytes) -> None:
    """Chunks emitted by the book chunker validate against BookChunkMetadata."""
    chunks = chunk_book(sample_book_pdf)
    for chunk in chunks:
        # model_validate should succeed and extra='forbid' means unknown keys
        # would have already raised during construction.
        assert BookChunkMetadata.model_validate(chunk.metadata.model_dump())


def test_epub_chunks_by_heading_tags() -> None:
    """EPUB with ``<h1>`` headings (no 'Chapter N' text) splits correctly."""
    epub = _make_epub(
        [
            ("ch1.xhtml", "<h1>The Beginning</h1><p>First chapter content.</p>", "ch1"),
            ("ch2.xhtml", "<h1>The Middle</h1><p>Second chapter content.</p>", "ch2"),
            ("ch3.xhtml", "<h1>The End</h1><p>Third chapter content.</p>", "ch3"),
        ]
    )
    chunks = chunk_book(epub)

    assert len(chunks) == 3
    assert [c.metadata.chapter for c in chunks] == [1, 2, 3]
    assert [c.metadata.heading for c in chunks] == [
        "The Beginning",
        "The Middle",
        "The End",
    ]
    assert all("Chapter" not in c.content for c in chunks)


def test_epub_heading_content_included_in_chunk() -> None:
    """Heading text appears in both chunk content and metadata."""
    epub = _make_epub(
        [
            ("ch1.xhtml", "<h1>Part One</h1><p>Some text.</p>", "ch1"),
        ]
    )
    chunks = chunk_book(epub)

    assert len(chunks) == 1
    assert chunks[0].metadata.heading == "Part One"
    assert chunks[0].content.startswith("Part One")
    assert "Some text." in chunks[0].content


def test_epub_subheadings_same_chapter() -> None:
    """``<h2>`` sub-headings produce additional chunks within the same chapter."""
    epub = _make_epub(
        [
            (
                "ch1.xhtml",
                "<h1>Chapter 1</h1>"
                "<p>Introduction.</p>"
                "<h2>Section A</h2><p>Content A.</p>"
                "<h2>Section B</h2><p>Content B.</p>",
                "ch1",
            ),
        ]
    )
    chunks = chunk_book(epub)

    assert len(chunks) == 3
    assert [c.metadata.chapter for c in chunks] == [1, 1, 1]
    assert [c.metadata.heading for c in chunks] == ["Chapter 1", "Section A", "Section B"]


def test_epub_fallback_to_regex_when_no_headings() -> None:
    """EPUB without heading tags falls back to regex heuristics."""
    epub = _make_epub(
        [
            (
                "ch1.xhtml",
                "<p>Chapter 1</p><p>The beginning of the story.</p>",
                "ch1",
            ),
            (
                "ch2.xhtml",
                "<p>Chapter 2</p><p>The story continues.</p>",
                "ch2",
            ),
        ]
    )
    chunks = chunk_book(epub)

    # Regex should detect "Chapter 1" / "Chapter 2" as chapter boundaries
    chapters = {c.metadata.chapter for c in chunks}
    assert 1 in chapters
    assert 2 in chapters
    # Exactly one chunk per chapter (no subheadings in plain <p> text)
    assert len(chunks) == 2
    assert [c.metadata.heading for c in chunks] == [None, None]
