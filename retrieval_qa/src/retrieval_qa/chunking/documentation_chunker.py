"""Document-type chunker for documentation (Markdown, HTML, PDF).

Extracts text from a documentation file and splits it into page/section-aware
chunks, emitting ``DocumentationChunkMetadata`` for each chunk.

Pages are identified by top-level headings (Markdown ``#``, HTML ``<h1>``, or
PDF page breaks). Sections include sub-headings and API endpoint lines such as
``GET /api/v1/users``.
"""

import re
from collections.abc import Sequence
from enum import StrEnum
from io import BytesIO

from pydantic import ConfigDict
from pypdf import PdfReader

from core.exceptions import IngestionError
from core.types.chunk import Chunk, DocumentationChunkMetadata
from core.types.document import DocumentType
from retrieval_qa._utils import count_tokens
from retrieval_qa.chunking._html_utils import _BaseHTMLTextExtractor
from retrieval_qa.chunking.base import DocumentChunker, register_chunker


class DocumentationFormat(StrEnum):
    """Supported documentation file formats for ingestion."""

    PDF = "pdf"
    HTML = "html"
    MARKDOWN = "markdown"


_HTTP_METHODS = ("GET", "POST")

# Matches Markdown headings like ``# Page`` or ``## Section``.
_HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+?)$")

# Matches API endpoint lines like ``GET /api/v1/users`` or ``POST /items``.
_API_ENTRY_PATTERN = re.compile(
    r"^(" + "|".join(_HTTP_METHODS) + r")\s+(/\S+)$",
)


class DocumentationChunk(Chunk):
    """A single chunk produced by the documentation chunker."""

    model_config = ConfigDict(extra="forbid")

    metadata: DocumentationChunkMetadata


class _HTMLTextExtractor(_BaseHTMLTextExtractor):
    """Convert HTML to Markdown-style headings so the text splitter can run."""

    def __init__(self) -> None:
        super().__init__()
        self._heading_level: int | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "head":
            self._skip = True
        elif tag in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            self._heading_level = int(tag[1])
            self._parts.append("\n")
            self._parts.append("#" * self._heading_level + " ")
        else:
            super().handle_starttag(tag, attrs)

    def handle_endtag(self, tag: str) -> None:
        if tag == "head":
            self._skip = False
        elif tag in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            self._heading_level = None
            self._parts.append("\n")
        else:
            super().handle_endtag(tag)


def _detect_format(file_bytes: bytes) -> DocumentationFormat:
    """Detect whether ``file_bytes`` is PDF, HTML, or Markdown/text."""
    if file_bytes.startswith(b"%PDF"):
        return DocumentationFormat.PDF

    head = file_bytes[:2048].lower()
    if b"<!doctype html" in head or b"<html" in head:
        return DocumentationFormat.HTML

    return DocumentationFormat.MARKDOWN


def _extract_text_from_html(html_bytes: bytes) -> str:
    """Strip HTML tags and convert headings to Markdown-style markers."""
    try:
        text = html_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise IngestionError(f"Failed to decode HTML as UTF-8: {exc}") from exc

    extractor = _HTMLTextExtractor()
    extractor.feed(text)
    extractor.close()
    return extractor.get_text()


def _extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """Extract text from a PDF, inserting a page marker before each page.

    The page markers let the Markdown-style splitter treat each PDF page as a
    top-level documentation page.
    """
    try:
        reader = PdfReader(BytesIO(pdf_bytes))
    except Exception as exc:
        raise IngestionError(f"Failed to parse PDF: {exc}") from exc

    parts: list[str] = []
    for page_index, page in enumerate(reader.pages, start=1):
        try:
            page_text = page.extract_text() or ""
        except Exception as exc:
            raise IngestionError(f"Failed to extract text from page {page_index}: {exc}") from exc

        stripped = page_text.strip()
        if stripped:
            parts.append(f"# Page {page_index}\n\n{stripped}")

    return "\n\n".join(parts)


def _split_documentation(text: str) -> list[tuple[str, str | None, str]]:
    """Split documentation text into page/section-aware pieces.

    Returns a list of ``(page, section, body)`` tuples. ``page`` is the current
    top-level heading (or ``"Document"`` if none is found). ``section`` is the
    most recent sub-heading or API endpoint line, or ``None`` when none has been
    seen within the current page.
    """
    sections: list[tuple[str, str | None, str]] = []
    current_page = "Document"
    current_section: str | None = None
    buffer_parts: list[str] = []

    def flush() -> None:
        if buffer_parts:
            body = "\n".join(buffer_parts).strip()
            if body:
                sections.append((current_page, current_section, body))
            buffer_parts.clear()

    for raw_line in text.splitlines():
        line = raw_line.strip()
        heading_match = _HEADING_PATTERN.match(line)
        if heading_match:
            flush()
            level = len(heading_match.group(1))
            title = heading_match.group(2).strip()
            if level == 1:
                current_page = title
                current_section = None
            else:
                current_section = title
            continue

        api_match = _API_ENTRY_PATTERN.match(line)
        if api_match:
            flush()
            current_section = f"{api_match.group(1).upper()} {api_match.group(2)}"
            buffer_parts.append(line)
            continue

        buffer_parts.append(raw_line)

    flush()
    return sections


def _extract_text(file_bytes: bytes, document_format: DocumentationFormat) -> str:
    """Extract plain text from a documentation file in the given format."""
    match document_format:
        case DocumentationFormat.PDF:
            return _extract_text_from_pdf(file_bytes)
        case DocumentationFormat.HTML:
            return _extract_text_from_html(file_bytes)
        case DocumentationFormat.MARKDOWN:
            try:
                return file_bytes.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise IngestionError(f"Failed to decode Markdown as UTF-8: {exc}") from exc
        case _ as unreachable:
            raise IngestionError(f"Unsupported documentation format: {unreachable}")


def chunk_documentation(file_bytes: bytes) -> list[DocumentationChunk]:
    """Extract text from a documentation file and split it into chunks.

    Args:
        file_bytes: Raw Markdown, HTML, or PDF file contents.

    Returns:
        A list of chunks ordered by their appearance in the document.

    Raises:
        IngestionError: The file could not be parsed or contains no extractable
            text.
    """
    document_format = _detect_format(file_bytes)
    raw_text = _extract_text(file_bytes, document_format)

    if not raw_text.strip():
        raise IngestionError("Documentation contains no extractable text")

    sections = _split_documentation(raw_text)
    if not sections:
        raise IngestionError("Documentation contains no extractable text")

    chunks: list[DocumentationChunk] = []
    for page, section, body in sections:
        if not body.strip():
            continue
        metadata = DocumentationChunkMetadata(page=page, section=section)
        chunks.append(
            DocumentationChunk(
                content=body,
                metadata=metadata,
                token_count=count_tokens(body),
            )
        )

    # If no section produced usable content, emit the whole document as a single
    # chunk so the pipeline never produces zero chunks.
    if not chunks:
        chunks.append(
            DocumentationChunk(
                content=raw_text.strip(),
                metadata=DocumentationChunkMetadata(page="Document", section=None),
                token_count=count_tokens(raw_text),
            )
        )

    return chunks


class DocumentationChunker(DocumentChunker):
    """Chunker for documentation."""

    metadata_model = DocumentationChunkMetadata

    def chunk(self, file_bytes: bytes) -> Sequence[Chunk]:
        """Chunk a Markdown/HTML/PDF documentation file into page-aware pieces."""
        return chunk_documentation(file_bytes)


register_chunker(DocumentType.DOCUMENTATION, DocumentationChunker)


__all__ = ["DocumentationChunk", "DocumentationChunker", "chunk_documentation"]
