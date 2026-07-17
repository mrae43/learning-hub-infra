"""Document-type chunker for books (PDF or EPUB).

Extracts text from a book and splits it into chapter/heading-aware chunks,
emitting ``BookChunkMetadata`` for each chunk.
"""

import os
import re
import xml.etree.ElementTree as ET
import zipfile
from html.parser import HTMLParser
from io import BytesIO

from pydantic import BaseModel, ConfigDict
from pypdf import PdfReader

from core.exceptions import IngestionError
from core.types.chunk import BookChunkMetadata

# Matches lines that look like chapter headers, e.g.:
#   "Chapter 1"
#   "Chapter 1: The Beginning"
#   "1 The Beginning"
_CHAPTER_PATTERN = re.compile(
    r"^(?:Chapter[ \t]+(\d+)(?:[ \t]*[:.\-]?[ \t]*(.*))?|(\d+)[ \t]+([A-Z][A-Za-z0-9\s\-:/]+))$",
    re.IGNORECASE | re.MULTILINE,
)

# Matches lines that look like headings inside a chapter, e.g.:
#   "Summary"
#   "1.1 The Roman Forum"
_HEADING_PATTERN = re.compile(
    r"^\s*((?:\d+(?:\.\d+)*\s*[:.\-]\s*)?[A-Z][A-Za-z0-9\s\-:/]+?)\s*$",
    re.MULTILINE,
)


class BookChunk(BaseModel):
    """A single chunk produced by the book chunker."""

    model_config = ConfigDict(extra="forbid")

    content: str
    metadata: BookChunkMetadata
    token_count: int


class _HTMLTextExtractor(HTMLParser):
    """Strip tags from EPUB HTML bodies while preserving some structure."""

    def __init__(self) -> None:
        super().__init__()
        self._parts: list[str] = []
        self._skip = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style"}:
            self._skip = True
        elif tag in {"p", "div", "h1", "h2", "h3", "h4", "li", "br"}:
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style"}:
            self._skip = False
        elif tag in {"p", "div", "h1", "h2", "h3", "h4", "li"}:
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._skip:
            self._parts.append(data)

    def get_text(self) -> str:
        text = "".join(self._parts)
        return re.sub(r"\n\s*\n+", "\n\n", text).strip()


def _count_tokens(text: str) -> int:
    """Approximate token count for chunk sizing.

    Uses a simple whitespace split; this is sufficient for MVP chunk ordering
    and sanity checks. More precise counting can be swapped in later without
    changing the chunker interface.
    """
    return max(1, len(text.split()))


def _looks_like_heading(line: str) -> bool:
    """Return True if ``line`` is plausibly a heading rather than body text.

    Heuristic: headings are relatively short, do not end with sentence
    punctuation, and are not entirely lowercase.
    """
    stripped = line.strip()
    if len(stripped) > 100:
        return False
    if stripped.endswith((".", ",", "!", "?")):
        return False
    return stripped != stripped.lower()


def _split_chapter_body(
    chapter: int,
    body: str,
    chapter_title: str | None = None,
) -> list[tuple[int, str | None, str]]:
    """Split a chapter's body into heading-bounded pieces.

    Returns a list of ``(chapter, heading, body)`` tuples. ``heading`` is
    ``chapter_title`` for the text before the first subheading in the chapter,
    or ``None`` if no chapter title was detected.
    """
    sections: list[tuple[int, str | None, str]] = []
    current_heading: str | None = chapter_title
    buffer_parts: list[str] = []

    def flush() -> None:
        if buffer_parts:
            content = "\n".join(buffer_parts).strip()
            if content:
                sections.append((chapter, current_heading, content))
            buffer_parts.clear()

    for line in body.splitlines():
        match = _HEADING_PATTERN.match(line.strip())
        if match and _looks_like_heading(line):
            flush()
            current_heading = match.group(1).strip()
            continue
        buffer_parts.append(line)

    flush()
    return sections


def _chapter_title(match: re.Match[str]) -> str | None:
    """Extract a chapter title from a chapter-header regex match if present."""
    title = match.group(2) or match.group(4)
    return title.strip() if title else None


def _split_into_chapters(text: str) -> list[tuple[int, str | None, str]]:
    """Split book text into chapter/heading-aware pieces.

    Detects chapter boundaries, then further splits each chapter by
    subheadings. If no chapter boundaries are found, the whole text is treated
    as chapter 1.
    """
    matches = list(_CHAPTER_PATTERN.finditer(text))
    if not matches:
        return _split_chapter_body(1, text)

    sections: list[tuple[int, str | None, str]] = []
    for index, match in enumerate(matches):
        chapter_number = int(match.group(1) or match.group(3))
        title = _chapter_title(match)
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        chapter_text = text[start:end]
        sections.extend(_split_chapter_body(chapter_number, chapter_text, title))

    return sections


def _extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """Extract text from a PDF byte stream."""
    try:
        reader = PdfReader(BytesIO(pdf_bytes))
    except Exception as exc:
        raise IngestionError(f"Failed to parse PDF: {exc}") from exc

    parts: list[str] = []
    for page in reader.pages:
        try:
            page_text = page.extract_text() or ""
        except Exception as exc:
            raise IngestionError("Failed to extract text from page") from exc
        parts.append(page_text)

    return "\n\n".join(parts)


_OPF_NS = "http://www.idpf.org/2007/opf"


def _opf_root_path(container_xml: str) -> str:
    """Parse META-INF/container.xml and return the full-path of the OPF."""
    root = ET.fromstring(container_xml)
    ns = {"c": "urn:oasis:names:tc:opendocument:xmlns:container"}
    elem = root.find(".//c:rootfile", ns)
    if elem is None:
        raise IngestionError("EPUB container.xml missing rootfile element")
    return elem.get("full-path", "")


def _parse_opf(opf_content: str) -> tuple[list[str], dict[str, str]]:
    """Parse content.opf, return (spine idref list, manifest {id -> href})."""
    root = ET.fromstring(opf_content)
    has_ns = root.tag.startswith(f"{{{_OPF_NS}}}")

    if has_ns:
        manifest_items = root.findall(f".//{{{_OPF_NS}}}manifest/{{{_OPF_NS}}}item")
        spine_refs = root.findall(f".//{{{_OPF_NS}}}spine/{{{_OPF_NS}}}itemref")
    else:
        manifest_items = root.findall(".//manifest/item")
        spine_refs = root.findall(".//spine/itemref")

    item_map: dict[str, str] = {}
    for item in manifest_items:
        item_id = item.get("id", "")
        href = item.get("href", "")
        if item_id:
            item_map[item_id] = href

    spine_order: list[str] = []
    for ref in spine_refs:
        idref = ref.get("idref", "")
        if idref:
            spine_order.append(idref)

    return spine_order, item_map


def _extract_text_from_epub(epub_bytes: bytes) -> str:
    """Extract text from an EPUB byte stream."""
    try:
        with zipfile.ZipFile(BytesIO(epub_bytes)) as archive:
            container_xml = archive.read("META-INF/container.xml").decode("utf-8")
            opf_path = _opf_root_path(container_xml)
            opf_dir = os.path.dirname(opf_path)

            opf_content = archive.read(opf_path).decode("utf-8")
            spine_order, item_map = _parse_opf(opf_content)

            parts: list[str] = []
            for item_id in spine_order:
                href = item_map.get(item_id)
                if href is None:
                    continue
                file_path = os.path.normpath(os.path.join(opf_dir, href)) if opf_dir else href
                try:
                    raw = archive.read(file_path).decode("utf-8")
                except KeyError:
                    raw = archive.read(href).decode("utf-8")

                extractor = _HTMLTextExtractor()
                extractor.feed(raw)
                text = extractor.get_text()
                if text:
                    parts.append(text)

            return "\n\n".join(parts)
    except Exception as exc:
        raise IngestionError(f"Failed to parse EPUB: {exc}") from exc


def _detect_format(file_bytes: bytes) -> str:
    """Detect whether ``file_bytes`` is a PDF or EPUB."""
    if file_bytes.startswith(b"%PDF"):
        return "pdf"

    if file_bytes.startswith(b"PK"):
        try:
            with zipfile.ZipFile(BytesIO(file_bytes)) as archive:
                if "mimetype" in archive.namelist():
                    mimetype = archive.read("mimetype").strip().lower()
                    if mimetype == b"application/epub+zip":
                        return "epub"
        except zipfile.BadZipFile:
            pass

    raise IngestionError("Unsupported book format: expected PDF or EPUB")


def chunk_book(file_bytes: bytes) -> list[BookChunk]:
    """Extract text from a book PDF or EPUB and split it into chunks.

    Args:
        file_bytes: Raw PDF or EPUB file contents.

    Returns:
        A list of chunks ordered by their appearance in the document.

    Raises:
        IngestionError: The file could not be parsed or is not a supported
            book format.
    """
    document_format = _detect_format(file_bytes)
    if document_format == "pdf":
        text = _extract_text_from_pdf(file_bytes)
    else:
        text = _extract_text_from_epub(file_bytes)

    if not text.strip():
        raise IngestionError("Book contains no extractable text")

    chunks: list[BookChunk] = []
    for chapter, heading, body in _split_into_chapters(text):
        if not body.strip():
            continue
        metadata = BookChunkMetadata(chapter=chapter, heading=heading)
        chunks.append(
            BookChunk(
                content=body,
                metadata=metadata,
                token_count=_count_tokens(body),
            )
        )

    # If the heuristic found no usable sections, emit the whole document as a
    # single chunk so the pipeline never produces zero chunks.
    if not chunks:
        chunks.append(
            BookChunk(
                content=text,
                metadata=BookChunkMetadata(chapter=1, heading=None),
                token_count=_count_tokens(text),
            )
        )

    return chunks


__all__ = ["BookChunk", "chunk_book"]
