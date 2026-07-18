"""Document-type chunker for academic papers (PDF).

Extracts text from a PDF and splits it into section/subsection-aware chunks,
emitting ``PaperChunkMetadata`` for each chunk.
"""

import re
from collections.abc import Sequence
from io import BytesIO

from pydantic import ConfigDict
from pypdf import PdfReader

from core.types.chunk import Chunk, PaperChunkMetadata
from core.types.document import DocumentType
from retrieval_qa.chunking.base import DocumentChunker, register_chunker

# Matches lines that look like paper section headers, e.g.:
#   "1 Introduction"
#   "2.3 Experimental Setup"
#   "4 Results and Discussion"
# Captures the section number prefix and the title text.
_SECTION_PATTERN = re.compile(
    r"^(?P<prefix>(?P<major>\d+)(?:\.(?P<minor>\d+))?\s+)(?P<title>[A-Z][A-Za-z0-9\s\-:/]+$)",
    re.MULTILINE,
)


class PaperChunk(Chunk):
    """A single chunk produced by the paper chunker."""

    model_config = ConfigDict(extra="forbid")

    metadata: PaperChunkMetadata


def _count_tokens(text: str) -> int:
    """Approximate token count for chunk sizing.

    Uses a simple whitespace split; this is sufficient for MVP chunk ordering
    and sanity checks. More precise counting can be swapped in later without
    changing the chunker interface.
    """
    return max(1, len(text.split()))


def _split_into_sections(text: str) -> list[tuple[str, str | None, str]]:
    """Split paper text into (section, subsection, body) pieces.

    The heuristic treats any line matching ``_SECTION_PATTERN`` as a header.
    A section with a single major number (e.g. ``3``) becomes a top-level
    section; a dotted number (e.g. ``3.1``) becomes a subsection of the
    current top-level section.
    """
    sections: list[tuple[str, str | None, str]] = []
    current_section = "Introduction"
    current_subsection: str | None = None
    buffer_parts: list[str] = []

    def flush() -> None:
        if buffer_parts:
            body = "\n".join(buffer_parts).strip()
            if body:
                sections.append((current_section, current_subsection, body))
            buffer_parts.clear()

    for line in text.splitlines():
        match = _SECTION_PATTERN.match(line.strip())
        if match:
            flush()
            title = match.group("title").strip()
            minor = match.group("minor")
            if minor is None:
                current_section = title
                current_subsection = None
            else:
                current_subsection = title
            continue
        buffer_parts.append(line)

    flush()
    return sections


def chunk_paper(pdf_bytes: bytes) -> list[PaperChunk]:
    """Extract text from a paper PDF and split it into chunks.

    Args:
        pdf_bytes: Raw PDF file contents.

    Returns:
        A list of chunks ordered by their appearance in the document.

    Raises:
        IngestionError: The PDF could not be read.
    """
    from core.exceptions import IngestionError

    try:
        reader = PdfReader(BytesIO(pdf_bytes))
    except Exception as exc:
        raise IngestionError(f"Failed to parse PDF: {exc}") from exc

    chunks: list[PaperChunk] = []
    for page_index, page in enumerate(reader.pages):
        page_number = page_index + 1
        try:
            page_text = page.extract_text() or ""
        except Exception as exc:
            raise IngestionError(f"Failed to extract text from page {page_number}: {exc}") from exc

        sections = _split_into_sections(page_text)
        for section, subsection, body in sections:
            if not body.strip():
                continue
            metadata = PaperChunkMetadata(
                section=section,
                subsection=subsection,
                page=page_number,
            )
            chunks.append(
                PaperChunk(
                    content=body,
                    metadata=metadata,
                    token_count=_count_tokens(body),
                )
            )

    # If the heuristic found no section headers, emit the whole document as a
    # single chunk so the pipeline never produces zero chunks.
    if not chunks:
        full_text = "\n".join((page.extract_text() or "") for page in reader.pages)
        if not full_text.strip():
            raise IngestionError("PDF contains no extractable text")
        chunks.append(
            PaperChunk(
                content=full_text,
                metadata=PaperChunkMetadata(
                    section="Document",
                    subsection=None,
                    page=1,
                ),
                token_count=_count_tokens(full_text),
            )
        )

    return chunks


class PaperChunker(DocumentChunker):
    """Chunker for academic papers."""

    metadata_model = PaperChunkMetadata

    def chunk(self, file_bytes: bytes) -> Sequence[Chunk]:
        """Chunk a paper PDF into section-aware pieces."""
        return chunk_paper(file_bytes)


register_chunker(DocumentType.PAPER, PaperChunker)


__all__ = ["PaperChunk", "PaperChunker", "chunk_paper"]
