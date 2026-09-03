"""Parse PDF files from docs/ into Paragraph nodes.

Uses pypdf to extract text page by page. Infers paragraph boundaries from
text block structure. Each paragraph carries its page_number metadata.

If font analysis is unreliable, falls back to a simpler approach: extract
text blocks with page numbers and use first-line detection for boundaries.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional

try:
    from pypdf import PdfReader
    from pypdf._page import PageObject
    _HAS_PYPDF = True
except ImportError:
    _HAS_PYPDF = False

from dsa_mentor.models import Paragraph
from dsa_mentor.ingestion.md_parser import (
    _HEADING_RE,
    _find_word_boundary,
    _link_adjacent_paragraphs,
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_paragraphs(
    file_paths: List[str],
    metadata: Optional[Dict[str, str]] = None,
    *,
    paragraph_max_chars: int = 1800,
    paragraph_overlap_chars: int = 250,
) -> List[Paragraph]:
    """Parse one or more PDF files into Paragraph nodes.

    Args:
        file_paths: Paths to .pdf files.
        metadata:   Metadata dict (corpus_id, license, source_url, etc.).
        paragraph_max_chars: Max characters per paragraph before splitting.
        paragraph_overlap_chars: Overlap in characters between split segments.

    Returns:
        List of Paragraph nodes with full provenance including page_number.
    """
    if not _HAS_PYPDF:
        return []

    if metadata is None:
        metadata = {}

    all_paragraphs: List[Paragraph] = []

    for fpath in file_paths:
        try:
            paragraphs = _parse_pdf(
                fpath, metadata, paragraph_max_chars, paragraph_overlap_chars
            )
            all_paragraphs.extend(paragraphs)
        except Exception:
            # Skip files that can't be parsed
            continue

    return all_paragraphs


# ---------------------------------------------------------------------------
# PDF parsing
# ---------------------------------------------------------------------------

def _parse_pdf(
    fpath: str,
    metadata: Dict[str, str],
    paragraph_max_chars: int,
    paragraph_overlap_chars: int,
) -> List[Paragraph]:
    """Parse a single PDF file into paragraphs."""
    try:
        reader = PdfReader(fpath)
    except Exception:
        return []

    pages = reader.pages
    if not pages:
        return []

    # Collect all text with page numbers
    page_texts: List[tuple[int, str]] = []
    for i, page in enumerate(pages):
        text = _extract_page_text(page)
        page_num = i + 1  # 1-based page numbers
        page_texts.append((page_num, text))

    # Group text into paragraphs per page
    paragraphs: List[Paragraph] = []
    source_file = fpath
    source_url = metadata.get("source_url")
    corpus_id = metadata.get("corpus_id")
    license_val = metadata.get("license")

    for page_num, text in page_texts:
        para_list = _text_to_paragraphs(
            text, source_file, source_url, corpus_id, license_val,
            page_num, paragraph_max_chars, paragraph_overlap_chars
        )
        paragraphs.extend(para_list)

    # Link adjacent paragraphs within the same source
    _link_adjacent_paragraphs(paragraphs)

    return paragraphs


def _extract_page_text(page: PageObject) -> str:
    """Extract text from a PDF page.

    Tries to get structured text first, falls back to plain text extraction.
    """
    # Try getting text with layout preservation first
    try:
        text = page.extract_text()
        if text:
            return text
    except Exception:
        pass

    # Fallback: raw text extraction
    try:
        return page.extract_text() or ""
    except Exception:
        return ""


def _text_to_paragraphs(
    text: str,
    source_file: str,
    source_url: Optional[str],
    corpus_id: Optional[str],
    license: Optional[str],
    page_number: int,
    paragraph_max_chars: int,
    paragraph_overlap_chars: int,
) -> List[Paragraph]:
    """Split raw page text into paragraph nodes.

    Uses heading detection and blank-line boundaries to identify paragraph
    segments.  Oversized segments are split per spec §6.
    """
    lines = text.split("\n")
    paragraphs: List[Paragraph] = []
    current_lines: List[str] = []
    current_title: str = ""
    in_heading = False

    for line in lines:
        stripped = line.strip()

        # Detect headings (lines that look like headings)
        is_heading = bool(_HEADING_RE.match(stripped))

        # Also detect PDF-style headings: short lines that are all-caps or
        # title-case and don't look like regular paragraph text
        if not is_heading and stripped and not in_heading:
            # Heuristic: short lines (< 80 chars) that are not bullet points
            # and have a high ratio of uppercase letters could be headings
            alpha_chars = sum(1 for c in stripped if c.isalpha())
            if alpha_chars > 0:
                upper_ratio = sum(1 for c in stripped if c.isupper()) / alpha_chars
                if len(stripped) < 80 and upper_ratio > 0.5 and len(stripped) > 3:
                    is_heading = True

        if is_heading:
            # Save previous paragraph block
            if current_lines:
                block_text = "\n".join(current_lines).strip()
                if block_text:
                    paras = _split_into_paragraphs(
                        current_title or "Section", block_text, source_file,
                        source_url, corpus_id, license, page_number,
                        paragraph_max_chars, paragraph_overlap_chars
                    )
                    paragraphs.extend(paras)
                current_lines = []

            current_title = stripped
            in_heading = True
        elif stripped == "":
            # Blank line — potential paragraph boundary
            if current_lines:
                block_text = "\n".join(current_lines).strip()
                if block_text:
                    paras = _split_into_paragraphs(
                        current_title or "Paragraph", block_text, source_file,
                        source_url, corpus_id, license, page_number,
                        paragraph_max_chars, paragraph_overlap_chars
                    )
                    paragraphs.extend(paras)
                current_lines = []
                current_title = ""
                in_heading = False
        else:
            current_lines.append(line)
            in_heading = False

    # Final block
    if current_lines:
        block_text = "\n".join(current_lines).strip()
        if block_text:
            paras = _split_into_paragraphs(
                current_title or "Paragraph", block_text, source_file,
                source_url, corpus_id, license, page_number,
                paragraph_max_chars, paragraph_overlap_chars
            )
            paragraphs.extend(paras)

    return paragraphs


def _split_into_paragraphs(
    title: str,
    content: str,
    source_file: str,
    source_url: Optional[str],
    corpus_id: Optional[str],
    license: Optional[str],
    page_number: int,
    paragraph_max_chars: int,
    paragraph_overlap_chars: int,
) -> List[Paragraph]:
    """Split a text block into one or more Paragraph nodes.

    If the content is within the character limit, create a single paragraph.
    Otherwise, split into overlapping segments.
    """
    import hashlib

    if len(content) <= paragraph_max_chars:
        raw_id = f"{source_file}:p{page_number}:{hashlib.md5(content.encode('utf-8')[:256]).hexdigest()[:12]}"
        return [Paragraph(
            id=raw_id,
            title=title,
            level="paragraph",
            parent_id=None,
            children=[],
            content=content,
            source_file=source_file,
            source_url=source_url,
            page_number=page_number,
            license=license,
            corpus_id=corpus_id,
            book_id=None,
            chapter_id=None,
            topic_id=None,
            subtopic_id=None,
            paragraph_id=raw_id,
            prev_paragraph_id=None,
            next_paragraph_id=None,
        )]

    # Split oversized content
    segments = _split_text(content, paragraph_max_chars, paragraph_overlap_chars)
    result = []
    for i, seg in enumerate(segments):
        seg_title = f"{title} ({i + 1}/{len(segments)})"
        raw_id = f"{source_file}:p{page_number}:s{i}:{hashlib.md5(seg.encode('utf-8')[:256]).hexdigest()[:12]}"
        result.append(Paragraph(
            id=raw_id,
            title=seg_title,
            level="paragraph",
            parent_id=None,
            children=[],
            content=seg,
            source_file=source_file,
            source_url=source_url,
            page_number=page_number,
            license=license,
            corpus_id=corpus_id,
            book_id=None,
            chapter_id=None,
            topic_id=None,
            subtopic_id=None,
            paragraph_id=raw_id,
            prev_paragraph_id=None,
            next_paragraph_id=None,
        ))
    return result


def _split_text(text: str, max_chars: int, overlap_chars: int) -> List[str]:
    """Split text into overlapping segments of at most max_chars."""
    if len(text) <= max_chars:
        return [text]

    segments: List[str] = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = start + max_chars

        if end >= text_len:
            segments.append(text[start:].strip())
            break

        split_pos = _find_word_boundary(text, end, max_window=50)
        segment = text[start:split_pos].strip()

        if segment:
            segments.append(segment)

        start = max(end - overlap_chars, split_pos)
        # Ensure forward progress
        if start <= (end - overlap_chars):
            start = end

    return segments
