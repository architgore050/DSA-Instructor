"""Parse .md files from docs/ into Paragraph nodes.

Supports markdown headings (## → chapter/topic, ### → subtopic, #### → paragraph).
Splits oversized paragraphs per config paragraph_max_chars with paragraph_overlap_chars overlap.
Handles edge cases: nested headings, code blocks, empty sections.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from dsa_mentor.models import Paragraph


@dataclass
class _Heading:
    """Parsed heading with its level, title, and start line."""
    level: int       # 1–6 (# → 1, ## → 2, etc.)
    title: str
    start: int       # 0-based line index where heading starts


@dataclass
class _Block:
    """Text block between two headings (or start/end of file)."""
    heading: Optional[_Heading]
    lines: List[str]
    start_line: int


@dataclass
class _SubtopicBlock:
    """A subtopic (### heading) containing sub-blocks (####+ headings or content)."""
    heading: Optional[_Heading]
    sub_blocks: List[_Block]


# ---------------------------------------------------------------------------
# Heading regex
# ---------------------------------------------------------------------------

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$")

# Markdown code fence markers
_CODE_FENCE_RE = re.compile(r"^(`{3,}|~{3,})")


def parse_paragraphs(
    file_paths: List[str],
    metadata: Optional[Dict[str, str]] = None,
    *,
    paragraph_max_chars: int = 1800,
    paragraph_overlap_chars: int = 250,
) -> List[Paragraph]:
    """Parse one or more .md files into Paragraph nodes.

    Args:
        file_paths: Absolute or relative paths to .md files.
        metadata:   Shared metadata dict with keys like source_type, corpus_id,
                    license, source_url. Applied to every paragraph.
        paragraph_max_chars: Max characters per paragraph before splitting (spec §6).
        paragraph_overlap_chars: Overlap in characters between split segments.

    Returns:
        List of Paragraph nodes with full provenance.
    """
    if metadata is None:
        metadata = {}

    all_paragraphs: List[Paragraph] = []

    for fpath in file_paths:
        try:
            text = _read_file(fpath)
        except (OSError, UnicodeDecodeError):
            continue

        if not text or not text.strip():
            continue

        # MD files: use ## (h2) headings as subtopic boundaries
        subtopic_blocks = _extract_subtopic_blocks(text, heading_level=2)

        if subtopic_blocks:
            paragraphs = _subtopics_to_paragraphs(
                subtopic_blocks, fpath, metadata, paragraph_max_chars, paragraph_overlap_chars
            )
        else:
            # No ## headings — create one implicit subtopic from the whole file
            blocks = _extract_blocks(text)
            implicit_st = [_SubtopicBlock(heading=None, sub_blocks=blocks)]
            paragraphs = _subtopics_to_paragraphs(
                implicit_st, fpath, metadata, paragraph_max_chars, paragraph_overlap_chars
            )

        # Link adjacent paragraphs within this source file only
        _link_adjacent_paragraphs(paragraphs)
        all_paragraphs.extend(paragraphs)

    return all_paragraphs


# ---------------------------------------------------------------------------
# File reading
# ---------------------------------------------------------------------------

def _read_file(path: str) -> str:
    """Read a file, trying UTF-8 first then latin-1 as fallback."""
    for encoding in ("utf-8", "latin-1"):
        try:
            with open(path, "r", encoding=encoding) as f:
                return f.read()
        except (OSError, UnicodeDecodeError):
            continue
    return ""


# ---------------------------------------------------------------------------
# Block extraction
# ---------------------------------------------------------------------------

def _extract_blocks(text: str) -> List[_Block]:
    """Extract text blocks delimited by markdown headings.

    Handles:
    - Nested headings (deeper headings start new blocks)
    - Code blocks (preserved as-is, not split)
    - Empty sections
    """
    lines = text.split("\n")
    blocks: List[_Block] = []
    current_lines: List[str] = []
    current_heading: Optional[_Heading] = None
    current_start = 0
    in_code_fence = False
    fence_char: str = ""

    for i, line in enumerate(lines):
        # Track code fences
        fence_match = _CODE_FENCE_RE.match(line.strip())
        if fence_match:
            fence = fence_match.group(1)
            if not in_code_fence:
                in_code_fence = True
                fence_char = fence[0]
                current_lines.append(line)
                continue
            elif fence.startswith(fence_char) and len(fence) >= len(fence_char):
                in_code_fence = False
                current_lines.append(line)
                continue

        if in_code_fence:
            current_lines.append(line)
            continue

        heading_match = _HEADING_RE.match(line)
        if heading_match:
            # Save previous block
            blocks.append(_Block(current_heading, current_lines, current_start))

            level = len(heading_match.group(1))
            title = heading_match.group(2).strip()
            current_heading = _Heading(level, title, i)
            current_lines = []
            current_start = i
        else:
            current_lines.append(line)

    # Final block
    blocks.append(_Block(current_heading, current_lines, current_start))

    return blocks


def _extract_subtopic_blocks(text: str, heading_level: int = 3) -> List[_SubtopicBlock]:
    """Extract subtopic blocks from markdown text.

    Subtopics are delimited by headings of the specified level.
    Within each subtopic, deeper headings create sub-blocks.

    Args:
        text: Markdown text to parse.
        heading_level: Heading level to treat as subtopic boundary (2 for ##, 3 for ###).

    Returns:
        List of subtopic blocks. Empty list if no headings of the specified level exist.
    """
    lines = text.split("\n")
    subtopics: List[_SubtopicBlock] = []
    current_subtopic: Optional[_SubtopicBlock] = None
    current_lines: List[str] = []
    current_heading: Optional[_Heading] = None
    current_start = 0
    in_code_fence = False
    fence_char = ""

    def _save_current_sub_block() -> None:
        """Save the current accumulating lines as a sub-block in current_subtopic."""
        nonlocal current_lines, current_heading, current_start
        if current_subtopic is not None:
            stripped = "\n".join(current_lines).strip()
            if current_heading is not None or stripped:
                current_subtopic.sub_blocks.append(
                    _Block(current_heading, current_lines, current_start)
                )
            current_lines = []
            current_heading = None
            current_start = 0

    def _add_current_subtopic() -> None:
        """Finalize current subtopic and add it to the list."""
        nonlocal current_subtopic
        _save_current_sub_block()
        if current_subtopic is not None:
            subtopics.append(current_subtopic)
            current_subtopic = None

    for i, line in enumerate(lines):
        # Track code fences
        fence_match = _CODE_FENCE_RE.match(line.strip())
        if fence_match:
            fence = fence_match.group(1)
            if not in_code_fence:
                in_code_fence = True
                fence_char = fence[0]
                if current_subtopic is not None:
                    current_lines.append(line)
                continue
            elif fence.startswith(fence_char) and len(fence) >= len(fence_char):
                in_code_fence = False
                if current_subtopic is not None:
                    current_lines.append(line)
                continue

        if in_code_fence:
            if current_subtopic is not None:
                current_lines.append(line)
            continue

        heading_match = _HEADING_RE.match(line)
        if heading_match:
            level = len(heading_match.group(1))
            title = heading_match.group(2).strip()

            if level == heading_level:
                # New heading at target level — finalize previous subtopic, start new one
                _add_current_subtopic()
                current_subtopic = _SubtopicBlock(
                    heading=_Heading(level, title, i),
                    sub_blocks=[],
                )
                current_lines = [line]
                current_heading = _Heading(level, title, i)
                current_start = i
            elif current_subtopic is not None and level > heading_level:
                # Deeper heading within a subtopic — save current, start sub-block
                _save_current_sub_block()
                current_lines = [line]
                current_heading = _Heading(level, title, i)
                current_start = i
            else:
                # Higher level heading (e.g., ## when looking for ###), or no current subtopic
                pass
        else:
            if current_subtopic is not None:
                current_lines.append(line)

    # Finalize last subtopic
    _add_current_subtopic()

    return subtopics


# ---------------------------------------------------------------------------
# Block → Paragraph conversion
# ---------------------------------------------------------------------------

def _subtopics_to_paragraphs(
    subtopic_blocks: List[_SubtopicBlock],
    source_file: str,
    metadata: Dict[str, str],
    paragraph_max_chars: int,
    paragraph_overlap_chars: int,
) -> List[Paragraph]:
    """Convert parsed subtopic blocks into Paragraph nodes.

    Each subtopic gets a subtopic_id. Sub-blocks within each subtopic become
    paragraphs with that subtopic_id attached.
    """
    paragraphs: List[Paragraph] = []
    source_url = metadata.get("source_url")
    corpus_id = metadata.get("corpus_id")
    license_val = metadata.get("license")

    for st_idx, st_block in enumerate(subtopic_blocks):
        if st_block.heading is not None:
            subtopic_title = st_block.heading.title
        else:
            subtopic_title = "Whole File"
        subtopic_id = f"{source_file}:subtopic-{st_idx}"

        for sb in st_block.sub_blocks:
            text = "\n".join(sb.lines).strip()

            # Skip empty sub-blocks
            if sb.heading is None and not text:
                continue

            if sb.heading is None:
                para = _make_paragraph(
                    title="Untitled",
                    content=text,
                    source_file=source_file,
                    source_url=source_url,
                    corpus_id=corpus_id,
                    license=license_val,
                    page_number=None,
                    paragraph_max_chars=paragraph_max_chars,
                    paragraph_overlap_chars=paragraph_overlap_chars,
                    subtopic_id=subtopic_id,
                )
                paragraphs.extend(para)
            else:
                para = _make_paragraph(
                    title=sb.heading.title,
                    content=text,
                    source_file=source_file,
                    source_url=source_url,
                    corpus_id=corpus_id,
                    license=license_val,
                    page_number=None,
                    paragraph_max_chars=paragraph_max_chars,
                    paragraph_overlap_chars=paragraph_overlap_chars,
                    subtopic_id=subtopic_id,
                )
                paragraphs.extend(para)

    return paragraphs


def _make_paragraph(
    title: str,
    content: str,
    source_file: str,
    source_url: Optional[str],
    corpus_id: Optional[str],
    license: Optional[str],
    page_number: Optional[int],
    paragraph_max_chars: int,
    paragraph_overlap_chars: int,
    subtopic_id: Optional[str] = None,
) -> List[Paragraph]:
    """Create one or more Paragraph nodes from title+content.

    If content exceeds paragraph_max_chars, split into overlapping segments.
    """
    if len(content) <= paragraph_max_chars:
        return [_create_single(title, content, source_file, source_url,
                                corpus_id, license, page_number, subtopic_id)]

    segments = _split_text(content, paragraph_max_chars, paragraph_overlap_chars)
    result = []
    for i, seg in enumerate(segments):
        seg_title = title if len(segments) == 1 else f"{title} ({i + 1}/{len(segments)})"
        result.append(_create_single(seg_title, seg, source_file, source_url,
                                      corpus_id, license, page_number, subtopic_id))
    return result


def _create_single(
    title: str,
    content: str,
    source_file: str,
    source_url: Optional[str],
    corpus_id: Optional[str],
    license: Optional[str],
    page_number: Optional[int],
    subtopic_id: Optional[str] = None,
) -> Paragraph:
    """Create a single Paragraph node with a deterministic id."""
    raw_id = f"{source_file}:{hashlib.md5(content.encode('utf-8')[:256]).hexdigest()[:12]}"
    return Paragraph(
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
        subtopic_id=subtopic_id,
        paragraph_id=raw_id,
        prev_paragraph_id=None,
        next_paragraph_id=None,
    )


def _split_text(text: str, max_chars: int, overlap_chars: int) -> List[str]:
    """Split text into overlapping segments of at most max_chars.

    Splits on word boundaries where possible to avoid cutting words in half.
    The overlap ensures continuity between segments.
    """
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

        # Try to split at a word boundary near the end
        split_pos = _find_word_boundary(text, end, max_chars)
        segment = text[start:split_pos].strip()

        if segment:
            segments.append(segment)

        # Move start forward: max_chars - overlap_chars to get overlap
        start = split_pos - overlap_chars if overlap_chars > 0 else end
        if start <= (split_pos - overlap_chars if overlap_chars > 0 else end - overlap_chars):
            # Ensure forward progress
            pass
        start = max(start, end - overlap_chars)

    return segments


def _find_word_boundary(text: str, pos: int, max_window: int = 50) -> int:
    """Find the nearest word boundary before pos, within max_window chars."""
    # Look backward for a space
    search_start = max(0, pos - max_window)
    last_space = text.rfind(" ", search_start, pos)

    if last_space > search_start:
        return last_space

    # Fall back to the exact position
    return pos


def _link_adjacent_paragraphs(paragraphs: List[Paragraph]) -> None:
    """Set prev_paragraph_id and next_paragraph_id for adjacent paragraphs."""
    for i, para in enumerate(paragraphs):
        if i > 0:
            para.prev_paragraph_id = paragraphs[i - 1].id
        if i < len(paragraphs) - 1:
            para.next_paragraph_id = paragraphs[i + 1].id
