"""Parse .txt files from docs/geeksforgeeks/ into Paragraph nodes.

Each file starts with a multi-line header:
    Source: <URL>
    Title: <title>
    Extractor: <version>

Rest is plain text with headings (##, ###, ####) and paragraphs.
Same oversized paragraph splitting logic as md_parser.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional

from dsa_mentor.models import Paragraph
from dsa_mentor.ingestion.md_parser import (
    _read_file,
    _extract_blocks,
    _blocks_to_paragraphs,
)


# ---------------------------------------------------------------------------
# Header parsing
# ---------------------------------------------------------------------------

_SOURCE_RE = re.compile(r"^Source:\s*(.+)$", re.IGNORECASE)
_TITLE_RE = re.compile(r"^Title:\s*(.+)$", re.IGNORECASE)
_EXTRACTOR_RE = re.compile(r"^Extractor:\s*(.+)$", re.IGNORECASE)


def _parse_header(lines: List[str]) -> Dict[str, str]:
    """Parse the GfG multi-line header into a metadata dict.

    Header format:
        Source: <URL>
        Title: <title>
        Extractor: <version>
    Returns (metadata_dict, content_start_line_index).
    """
    meta: Dict[str, str] = {"source_url": "", "title": "", "extractor": ""}
    content_start = 0

    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue

        source_match = _SOURCE_RE.match(stripped)
        title_match = _TITLE_RE.match(stripped)
        extractor_match = _EXTRACTOR_RE.match(stripped)

        if source_match:
            meta["source_url"] = source_match.group(1).strip()
            content_start = i + 1
        elif title_match:
            meta["title"] = title_match.group(1).strip()
            content_start = i + 1
        elif extractor_match:
            meta["extractor"] = extractor_match.group(1).strip()
            content_start = i + 1
        else:
            # Non-header line — content starts here
            content_start = i
            break

    return meta, content_start


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
    """Parse one or more .txt files (GfG format) into Paragraph nodes.

    Args:
        file_paths: Paths to .txt files.
        metadata:   Additional metadata dict (corpus_id, license, etc.).
                    source_url and title from the file header override these.
        paragraph_max_chars: Max characters per paragraph before splitting.
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

        lines = text.split("\n")

        # Parse multi-line header
        header_meta, content_start = _parse_header(lines)

        # Merge header metadata with provided metadata
        file_meta = dict(metadata)
        if header_meta.get("source_url"):
            file_meta.setdefault("source_url", header_meta["source_url"])
        if header_meta.get("title"):
            file_meta.setdefault("title", header_meta["title"])

        # Remaining text (skip the header lines)
        body_text = "\n".join(lines[content_start:]).strip()

        if not body_text:
            continue

        blocks = _extract_blocks(body_text)
        paragraphs = _blocks_to_paragraphs(
            blocks, fpath, file_meta, paragraph_max_chars, paragraph_overlap_chars
        )
        all_paragraphs.extend(paragraphs)

    return all_paragraphs
