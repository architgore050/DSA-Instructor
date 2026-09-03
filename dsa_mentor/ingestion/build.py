"""Orchestrate document ingestion: scan docs/, parse files, build hierarchy,
produce KnowledgeManifest, and save knowledge base as JSON.

Usage:
    python -m dsa_mentor.ingestion.build [--config config.json] [--output knowledge_base.json]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dsa_mentor.config import Config
from dsa_mentor.models import (
    Book,
    Chapter,
    KnowledgeManifest,
    Paragraph,
    RetrievalResult,
    Subtopic,
    Topic,
)

# Import parsers
from dsa_mentor.ingestion.md_parser import parse_paragraphs as parse_md_paragraphs
from dsa_mentor.ingestion.txt_parser import parse_paragraphs as parse_txt_paragraphs
from dsa_mentor.ingestion.pdf_parser import parse_paragraphs as parse_pdf_paragraphs


# ---------------------------------------------------------------------------
# File scanning
# ---------------------------------------------------------------------------

def scan_docs(docs_dir: str) -> Dict[str, List[str]]:
    """Scan docs/ directory for source files, classified by extension.

    Returns:
        Dict mapping extension (".md", ".txt", ".pdf") to list of file paths.
    """
    classification: Dict[str, List[str]] = {
        ".md": [],
        ".txt": [],
        ".pdf": [],
    }

    docs_path = Path(docs_dir)
    if not docs_path.exists():
        return classification

    for root, _dirs, files in os.walk(docs_path):
        for fname in files:
            ext = os.path.splitext(fname)[1].lower()
            if ext in classification:
                fpath = os.path.join(root, fname)
                classification[ext].append(fpath)

    # Sort for deterministic ordering
    for ext in classification:
        classification[ext].sort()

    return classification


# ---------------------------------------------------------------------------
# Hierarchical ID assignment
# ---------------------------------------------------------------------------

def _assign_hierarchical_ids(
    files_by_type: Dict[str, List[str]],
    metadata_map: Dict[str, Dict[str, str]],
) -> Dict[str, Tuple[str, str, str, str]]:
    """Assign corpus_id, book_id, chapter_id, topic_id to each source file.

    Returns:
        Dict mapping source_file -> (corpus_id, book_id, chapter_id, topic_id).
    """
    result: Dict[str, Tuple[str, str, str, str]] = {}
    file_list: List[str] = []

    for ext in (".md", ".txt", ".pdf"):
        file_list.extend(files_by_type.get(ext, []))

    # Group files into books based on directory structure
    # Each top-level directory in docs/ becomes a "book"
    book_groups: Dict[str, List[str]] = {}

    for fpath in file_list:
        rel = os.path.relpath(fpath, str(Path(fpath).parent.parent.parent))
        # Top-level directory name (after docs/)
        parts = rel.split(os.sep)
        if len(parts) >= 1:
            book_name = parts[0]
        else:
            book_name = "uncategorized"
        book_groups.setdefault(book_name, []).append(fpath)

    # Assign book IDs
    book_counter = 0
    book_id_map: Dict[str, str] = {}

    for book_name in sorted(book_groups.keys()):
        book_counter += 1
        book_id = f"book-{book_counter:03d}"
        book_id_map[book_name] = book_id

        # Assign chapter/topic IDs within each book
        chapter_counter = 0
        chapter_id_map: Dict[str, str] = {}

        for fpath in sorted(book_groups[book_name]):
            # Derive chapter from subdirectory or file name
            rel = os.path.relpath(fpath, str(Path(fpath).parent.parent.parent))
            parts = rel.split(os.sep)

            if len(parts) >= 2:
                chapter_name = parts[1]
            else:
                chapter_name = os.path.splitext(os.path.basename(fpath))[0]

            if chapter_name not in chapter_id_map:
                chapter_counter += 1
                chapter_id_map[chapter_name] = f"ch-{book_counter:03d}-{chapter_counter:03d}"

            chapter_id = chapter_id_map[chapter_name]

            # Derive topic from file name
            topic_name = os.path.splitext(os.path.basename(fpath))[0]
            topic_id = f"topic-{book_counter:03d}-{chapter_counter:03d}-{len(chapter_id_map):03d}"

            corpus_id = metadata_map.get(fpath, {}).get("corpus_id", book_name)

            result[fpath] = (corpus_id, book_id, chapter_id, topic_id)

    return result


# ---------------------------------------------------------------------------
# Metadata loading
# ---------------------------------------------------------------------------

def _build_metadata_map(
    files_by_type: Dict[str, List[str]],
) -> Dict[str, Dict[str, str]]:
    """Build metadata dict for each file based on its path and type.

    Returns:
        Dict mapping source_file -> metadata dict.
    """
    metadata_map: Dict[str, Dict[str, str]] = {}

    for fpath in files_by_type.get(".md", []):
        rel = os.path.relpath(fpath, "docs")
        parts = rel.split(os.sep)
        corpus_id = parts[0] if parts else "uncategorized"
        metadata_map[fpath] = {
            "source_type": "markdown",
            "corpus_id": corpus_id,
            "license": None,
            "source_url": None,
        }

    for fpath in files_by_type.get(".txt", []):
        rel = os.path.relpath(fpath, "docs")
        parts = rel.split(os.sep)
        corpus_id = parts[0] if parts else "geeksforgeeks"
        metadata_map[fpath] = {
            "source_type": "text",
            "corpus_id": corpus_id,
            "license": None,
            "source_url": None,
        }

    for fpath in files_by_type.get(".pdf", []):
        basename = os.path.basename(fpath)
        corpus_id = os.path.splitext(basename)[0]
        metadata_map[fpath] = {
            "source_type": "pdf",
            "corpus_id": corpus_id,
            "license": None,
            "source_url": None,
        }

    return metadata_map


# ---------------------------------------------------------------------------
# Parsing dispatch
# ---------------------------------------------------------------------------

def _parse_file(
    fpath: str,
    ext: str,
    metadata: Dict[str, str],
    paragraph_max_chars: int,
    paragraph_overlap_chars: int,
    hierarchy_ids: Dict[str, Tuple[str, str, str, str]],
) -> List[Paragraph]:
    """Parse a single file using the appropriate parser."""
    corpus_id, book_id, chapter_id, topic_id = hierarchy_ids.get(
        fpath, (None, None, None, None)
    )

    # Update metadata with hierarchy IDs
    meta = dict(metadata)
    meta["corpus_id"] = corpus_id
    meta["book_id"] = book_id
    meta["chapter_id"] = chapter_id
    meta["topic_id"] = topic_id

    if ext == ".md":
        paragraphs = parse_md_paragraphs(
            [fpath], meta,
            paragraph_max_chars=paragraph_max_chars,
            paragraph_overlap_chars=paragraph_overlap_chars,
        )
    elif ext == ".txt":
        paragraphs = parse_txt_paragraphs(
            [fpath], meta,
            paragraph_max_chars=paragraph_max_chars,
            paragraph_overlap_chars=paragraph_overlap_chars,
        )
    elif ext == ".pdf":
        paragraphs = parse_pdf_paragraphs(
            [fpath], meta,
            paragraph_max_chars=paragraph_max_chars,
            paragraph_overlap_chars=paragraph_overlap_chars,
        )
    else:
        return []

    # Attach hierarchy IDs to each paragraph
    for para in paragraphs:
        para.corpus_id = corpus_id
        para.book_id = book_id
        para.chapter_id = chapter_id
        para.topic_id = topic_id

    return paragraphs


# ---------------------------------------------------------------------------
# Hierarchy tree building
# ---------------------------------------------------------------------------

def _build_hierarchy(
    paragraphs: List[Paragraph],
    hierarchy_ids: Dict[str, Tuple[str, str, str, str]],
) -> Tuple[List[Book], List[Chapter], List[Topic], List[Subtopic]]:
    """Build the full hierarchy tree from parsed paragraphs.

    Returns:
        (books, chapters, topics, subtopics) lists.
    """
    books: Dict[str, Book] = {}
    chapters: Dict[str, Chapter] = {}
    topics: Dict[str, Topic] = {}
    subtopics: Dict[str, Subtopic] = {}

    for para in paragraphs:
        # Book
        if para.book_id and para.book_id not in books:
            books[para.book_id] = Book(
                id=para.book_id,
                title=para.corpus_id or para.book_id,
                level="book",
                parent_id=None,
                children=[],
                corpus_id=para.corpus_id,
                license=para.license,
            )

        # Chapter
        if para.chapter_id and para.chapter_id not in chapters:
            chapters[para.chapter_id] = Chapter(
                id=para.chapter_id,
                title=para.chapter_id.replace("-", " ").title(),
                level="chapter",
                parent_id=para.book_id,
                children=[],
                book_id=para.book_id,
                chapter_id=para.chapter_id,
            )

        # Topic
        if para.topic_id and para.topic_id not in topics:
            topics[para.topic_id] = Topic(
                id=para.topic_id,
                title=para.topic_id.replace("-", " ").title(),
                level="topic",
                parent_id=para.chapter_id,
                children=[],
                chapter_id=para.chapter_id,
                book_id=para.book_id,
                full_text=None,
            )

    # Link hierarchy
    for para in paragraphs:
        if para.book_id and para.book_id in books:
            if para.chapter_id and para.chapter_id not in books[para.book_id].children:
                books[para.book_id].children.append(para.chapter_id)
        if para.chapter_id and para.chapter_id in chapters:
            if para.book_id and para.book_id not in chapters[para.chapter_id].children:
                chapters[para.chapter_id].children.append(para.book_id)
            if para.topic_id and para.topic_id not in chapters[para.chapter_id].children:
                chapters[para.chapter_id].children.append(para.topic_id)
        if para.topic_id and para.topic_id in topics:
            if para.chapter_id and para.chapter_id not in topics[para.topic_id].children:
                topics[para.topic_id].children.append(para.chapter_id)

    return (
        list(books.values()),
        list(chapters.values()),
        list(topics.values()),
        list(subtopics.values()),
    )


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

def _compute_manifest(
    files_by_type: Dict[str, List[str]],
    all_paragraphs: List[Paragraph],
) -> KnowledgeManifest:
    """Compute KnowledgeManifest with per-file sha256 hashes."""
    file_hashes: Dict[str, str] = {}

    for ext in (".md", ".txt", ".pdf"):
        for fpath in files_by_type.get(ext, []):
            try:
                h = hashlib.sha256()
                with open(fpath, "rb") as f:
                    for chunk in iter(lambda: f.read(8192), b""):
                        h.update(chunk)
                file_hashes[fpath] = h.hexdigest()
            except OSError:
                file_hashes[fpath] = ""

    total_nodes = len(all_paragraphs)  # paragraphs are the main nodes

    return KnowledgeManifest.create(
        file_hashes=file_hashes,
        total_nodes=total_nodes,
        total_paragraphs=len(all_paragraphs),
    )


# ---------------------------------------------------------------------------
# Save knowledge base
# ---------------------------------------------------------------------------

def _save_knowledge_base(
    output_path: str,
    books: List[Book],
    chapters: List[Chapter],
    topics: List[Topic],
    subtopics: List[Subtopic],
    paragraphs: List[Paragraph],
    manifest: KnowledgeManifest,
) -> None:
    """Save the full knowledge base as JSON."""
    data = {
        "manifest": manifest.to_dict(),
        "books": [b.to_dict() for b in books],
        "chapters": [c.to_dict() for c in chapters],
        "topics": [t.to_dict() for t in topics],
        "subtopics": [s.to_dict() for s in subtopics],
        "paragraphs": [p.to_dict() for p in paragraphs],
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def build(
    config_path: str = "config.json",
    output_path: Optional[str] = None,
    docs_dir: str = "docs",
) -> None:
    """Run the full ingestion pipeline.

    Args:
        config_path: Path to config.json.
        output_path: Output path for knowledge_base.json (default: same dir as config).
        docs_dir:    Directory containing source documents.
    """
    config = Config.get(config_path)

    if output_path is None:
        config_dir = os.path.dirname(os.path.abspath(config_path))
        output_path = os.path.join(config_dir, "knowledge_base.json")

    paragraph_max_chars = config.retrieval.paragraph_max_chars
    paragraph_overlap_chars = config.retrieval.paragraph_overlap_chars

    # Step 1: Scan docs/
    print(f"Scanning {docs_dir}/ for source files...")
    files_by_type = scan_docs(docs_dir)

    total_files = sum(len(v) for v in files_by_type.values())
    for ext, paths in files_by_type.items():
        if paths:
            print(f"  {ext}: {len(paths)} files")

    if total_files == 0:
        print("No source files found. Exiting.")
        return

    # Step 2: Build metadata map
    metadata_map = _build_metadata_map(files_by_type)

    # Step 3: Assign hierarchical IDs
    hierarchy_ids = _assign_hierarchical_ids(files_by_type, metadata_map)

    # Step 4: Parse all files
    print("Parsing source files...")
    all_paragraphs: List[Paragraph] = []
    files_parsed = 0

    for ext in (".md", ".txt", ".pdf"):
        for fpath in files_by_type.get(ext, []):
            meta = metadata_map.get(fpath, {})
            paras = _parse_file(
                fpath, ext, meta, paragraph_max_chars, paragraph_overlap_chars,
                hierarchy_ids,
            )
            all_paragraphs.extend(paras)
            files_parsed += 1

    print(f"  Files parsed: {files_parsed}")
    print(f"  Paragraphs extracted: {len(all_paragraphs)}")

    # Step 5: Build hierarchy
    print("Building hierarchy tree...")
    books, chapters, topics, subtopics = _build_hierarchy(all_paragraphs, hierarchy_ids)
    print(f"  Books: {len(books)}")
    print(f"  Chapters: {len(chapters)}")
    print(f"  Topics: {len(topics)}")
    print(f"  Subtopics: {len(subtopics)}")

    # Step 6: Compute manifest
    print("Computing manifest...")
    manifest = _compute_manifest(files_by_type, all_paragraphs)
    print(f"  Files hashed: {len(manifest.file_hashes)}")

    # Step 7: Save knowledge base
    print(f"Saving knowledge base to {output_path}...")
    _save_knowledge_base(
        output_path, books, chapters, topics, subtopics, all_paragraphs, manifest
    )
    print("Done.")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="DSA Mentor — document ingestion")
    parser.add_argument(
        "--config", default="config.json",
        help="Path to config.json (default: config.json)",
    )
    parser.add_argument(
        "--output", default=None,
        help="Output path for knowledge_base.json (default: same dir as config)",
    )
    parser.add_argument(
        "--docs-dir", default="docs",
        help="Directory containing source documents (default: docs)",
    )
    args = parser.parse_args()

    build(
        config_path=args.config,
        output_path=args.output,
        docs_dir=args.docs_dir,
    )


if __name__ == "__main__":
    main()
