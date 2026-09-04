"""Orchestrate document ingestion: scan docs/, parse files, build hierarchy,
produce KnowledgeManifest, and save knowledge base as JSON.

Usage:
    python -m dsa_mentor.ingestion.build [--config config.json] [--output knowledge_base.json]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dsa_mentor.config import Config, load_config
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

try:
    from tqdm import tqdm
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False

    class _NoOpTqdm:
        def __init__(self, *args, **kwargs):
            self._total = kwargs.get("total", 0)
        def __iter__(self):
            return iter(self._items)
        def update(self, n=1):
            pass
        def set_description(self, desc=""):
            pass
        def set_postfix(self, **kwargs):
            pass
        @staticmethod
        def write(msg):
            print(msg, flush=True)
    tqdm = _NoOpTqdm  # type: ignore

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

_logger = logging.getLogger("dsa_mentor.ingestion")


def _setup_logging(verbose: bool = False) -> None:
    """Configure logging for the ingestion pipeline."""
    level = logging.DEBUG if verbose else logging.INFO
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    handler.setFormatter(formatter)
    _logger.setLevel(level)
    if not _logger.handlers:
        _logger.addHandler(handler)


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
        _logger.warning("docs_dir '%s' does not exist", docs_dir)
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

    _logger.info("scan_docs: .md=%d, .txt=%d, .pdf=%d",
                 len(classification[".md"]),
                 len(classification[".txt"]),
                 len(classification[".pdf"]))
    return classification


def _detect_book_root(fpath: str, docs_dir: str) -> str:
    """Detect the book root directory for a file.

    The book root is the first directory under docs/.
    E.g., docs/cp-algorithms/src/graph/dijkstra.md -> "cp-algorithms"
         docs/geeksforgeeks/0-1-bfs.txt -> "geeksforgeeks"
         docs/javascript-algorithms/algorithms/sorting/quick-sort.md -> "javascript-algorithms"
         docs/thealgorithms/python/sorts/bubble_sort.md -> "thealgorithms"
    """
    rel = os.path.relpath(fpath, docs_dir)
    parts = rel.split(os.sep)
    if parts and parts[0]:
        return parts[0]
    return "uncategorized"


def _detect_chapter(fpath: str, docs_dir: str, book_name: str) -> str:
    """Detect the chapter for a file within its book.

    The chapter is the second directory level under docs/<book_name>/.
    E.g., docs/cp-algorithms/src/graph/dijkstra.md -> "src/graph"
         docs/geeksforgeeks/0-1-bfs.txt -> "root" (flat directory)
         docs/javascript-algorithms/algorithms/sorting/quick-sort.md -> "algorithms/sorting"
    """
    rel = os.path.relpath(fpath, docs_dir)
    parts = rel.split(os.sep)
    # Skip book_name (parts[0]), chapter is parts[1] onwards
    if len(parts) >= 2:
        chapter_parts = parts[1:-1]  # everything between book and filename
        if chapter_parts:
            return os.path.join(*chapter_parts)
    return "root"


def _assign_hierarchical_ids(
    files_by_type: Dict[str, List[str]],
    metadata_map: Dict[str, Dict[str, str]],
    docs_dir: str = "docs",
) -> Dict[str, Tuple[str, str, str, str]]:
    """Assign corpus_id, book_id, chapter_id, topic_id to each source file.

    Books are grouped by their first directory level under docs/:
        - Each source directory (e.g. cp-algorithms, geeksforgeeks, thealgorithms)
          becomes its own Book node.
        - PDFs are grouped under a "textbooks" book.

    Returns:
        Dict mapping source_file -> (corpus_id, book_id, chapter_id, topic_id).
    """
    result: Dict[str, Tuple[str, str, str, str]] = {}
    file_list: List[str] = []

    for ext in (".md", ".txt", ".pdf"):
        file_list.extend(files_by_type.get(ext, []))

    _logger.info("assign_hierarchical_ids: %d files to process", len(file_list))

    # Step 1: Group files by book root (first directory under docs/)
    # PDFs → "textbooks", everything else → its source directory name
    pdf_files: List[str] = files_by_type.get(".pdf", [])
    book_groups: Dict[str, List[str]] = {"textbooks": sorted(pdf_files)}

    for ext in (".md", ".txt"):
        for fpath in files_by_type.get(ext, []):
            book_root = _detect_book_root(fpath, docs_dir)
            book_groups.setdefault(book_root, []).append(fpath)

    # Sort files within each group for deterministic IDs
    for bname in book_groups:
        book_groups[bname] = sorted(book_groups[bname])

    _logger.info("assign_hierarchical_ids: %d canonical books", len(book_groups))
    for bname, bfiles in book_groups.items():
        _logger.info("  book '%s': %d files", bname, len(bfiles))

    # Step 2: Assign book IDs sequentially
    book_names = sorted(book_groups.keys())
    book_id_map: Dict[str, str] = {}
    for idx, bname in enumerate(book_names, start=1):
        book_id_map[bname] = f"book-{idx:03d}"

    # Step 3: Process each book
    for book_name in book_names:
        book_files = book_groups[book_name]
        book_id = book_id_map[book_name]
        corpus_id = book_name

        _logger.info("assign_hierarchical_ids: assigning IDs for book '%s' (book_id=%s, corpus_id=%s)",
                     book_name, book_id, corpus_id)

        if book_name == "textbooks":
            # Each PDF is its own chapter (one chapter per PDF file)
            chapter_counter = 0
            for fpath in book_files:
                chapter_counter += 1
                chapter_id = f"ch-{int(book_id.replace('book-', '')):03d}-{chapter_counter:03d}"
                topic_id = f"topic-{int(book_id.replace('book-', '')):03d}-{chapter_counter:03d}-001"
                result[fpath] = (corpus_id, book_id, chapter_id, topic_id)
                _logger.info("    PDF '%s' -> chapter=%s, topic=%s",
                             os.path.basename(fpath), chapter_id, topic_id)

        else:
            # Web corpus: group by chapter (second directory level under docs/)
            chapter_groups: Dict[str, List[str]] = {}
            for fpath in book_files:
                chapter_name = _detect_chapter(fpath, docs_dir, book_name)
                chapter_groups.setdefault(chapter_name, []).append(fpath)

            _logger.info("  book '%s': %d chapters", book_name, len(chapter_groups))
            for cname in sorted(chapter_groups.keys()):
                _logger.info("    chapter '%s': %d files", cname, len(chapter_groups[cname]))

            # Assign chapter IDs scoped to this book
            book_prefix = book_id.replace("book-", "")
            chapter_counter = 0
            chapter_id_map_local: Dict[str, str] = {}
            for chapter_name in sorted(chapter_groups.keys()):
                chapter_counter += 1
                chapter_id = f"ch-{book_prefix}-{chapter_counter:03d}"
                chapter_id_map_local[chapter_name] = chapter_id

            # Assign topic IDs (PER-FILE, scoped per chapter)
            for chapter_name in sorted(chapter_groups.keys()):
                chapter_id = chapter_id_map_local[chapter_name]
                file_counter = 0
                for fpath in sorted(chapter_groups[chapter_name]):
                    file_counter += 1
                    topic_id = f"topic-{book_prefix}-{chapter_counter:03d}-{file_counter:03d}"
                    result[fpath] = (corpus_id, book_id, chapter_id, topic_id)

            _logger.info("  book '%s': %d chapters, %d topics assigned",
                         book_name, chapter_counter, len(book_files))

    _logger.info("assign_hierarchical_ids: total %d files assigned IDs", len(result))
    return result


# ---------------------------------------------------------------------------
# Metadata loading
# ---------------------------------------------------------------------------

def _build_metadata_map(
    files_by_type: Dict[str, List[str]],
    docs_dir: str = "docs",
) -> Dict[str, Dict[str, str]]:
    """Build metadata dict for each file based on its path and type.

    Assigns corpus_id based on first directory under docs/:
        - Each source directory gets its own corpus_id (e.g. cp-algorithms, geeksforgeeks)
        - PDFs get corpus_id from filename

    Returns:
        Dict mapping source_file -> metadata dict.
    """
    metadata_map: Dict[str, Dict[str, str]] = {}

    for fpath in files_by_type.get(".md", []):
        corpus_id = _detect_book_root(fpath, docs_dir)
        metadata_map[fpath] = {
            "source_type": "markdown",
            "corpus_id": corpus_id,
            "license": None,
            "source_url": None,
        }

    for fpath in files_by_type.get(".txt", []):
        corpus_id = _detect_book_root(fpath, docs_dir)
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

    _logger.info("metadata_map: %d entries built", len(metadata_map))
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
    """Parse a single file using the appropriate parser.

    Attaches hierarchy IDs and hierarchical_chunk_id to each paragraph.
    hierarchical_chunk_id = subtopic_id (all paragraphs in same subtopic share it).
    """
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

    # Attach hierarchy IDs and hierarchical_chunk_id to each paragraph
    for para in paragraphs:
        para.corpus_id = corpus_id
        para.book_id = book_id
        para.chapter_id = chapter_id
        para.topic_id = topic_id
        # hierarchical_chunk_id = subtopic_id for deduplication
        # All paragraphs within the same subtopic share this ID
        para.hierarchical_chunk_id = para.subtopic_id

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

    # Track subtopic titles from paragraph data for better naming
    subtopic_title_map: Dict[str, str] = {}

    # Readable titles for corpus sources
    _BOOK_TITLES: Dict[str, str] = {
        "textbooks": "Textbooks",
        "cp-algorithms": "CP-Algorithms",
        "geeksforgeeks": "GeeksforGeeks DSA Tutorial",
        "javascript-algorithms": "JavaScript Algorithms",
        "thealgorithms": "The Algorithms",
    }

    for para in paragraphs:
        # Book
        if para.book_id and para.book_id not in books:
            corpus = para.corpus_id or ""
            books[para.book_id] = Book(
                id=para.book_id,
                title=_BOOK_TITLES.get(corpus, corpus.replace("-", " ").title() if corpus else para.book_id),
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

        # Subtopic
        if para.subtopic_id and para.subtopic_id not in subtopics:
            # Try to derive a readable title from the subtopic_id
            # Format: "docs/.../file.md:subtopic-0" or "docs/.../file.txt:subtopic-0"
            st_parts = para.subtopic_id.split(":subtopic-")
            if len(st_parts) >= 2:
                file_part = st_parts[0]
                st_num = st_parts[1]
                # Extract file name without extension
                file_basename = os.path.basename(file_part)
                file_name = os.path.splitext(file_basename)[0]
                # Use file name as subtopic title with index
                st_title = f"{file_name} (subtopic {st_num})"
            else:
                st_title = f"Subtopic {para.subtopic_id}"

            subtopic_title_map[para.subtopic_id] = st_title
            subtopics[para.subtopic_id] = Subtopic(
                id=para.subtopic_id,
                title=st_title,
                level="subtopic",
                parent_id=para.topic_id,
                children=[],
                topic_id=para.topic_id,
                chapter_id=para.chapter_id,
                book_id=para.book_id,
            )

    _logger.info("hierarchy counts: books=%d, chapters=%d, topics=%d, subtopics=%d",
                 len(books), len(chapters), len(topics), len(subtopics))

    # Link hierarchy (CORRECT parent->child direction)
    for para in paragraphs:
        if para.book_id and para.book_id in books:
            if para.chapter_id and para.chapter_id not in books[para.book_id].children:
                books[para.book_id].children.append(para.chapter_id)
        if para.chapter_id and para.chapter_id in chapters:
            if para.topic_id and para.topic_id not in chapters[para.chapter_id].children:
                chapters[para.chapter_id].children.append(para.topic_id)
        if para.topic_id and para.topic_id in topics:
            if para.subtopic_id and para.subtopic_id not in topics[para.topic_id].children:
                topics[para.topic_id].children.append(para.subtopic_id)
        if para.subtopic_id and para.subtopic_id in subtopics:
            if para.id not in subtopics[para.subtopic_id].children:
                subtopics[para.subtopic_id].children.append(para.id)

    _logger.info("hierarchy linking complete")
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

    _logger.info("saved knowledge_base.json: %d books, %d chapters, %d topics, %d subtopics, %d paragraphs",
                 len(books), len(chapters), len(topics), len(subtopics), len(paragraphs))


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def build(
    config_path: str = "config.json",
    output_path: Optional[str] = None,
    docs_dir: str = "docs",
    build_index: bool = False,
    force: bool = False,
) -> None:
    """Run the full ingestion pipeline.

    Args:
        config_path: Path to config.json.
        output_path: Output path for knowledge_base.json (default: same dir as config).
        docs_dir:    Directory containing source documents.
        build_index: If True, also build FAISS vector indices after saving KB.
    """
    config = load_config(config_path)

    if output_path is None:
        config_dir = os.path.dirname(os.path.abspath(config_path))
        output_path = os.path.join(config_dir, "knowledge_base.json")

    paragraph_max_chars = config.retrieval.paragraph_max_chars
    paragraph_overlap_chars = config.retrieval.paragraph_overlap_chars

    _logger.info("build started: config=%s, output=%s, docs_dir=%s, build_index=%s",
                 config_path, output_path, docs_dir, build_index)
    _logger.info("build: paragraph_max_chars=%d, paragraph_overlap_chars=%d",
                 paragraph_max_chars, paragraph_overlap_chars)

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
    metadata_map = _build_metadata_map(files_by_type, docs_dir)

    # Step 3: Assign hierarchical IDs
    _logger.info("=== Step 3: Assigning hierarchical IDs ===")
    hierarchy_ids = _assign_hierarchical_ids(files_by_type, metadata_map, docs_dir)

    # Step 4: Parse all files
    print("Parsing source files...")
    all_paragraphs: List[Paragraph] = []
    files_parsed = 0
    total_subtopics = 0

    # Collect all files to parse in order
    files_to_parse: List[Tuple[str, str, Dict[str, str]]] = []
    for ext in (".md", ".txt", ".pdf"):
        for fpath in files_by_type.get(ext, []):
            meta = metadata_map.get(fpath, {})
            files_to_parse.append((fpath, ext, meta))

    for fpath, ext, meta in tqdm(files_to_parse, desc="Parsing files",
                                  disable=not _HAS_TQDM, unit="file"):
        paras = _parse_file(
            fpath, ext, meta, paragraph_max_chars, paragraph_overlap_chars,
            hierarchy_ids,
        )
        all_paragraphs.extend(paras)
        files_parsed += 1
        # Count subtopics per file
        for p in paras:
            if p.subtopic_id:
                total_subtopics += 1

    print(f"  Files parsed: {files_parsed}")
    print(f"  Paragraphs extracted: {len(all_paragraphs)}")
    print(f"  Paragraphs with subtopic_id: {total_subtopics}")

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
    print("Knowledge base saved.")

    # Step 8: Build vector indices (optional, controlled by --index flag)
    if build_index:
        print()
        _build_vector_index(
            config_path=config_path,
            kb_path=output_path,
            index_dir=os.path.join(os.path.dirname(os.path.abspath(config_path)), "index"),
            force=force,
        )
        print("\nBuild complete with vector index.")
    else:
        print("Done. (Run with --index to also build FAISS vector indices)")
        _logger.info("build complete (no index)")


# ---------------------------------------------------------------------------
# Index building (vectorization)
# ---------------------------------------------------------------------------

def _build_vector_index(
    config_path: str = "config.json",
    kb_path: Optional[str] = None,
    index_dir: str = "index",
    force: bool = False,
) -> None:
    """Build FAISS vector indices from an existing knowledge_base.json.

    This is the one-time vectorization step. It loads the knowledge base,
    creates an EmbeddingClient, builds 4 FAISS indices (book/chapter/topic/
    paragraph), and saves them to a directory.

    Args:
        config_path: Path to config.json (for embedding model config).
        kb_path:     Path to knowledge_base.json (default: same dir as config).
        index_dir:   Output directory for FAISS indices (default: "index/").
        force:       If True, rebuild even if index already exists.
    """
    from dsa_mentor.embeddings import EmbeddingClient
    from dsa_mentor.index.multi import MultiIndexManager

    config = load_config(config_path)

    if kb_path is None:
        config_dir = os.path.dirname(os.path.abspath(config_path))
        kb_path = os.path.join(config_dir, "knowledge_base.json")

    kb_path = os.path.abspath(kb_path)
    index_dir = os.path.abspath(index_dir)

    # Check if index already exists
    if not force and os.path.isdir(index_dir):
        faiss_files = [
            os.path.join(index_dir, "books", "index.faiss"),
            os.path.join(index_dir, "chapters", "index.faiss"),
            os.path.join(index_dir, "topics", "index.faiss"),
            os.path.join(index_dir, "subtopics", "index.faiss"),
            os.path.join(index_dir, "paragraphs", "index.faiss"),
        ]
        if all(os.path.exists(f) for f in faiss_files):
            print(f"Index already exists at {index_dir}/ (use --force to rebuild)")
            return

    if not os.path.isfile(kb_path):
        print(f"Error: knowledge_base.json not found at {kb_path}")
        print("Run 'python -m dsa_mentor.ingestion.build' first to build the knowledge base.")
        return

    # Load knowledge base
    print(f"\nLoading knowledge base from {kb_path}...")
    with open(kb_path, "r", encoding="utf-8", errors="replace") as f:
        kb_data = json.load(f)

    from dsa_mentor.models import Book, Chapter, Topic, Paragraph, Subtopic

    nodes: Dict[str, Any] = {}
    for book_dict in kb_data.get("books", []):
        nodes[book_dict["id"]] = Book(**book_dict)
    for ch_dict in kb_data.get("chapters", []):
        nodes[ch_dict["id"]] = Chapter(**ch_dict)
    for topic_dict in kb_data.get("topics", []):
        nodes[topic_dict["id"]] = Topic(**topic_dict)
    for para_dict in kb_data.get("paragraphs", []):
        nodes[para_dict["id"]] = Paragraph(**para_dict)
    for sub_dict in kb_data.get("subtopics", []):
        nodes[sub_dict["id"]] = Subtopic(**sub_dict)

    paragraphs = [n for n in nodes.values() if isinstance(n, Paragraph)]
    books = [n for n in nodes.values() if isinstance(n, Book)]
    chapters = [n for n in nodes.values() if isinstance(n, Chapter)]
    topics = [n for n in nodes.values() if isinstance(n, Topic)]
    subtopics = [n for n in nodes.values() if isinstance(n, Subtopic)]

    print(f"  Loaded: {len(books)} books, {len(chapters)} chapters, "
          f"{len(topics)} topics, {len(subtopics)} subtopics, {len(paragraphs)} paragraphs")

    # Initialize embedding client
    print("\nInitializing embedding client...")
    emb_client = EmbeddingClient(config)
    print(f"  Embedding backend: {emb_client._backend}")

    # Build multi-index
    print("\nBuilding FAISS indices...")
    mgr = MultiIndexManager(embedding_client=emb_client)

    # Derive hierarchy mappings from node children arrays
    book_chapters: Dict[str, List[str]] = {}
    chapter_topics: Dict[str, List[str]] = {}
    topic_paragraphs: Dict[str, List[str]] = {}
    subtopic_paragraphs: Dict[str, List[str]] = {}

    for book in books:
        children = getattr(book, "children", []) or []
        if children:
            book_chapters[book.id] = children

    for ch in chapters:
        children = getattr(ch, "children", []) or []
        if children:
            chapter_topics[ch.id] = children

    for topic in topics:
        children = getattr(topic, "children", []) or []
        if children:
            topic_paragraphs[topic.id] = children

    for sub in subtopics:
        children = getattr(sub, "children", []) or []
        if children:
            subtopic_paragraphs[sub.id] = children

    # Build hierarchy dict expected by build_index
    hierarchy = {
        "books": books,
        "chapters": chapters,
        "topics": topics,
        "subtopics": subtopics,
        "book_chapters": book_chapters,
        "chapter_topics": chapter_topics,
        "topic_paragraphs": topic_paragraphs,
        "subtopic_paragraphs": subtopic_paragraphs,
    }

    # Build all indices at once (this populates all internal dicts)
    mgr.build_index(paragraphs, hierarchy)

    print(f"  [book] Index built: {mgr._book_count()} vectors")
    print(f"  [chapter] Index built: {mgr._chapter_count()} vectors")
    print(f"  [topic] Index built: {mgr._topic_count()} vectors")
    print(f"  [subtopic] Index built: {mgr._subtopic_count()} vectors")
    print(f"  [paragraph] Index built: {mgr._paragraph_count()} vectors")

    # Save index to disk
    print(f"\nSaving index to {index_dir}/...")
    mgr.save(index_dir)
    print(f"Index saved: {len(books)} books, {len(chapters)} chapters, "
          f"{len(topics)} topics, {mgr._subtopic_count()} subtopics, "
          f"{mgr._paragraph_count()} paragraphs")
    _logger.info("vector index built: %s", index_dir)


# ---------------------------------------------------------------------------
# Main orchestration
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
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable debug logging",
    )
    parser.add_argument(
        "--index", action="store_true",
        help="Also build FAISS vector indices after saving knowledge base",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Force rebuild of vector indices even if they already exist",
    )
    args = parser.parse_args()

    _setup_logging(verbose=args.verbose)

    build(
        config_path=args.config,
        output_path=args.output,
        docs_dir=args.docs_dir,
        build_index=args.index,
        force=args.force,
    )


if __name__ == "__main__":
    main()
