"""Dataclasses for the knowledge hierarchy and retrieval result.

Implements the spec §4 (knowledge hierarchy) and §60 (RetrievalResult) contracts.
"""

from __future__ import annotations

import dataclasses
from dataclasses import asdict, fields
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Hierarchy base
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class HierarchyNode:
    """Base class for all hierarchy levels (spec §4)."""
    id: str
    title: str
    level: str          # "book" | "chapter" | "topic" | "subtopic" | "paragraph"
    parent_id: Optional[str]
    children: List[str]  # list of child node ids

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Paragraph
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class Paragraph(HierarchyNode):
    """Atomic evidence unit — spec §5.

    Every paragraph knows its full provenance within the hierarchy and its
    source document.
    """
    content: Optional[str] = None
    source_file: Optional[str] = None
    source_url: Optional[str] = None
    page_number: Optional[int] = None
    license: Optional[str] = None
    corpus_id: Optional[str] = None
    book_id: Optional[str] = None
    chapter_id: Optional[str] = None
    topic_id: Optional[str] = None
    subtopic_id: Optional[str] = None
    hierarchical_chunk_id: Optional[str] = None
    paragraph_id: Optional[str] = None
    prev_paragraph_id: Optional[str] = None
    next_paragraph_id: Optional[str] = None

    def __post_init__(self) -> None:
        if self.paragraph_id is None:
            self.paragraph_id = self.id

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Book
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class Book(HierarchyNode):
    corpus_id: Optional[str] = None
    license: Optional[str] = None

    def __post_init__(self) -> None:
        if self.level is None:
            self.level = "book"


# ---------------------------------------------------------------------------
# Chapter
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class Chapter(HierarchyNode):
    book_id: Optional[str] = None
    chapter_id: Optional[str] = None

    def __post_init__(self) -> None:
        if self.level is None:
            self.level = "chapter"
        if self.chapter_id is None:
            self.chapter_id = self.id


# ---------------------------------------------------------------------------
# Topic
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class Topic(HierarchyNode):
    chapter_id: Optional[str] = None
    book_id: Optional[str] = None
    full_text: Optional[str] = None  # spec §7 — full topic text for expansion

    def __post_init__(self) -> None:
        if self.level is None:
            self.level = "topic"


# ---------------------------------------------------------------------------
# Subtopic
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class Subtopic(HierarchyNode):
    topic_id: Optional[str] = None
    chapter_id: Optional[str] = None
    book_id: Optional[str] = None

    def __post_init__(self) -> None:
        if self.level is None:
            self.level = "subtopic"


# ---------------------------------------------------------------------------
# KneeData
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class KneeData:
    """Describes the result of knee detection at one index level.

    Fields (per spec §12):
        index:        which level ("book", "chapter", "topic", "paragraph")
        candidate_k:  how many candidates were considered
        selected_k:   how many were retained after knee detection
        knee_index:   the rank at which the knee was detected (1-based)
        threshold:    the similarity threshold used
    """
    index: str
    candidate_k: int
    selected_k: int
    knee_index: int
    threshold: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# ToolCall
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class ToolCall:
    query: str
    results: List[Any]
    index: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "results": self._safe_results(self.results),
            "index": self.index,
        }

    @staticmethod
    def _safe_results(results: List[Any]) -> List[Any]:
        out = []
        for r in results:
            if isinstance(r, dict):
                out.append(ToolCall._sanitize_dict(r))
            elif hasattr(r, "to_dict"):
                out.append(r.to_dict())
            else:
                out.append(r)
        return out

    @staticmethod
    def _sanitize_dict(d: Dict[str, Any]) -> Dict[str, Any]:
        out = {}
        for k, v in d.items():
            if isinstance(v, (str, int, float, bool, type(None))):
                out[k] = v
            elif isinstance(v, dict):
                out[k] = ToolCall._sanitize_dict(v)
            elif isinstance(v, (list, tuple)):
                out[k] = [ToolCall._sanitize_item(i) for i in v]
            elif hasattr(v, "to_dict"):
                out[k] = v.to_dict()
            else:
                out[k] = str(v)
        return out

    @staticmethod
    def _sanitize_item(item: Any) -> Any:
        if isinstance(item, dict):
            return ToolCall._sanitize_dict(item)
        elif isinstance(item, (str, int, float, bool, type(None))):
            return item
        elif hasattr(item, "to_dict"):
            return item.to_dict()
        else:
            return str(item)


# ---------------------------------------------------------------------------
# RetrievalResult
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class RetrievalResult:
    """Spec §60 — serialisable retrieval state object.

    Contains all retrieval-level outputs, knee metadata, tool calls, and
    context budget tracking.
    """
    query: str
    books: List[Any] = dataclasses.field(default_factory=list)
    chapters: List[Any] = dataclasses.field(default_factory=list)
    topics: List[Any] = dataclasses.field(default_factory=list)
    paragraphs: List[Any] = dataclasses.field(default_factory=list)
    expanded_topics: List[Any] = dataclasses.field(default_factory=list)
    knee: Optional[KneeData] = None
    knees: Optional[Dict[str, KneeData]] = None  # per-level knee data (Phase 4)
    tool_calls: List[ToolCall] = dataclasses.field(default_factory=list)
    context_tokens: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable dict.

        Converts every field to JSON-safe types (str/int/float/bool/list/dict).
        """
        result: Dict[str, Any] = {
            "query": self.query,
            "books": self._serialize_list(self.books),
            "chapters": self._serialize_list(self.chapters),
            "topics": self._serialize_list(self.topics),
            "paragraphs": self._serialize_list(self.paragraphs),
            "expanded_topics": self._serialize_list(self.expanded_topics),
            "context_tokens": self.context_tokens,
        }
        if self.knees is not None:
            result["knees"] = {k: v.to_dict() for k, v in self.knees.items()}
        if self.knee is not None:
            result["knee"] = self.knee.to_dict()
        if self.tool_calls:
            result["tool_calls"] = [tc.to_dict() for tc in self.tool_calls]
        return result

    @staticmethod
    def _serialize_list(items: List[Any]) -> List[Any]:
        out = []
        for item in items:
            if isinstance(item, dict):
                out.append(RetrievalResult._sanitize_dict(item))
            elif hasattr(item, "to_dict"):
                out.append(item.to_dict())
            elif isinstance(item, (str, int, float, bool, type(None))):
                out.append(item)
            else:
                out.append(str(item))
        return out

    @staticmethod
    def _sanitize_dict(d: Dict[str, Any]) -> Dict[str, Any]:
        out = {}
        for k, v in d.items():
            if isinstance(v, (str, int, float, bool, type(None))):
                out[k] = v
            elif isinstance(v, dict):
                out[k] = RetrievalResult._sanitize_dict(v)
            elif isinstance(v, (list, tuple)):
                out[k] = [RetrievalResult._sanitize_item(i) for i in v]
            elif hasattr(v, "to_dict"):
                out[k] = v.to_dict()
            else:
                out[k] = str(v)
        return out

    @staticmethod
    def _sanitize_item(item: Any) -> Any:
        if isinstance(item, dict):
            return RetrievalResult._sanitize_dict(item)
        elif isinstance(item, (str, int, float, bool, type(None))):
            return item
        elif hasattr(item, "to_dict"):
            return item.to_dict()
        else:
            return str(item)


# ---------------------------------------------------------------------------
# KnowledgeManifest
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class KnowledgeManifest:
    """Spec §62 — reproducibility manifest.

    Tracks source-file hashes and corpus size for reproducibility.
    """
    file_hashes: Dict[str, str]          # source_file -> sha256 hex digest
    total_nodes: int
    total_paragraphs: int
    created_at: str                      # ISO 8601 UTC timestamp

    def to_dict(self) -> Dict[str, Any]:
        return {
            "file_hashes": self.file_hashes,
            "total_nodes": self.total_nodes,
            "total_paragraphs": self.total_paragraphs,
            "created_at": self.created_at,
        }

    @classmethod
    def create(cls, file_hashes: Dict[str, str], total_nodes: int,
               total_paragraphs: int) -> "KnowledgeManifest":
        return cls(
            file_hashes=file_hashes,
            total_nodes=total_nodes,
            total_paragraphs=total_paragraphs,
            created_at=datetime.now(timezone.utc).isoformat(),
        )
