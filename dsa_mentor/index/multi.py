"""Multi-level index manager for hierarchical retrieval.

Implements spec §8 (four vector indices: book/chapter/topic/paragraph) and
spec §87 Phase 3 (hierarchical retrieval architecture).

Manages separate FAISS indices at each hierarchy level, with metadata
mappings for node lookup and position-based filtering.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..embeddings import EmbeddingClient
from ..models import Book, Chapter, Paragraph, Topic
from .base import FAISSIndex

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal node representation for index vectors
# ---------------------------------------------------------------------------

@dataclass
class _NodeVector:
    """A single vector with its associated node and metadata."""
    node_id: str
    title: str
    text: str           # text used for embedding
    node: Any           # Book, Chapter, Topic, or Paragraph object
    parent_id: Optional[str] = None


# ---------------------------------------------------------------------------
# MultiIndexManager
# ---------------------------------------------------------------------------

class MultiIndexManager:
    """Manages book/chapter/topic/paragraph FAISS indices.

    Builds and maintains four separate FAISS indices, one per hierarchy level.
    Each index stores:
      - FAISS index object
      - metadata: node_id -> node mapping
      - reverse mapping: node_id -> FAISS index position

    Parameters
    ----------
    embedding_client : EmbeddingClient or None
        Embedding client for generating vectors. If None, created from config.
    """

    def __init__(self, embedding_client: Optional[EmbeddingClient] = None) -> None:
        self._embedding_client = embedding_client
        self._dimensions: Optional[int] = None

        # Four indices
        self._book_index: Optional[FAISSIndex] = None
        self._chapter_index: Optional[FAISSIndex] = None
        self._topic_index: Optional[FAISSIndex] = None
        self._paragraph_index: Optional[FAISSIndex] = None

        # Metadata: node_id -> node
        self._books: Dict[str, Book] = {}
        self._chapters: Dict[str, Chapter] = {}
        self._topics: Dict[str, Topic] = {}
        self._paragraphs: Dict[str, Paragraph] = {}

        # Reverse mapping: node_id -> FAISS index position
        self._book_pos: Dict[str, int] = {}
        self._chapter_pos: Dict[str, int] = {}
        self._topic_pos: Dict[str, int] = {}
        self._paragraph_pos: Dict[str, int] = {}

        # Hierarchy structure
        self._book_chapters: Dict[str, List[str]] = {}    # book_id -> [chapter_ids]
        self._chapter_topics: Dict[str, List[str]] = {}   # chapter_id -> [topic_ids]
        self._topic_paragraphs: Dict[str, List[str]] = {} # topic_id -> [paragraph_ids]

    @property
    def embedding_client(self) -> EmbeddingClient:
        if self._embedding_client is None:
            from ..embeddings import EmbeddingClient
            self._embedding_client = EmbeddingClient()
        return self._embedding_client

    # ------------------------------------------------------------------
    # Index building
    # ------------------------------------------------------------------

    def build_index(self, paragraphs: List[Paragraph],
                    hierarchy: Dict[str, Any]) -> None:
        """Build all four hierarchy-level indices.

        Parameters
        ----------
        paragraphs : list[Paragraph]
            All paragraphs in the corpus.
        hierarchy : dict
            Hierarchical structure mapping:
              - "books": list of Book objects
              - "chapters": list of Chapter objects
              - "topics": list of Topic objects
              - "book_chapters": {book_id: [chapter_ids]}
              - "chapter_topics": {chapter_id: [topic_ids]}
              - "topic_paragraphs": {topic_id: [paragraph_ids]}
        """
        books = hierarchy.get("books", [])
        chapters = hierarchy.get("chapters", [])
        topics = hierarchy.get("topics", [])

        self._book_chapters = hierarchy.get("book_chapters", {})
        self._chapter_topics = hierarchy.get("chapter_topics", {})
        self._topic_paragraphs = hierarchy.get("topic_paragraphs", {})

        # Store all nodes
        for book in books:
            self._books[book.id] = book
        for chapter in chapters:
            self._chapters[chapter.id] = chapter
        for topic in topics:
            self._topics[topic.id] = topic
        for para in paragraphs:
            self._paragraphs[para.id] = para

        # Build indices level by level
        self._build_book_index(books)
        self._build_chapter_index(chapters)
        self._build_topic_index(topics)
        self._build_paragraph_index(paragraphs)

        logger.info("MultiIndexManager built: %d books, %d chapters, "
                     "%d topics, %d paragraphs",
                     self._book_count(), self._chapter_count(),
                     self._topic_count(), self._paragraph_count())

    def _build_book_index(self, books: List[Book]) -> None:
        """Build the book-level index. Each book gets an embedding from its
        title + summary (aggregated from chapter texts if no explicit summary)."""
        if not books:
            logger.warning("No books to index")
            return

        texts = []
        nodes = []
        for book in books:
            # Aggregate book text from chapters if no explicit summary
            book_text = book.title
            # Book dataclass has no content field; aggregate from child chapters
            child_chapter_ids = self._book_chapters.get(book.id, [])
            for ch_id in child_chapter_ids:
                ch = self._chapters.get(ch_id)
                if ch:
                    ch_text = ch.content if hasattr(ch, 'content') and ch.content else ''
                    book_text += f"\n\n{ch.title}: {ch_text}"
            texts.append(book_text)
            nodes.append(_NodeVector(
                node_id=book.id,
                title=book.title,
                text=book_text,
                node=book,
            ))

        vectors = self._embed_texts(texts)
        self._book_index = FAISSIndex(vectors.shape[1], metric="cosine")
        self._book_index.add_with_ids(vectors, [n.node_id for n in nodes])

        for i, node in enumerate(nodes):
            self._book_pos[node.node_id] = i

    def _build_chapter_index(self, chapters: List[Chapter]) -> None:
        """Build the chapter-level index. Each chapter gets an embedding from
        its title + content."""
        if not chapters:
            logger.warning("No chapters to index")
            return

        texts = []
        nodes = []
        for ch in chapters:
            ch_text = ch.title
            ch_content = ch.content if hasattr(ch, 'content') and ch.content else ''
            if ch_content:
                ch_text += f"\n\n{ch_content}"
            texts.append(ch_text)
            nodes.append(_NodeVector(
                node_id=ch.id,
                title=ch.title,
                text=ch_text,
                node=ch,
                parent_id=ch.book_id,
            ))

        vectors = self._embed_texts(texts)
        self._chapter_index = FAISSIndex(vectors.shape[1], metric="cosine")
        self._chapter_index.add_with_ids(vectors, [n.node_id for n in nodes])

        for i, node in enumerate(nodes):
            self._chapter_pos[node.node_id] = i

    def _build_topic_index(self, topics: List[Topic]) -> None:
        """Build the topic-level index. Each topic gets an embedding from its
        title + full_text (spec §7)."""
        if not topics:
            logger.warning("No topics to index")
            return

        texts = []
        nodes = []
        for topic in topics:
            topic_text = topic.title
            if topic.full_text:
                topic_text += f"\n\n{topic.full_text}"
            elif topic.content:
                topic_text += f"\n\n{topic.content}"
            texts.append(topic_text)
            nodes.append(_NodeVector(
                node_id=topic.id,
                title=topic.title,
                text=topic_text,
                node=topic,
                parent_id=topic.chapter_id,
            ))

        vectors = self._embed_texts(texts)
        self._topic_index = FAISSIndex(vectors.shape[1], metric="cosine")
        self._topic_index.add_with_ids(vectors, [n.node_id for n in nodes])

        for i, node in enumerate(nodes):
            self._topic_pos[node.node_id] = i

    def _build_paragraph_index(self, paragraphs: List[Paragraph]) -> None:
        """Build the paragraph-level index. One vector per paragraph
        (same as flat retriever)."""
        if not paragraphs:
            logger.warning("No paragraphs to index")
            return

        texts = []
        valid_ids = []
        for para in paragraphs:
            if para.content is None or para.content.strip() == "":
                logger.warning("Skipping paragraph %s: empty content", para.id)
                continue
            texts.append(para.content)
            valid_ids.append(para.id)

        if not texts:
            logger.warning("All paragraphs had empty content — no indexing")
            return

        vectors = self._embed_texts(texts)
        self._paragraph_index = FAISSIndex(vectors.shape[1], metric="cosine")
        self._paragraph_index.add_with_ids(vectors, valid_ids)

        for i, para_id in enumerate(valid_ids):
            self._paragraph_pos[para_id] = i

    def _embed_texts(self, texts: List[str]) -> np.ndarray:
        """Embed texts and cache dimensionality."""
        vectors = self.embedding_client.embed(texts)
        if self._dimensions is None and vectors.shape[1] > 0:
            self._dimensions = vectors.shape[1]
        return vectors

    # ------------------------------------------------------------------
    # Search methods
    # ------------------------------------------------------------------

    def search_book(self, query: str, k: int = 5
                    ) -> List[Tuple[Book, float]]:
        """Search the book index, returning top-k books with similarity scores.

        Parameters
        ----------
        query : str
            Query text.
        k : int
            Number of results.

        Returns
        -------
        list[tuple[Book, float]]
            (book, similarity) sorted by descending similarity.
            Returns empty list if index is not built or empty.
        """
        if self._book_index is None or self._book_index.count() == 0:
            return []

        query_vec = self.embedding_client.embed([query])
        distances, indices = self._book_index.search(query_vec, k=k)

        results: List[Tuple[Book, float]] = []
        for j in range(distances.shape[1]):
            idx = int(indices[0, j])
            dist = float(distances[0, j])
            if idx == -1:
                continue
            book_id = self._book_index.metadata[idx]
            book = self._books.get(book_id)
            if book is not None:
                results.append((book, dist))

        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def search_chapter(self, query: str, book_ids: Optional[List[str]] = None,
                       k: int = 8) -> List[Tuple[Chapter, float]]:
        """Search the chapter index, restricted to given book_ids.

        Parameters
        ----------
        query : str
            Query text.
        book_ids : list[str] or None
            If provided, only return chapters belonging to these books.
        k : int
            Number of results.

        Returns
        -------
        list[tuple[Chapter, float]]
            (chapter, similarity) sorted by descending similarity.
            Returns empty list if index is not built or empty.
        """
        if self._chapter_index is None or self._chapter_index.count() == 0:
            return []

        query_vec = self.embedding_client.embed([query])
        distances, indices = self._chapter_index.search(query_vec, k=k)

        results: List[Tuple[Chapter, float]] = []
        for j in range(distances.shape[1]):
            idx = int(indices[0, j])
            dist = float(distances[0, j])
            if idx == -1:
                continue
            ch_id = self._chapter_index.metadata[idx]
            ch = self._chapters.get(ch_id)
            if ch is None:
                continue
            if book_ids is not None and ch.book_id not in book_ids:
                continue
            results.append((ch, dist))

        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def search_topic(self, query: str, chapter_ids: Optional[List[str]] = None,
                     k: int = 6) -> List[Tuple[Topic, float]]:
        """Search the topic index, restricted to given chapter_ids.

        Parameters
        ----------
        query : str
            Query text.
        chapter_ids : list[str] or None
            If provided, only return topics belonging to these chapters.
        k : int
            Number of results.

        Returns
        -------
        list[tuple[Topic, float]]
            (topic, similarity) sorted by descending similarity.
            Returns empty list if index is not built or empty.
        """
        if self._topic_index is None or self._topic_index.count() == 0:
            return []

        query_vec = self.embedding_client.embed([query])
        distances, indices = self._topic_index.search(query_vec, k=k)

        results: List[Tuple[Topic, float]] = []
        for j in range(distances.shape[1]):
            idx = int(indices[0, j])
            dist = float(distances[0, j])
            if idx == -1:
                continue
            topic_id = self._topic_index.metadata[idx]
            topic = self._topics.get(topic_id)
            if topic is None:
                continue
            if chapter_ids is not None and topic.chapter_id not in chapter_ids:
                continue
            results.append((topic, dist))

        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def search_paragraph(self, query: str, topic_ids: Optional[List[str]] = None,
                         k: int = 15) -> List[Tuple[Paragraph, float]]:
        """Search the paragraph index, restricted to given topic_ids.

        Parameters
        ----------
        query : str
            Query text.
        topic_ids : list[str] or None
            If provided, only return paragraphs belonging to these topics.
        k : int
            Number of results.

        Returns
        -------
        list[tuple[Paragraph, float]]
            (paragraph, similarity) sorted by descending similarity.
            Returns empty list if index is not built or empty.
        """
        if self._paragraph_index is None or self._paragraph_index.count() == 0:
            return []

        query_vec = self.embedding_client.embed([query])
        distances, indices = self._paragraph_index.search(query_vec, k=k)

        results: List[Tuple[Paragraph, float]] = []
        for j in range(distances.shape[1]):
            idx = int(indices[0, j])
            dist = float(distances[0, j])
            if idx == -1:
                continue
            para_id = self._paragraph_index.metadata[idx]
            para = self._paragraphs.get(para_id)
            if para is None:
                continue
            if topic_ids is not None and para.topic_id not in topic_ids:
                continue
            results.append((para, dist))

        results.sort(key=lambda x: x[1], reverse=True)
        return results

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Persist all 4 indices + metadata to disk.

        Parameters
        ----------
        path : str
            Directory path. Creates subdirectories:
              books/, chapters/, topics/, paragraphs/
            Each containing index.faiss and metadata.json.
            Also saves:
              - nodes.json (all node objects)
              - hierarchy.json (book_chapters, chapter_topics, topic_paragraphs)
        """
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)

        import faiss  # noqa: F811

        # Save each index
        for name, idx in [("books", self._book_index),
                          ("chapters", self._chapter_index),
                          ("topics", self._topic_index),
                          ("paragraphs", self._paragraph_index)]:
            if idx is not None:
                save_dir = p / name
                save_dir.mkdir(parents=True, exist_ok=True)
                faiss.write_index(idx._index, str(save_dir / "index.faiss"))
                with open(save_dir / "metadata.json", "w", encoding="utf-8") as f:
                    json.dump({
                        "dimensions": idx._dimensions,
                        "metric": idx._metric,
                        "ids": idx._metadata,
                    }, f, indent=2)

        # Save node mappings
        nodes_data = {
            "books": {bid: b.to_dict() for bid, b in self._books.items()},
            "chapters": {cid: c.to_dict() for cid, c in self._chapters.items()},
            "topics": {tid: t.to_dict() for tid, t in self._topics.items()},
            "paragraphs": {pid: p.to_dict() for pid, p in self._paragraphs.items()},
        }
        with open(p / "nodes.json", "w", encoding="utf-8") as f:
            json.dump(nodes_data, f, indent=2)

        # Save hierarchy
        hierarchy_data = {
            "book_chapters": self._book_chapters,
            "chapter_topics": self._chapter_topics,
            "topic_paragraphs": self._topic_paragraphs,
        }
        with open(p / "hierarchy.json", "w", encoding="utf-8") as f:
            json.dump(hierarchy_data, f, indent=2)

        # Save positions
        positions = {
            "book_pos": self._book_pos,
            "chapter_pos": self._chapter_pos,
            "topic_pos": self._topic_pos,
            "paragraph_pos": self._paragraph_pos,
        }
        with open(p / "positions.json", "w", encoding="utf-8") as f:
            json.dump(positions, f, indent=2)

        logger.info("Saved MultiIndexManager to %s", p)

    @classmethod
    def load(cls, path: str,
             embedding_client: Optional[EmbeddingClient] = None
             ) -> "MultiIndexManager":
        """Load all indices + metadata from disk.

        Parameters
        ----------
        path : str
            Directory path containing saved indices and metadata.
        embedding_client : EmbeddingClient or None
            Embedding client to use for search. If None, created from config.

        Returns
        -------
        MultiIndexManager
            Restored manager instance.
        """
        p = Path(path)
        instance = cls(embedding_client=embedding_client)

        import faiss  # noqa: F811

        # Load each index
        for name, attr in [("books", "_book_index"),
                           ("chapters", "_chapter_index"),
                           ("topics", "_topic_index"),
                           ("paragraphs", "_paragraph_index")]:
            save_dir = p / name
            faiss_path = save_dir / "index.faiss"
            meta_path = save_dir / "metadata.json"
            if faiss_path.exists() and meta_path.exists():
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                idx = faiss.read_index(str(faiss_path))
                faiss_idx = FAISSIndex.__new__(FAISSIndex)
                faiss_idx._dimensions = meta["dimensions"]
                faiss_idx._metric = meta["metric"]
                faiss_idx._index = idx
                faiss_idx._metadata = meta["ids"]
                setattr(instance, attr, faiss_idx)

        # Load nodes
        nodes_path = p / "nodes.json"
        if nodes_path.exists():
            with open(nodes_path, "r", encoding="utf-8") as f:
                nodes_data = json.load(f)
            from ..models import Book, Chapter, Topic, Paragraph
            for bid, data in nodes_data.get("books", {}).items():
                instance._books[bid] = Book(**{k: v for k, v in data.items()
                                               if k in Book.__dataclass_fields__})
            for cid, data in nodes_data.get("chapters", {}).items():
                instance._chapters[cid] = Chapter(**{k: v for k, v in data.items()
                                                     if k in Chapter.__dataclass_fields__})
            for tid, data in nodes_data.get("topics", {}).items():
                instance._topics[tid] = Topic(**{k: v for k, v in data.items()
                                                 if k in Topic.__dataclass_fields__})
            for pid, data in nodes_data.get("paragraphs", {}).items():
                instance._paragraphs[pid] = Paragraph(**{k: v for k, v in data.items()
                                                         if k in Paragraph.__dataclass_fields__})

        # Load hierarchy
        hier_path = p / "hierarchy.json"
        if hier_path.exists():
            with open(hier_path, "r", encoding="utf-8") as f:
                hier_data = json.load(f)
            instance._book_chapters = hier_data.get("book_chapters", {})
            instance._chapter_topics = hier_data.get("chapter_topics", {})
            instance._topic_paragraphs = hier_data.get("topic_paragraphs", {})

        # Load positions
        pos_path = p / "positions.json"
        if pos_path.exists():
            with open(pos_path, "r", encoding="utf-8") as f:
                pos_data = json.load(f)
            instance._book_pos = pos_data.get("book_pos", {})
            instance._chapter_pos = pos_data.get("chapter_pos", {})
            instance._topic_pos = pos_data.get("topic_pos", {})
            instance._paragraph_pos = pos_data.get("paragraph_pos", {})

        logger.info("Loaded MultiIndexManager from %s", p)
        return instance

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def _book_count(self) -> int:
        return self._book_index.count() if self._book_index else 0

    def _chapter_count(self) -> int:
        return self._chapter_index.count() if self._chapter_index else 0

    def _topic_count(self) -> int:
        return self._topic_index.count() if self._topic_index else 0

    def _paragraph_count(self) -> int:
        return self._paragraph_index.count() if self._paragraph_index else 0

    @property
    def books(self) -> Dict[str, Book]:
        return dict(self._books)

    @property
    def chapters(self) -> Dict[str, Chapter]:
        return dict(self._chapters)

    @property
    def topics(self) -> Dict[str, Topic]:
        return dict(self._topics)

    @property
    def paragraphs(self) -> Dict[str, Paragraph]:
        return dict(self._paragraphs)

    @property
    def book_chapters(self) -> Dict[str, List[str]]:
        return dict(self._book_chapters)

    @property
    def chapter_topics(self) -> Dict[str, List[str]]:
        return dict(self._chapter_topics)

    @property
    def topic_paragraphs(self) -> Dict[str, List[str]]:
        return dict(self._topic_paragraphs)
