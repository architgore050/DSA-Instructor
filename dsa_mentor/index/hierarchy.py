"""Hierarchical retriever implementing coarse-to-fine pipeline.

Implements spec §87 Phase 3 (hierarchical retrieval) and Phase 2 (flat
ablation). The pipeline flows:

    query -> book -> chapter -> topic -> paragraph

with deduplication at each level and topic expansion (spec §7).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from ..embeddings import EmbeddingClient
from ..models import Book, Chapter, Paragraph, RetrievalResult, Topic
from .base import FAISSIndex
from .flat import FlatRetriever
from .multi import MultiIndexManager

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Default config for hierarchical retrieval
# ---------------------------------------------------------------------------

@dataclass
class HierarchicalConfig:
    """Default retrieval parameters for the hierarchical pipeline.

    These are fixed top_k values (Phase 3); knee detection (Phase 4) will
    replace them with dynamic selection.
    """
    k_book: int = 5
    k_chapter: int = 8
    k_topic: int = 6
    k_paragraph: int = 15
    flat_k: int = 15


# ---------------------------------------------------------------------------
# HierarchicalRetriever
# ---------------------------------------------------------------------------

class HierarchicalRetriever:
    """Coarse-to-fine hierarchical retriever.

    Parameters
    ----------
    multi_index_manager : MultiIndexManager
        Pre-built multi-level index manager.
    embedding_client : EmbeddingClient or None
        Embedding client for query embedding. If None, uses manager's client.
    config : HierarchicalConfig or dict or None
        Retrieval parameters. If dict, converted to HierarchicalConfig.
    """

    def __init__(self, multi_index_manager: MultiIndexManager,
                 embedding_client: Optional[EmbeddingClient] = None,
                 config: Optional[Any] = None) -> None:
        self._manager = multi_index_manager

        if config is None:
            self._config = HierarchicalConfig()
        elif isinstance(config, dict):
            self._config = HierarchicalConfig(**config)
        else:
            self._config = config

        # Use manager's embedding client if none provided
        if embedding_client is not None:
            self._embedding_client = embedding_client
        else:
            self._embedding_client = None

    @property
    def embedding_client(self) -> EmbeddingClient:
        if self._embedding_client is None:
            self._embedding_client = self._manager.embedding_client
        return self._embedding_client

    # ------------------------------------------------------------------
    # Hierarchical retrieval (spec §87 Phase 3)
    # ------------------------------------------------------------------

    def retrieve(self, query: str,
                 k_book: Optional[int] = None,
                 k_chapter: Optional[int] = None,
                 k_topic: Optional[int] = None,
                 k_paragraph: Optional[int] = None
                 ) -> RetrievalResult:
        """Execute the full coarse-to-fine hierarchical retrieval pipeline.

        Pipeline (spec §87):
          1. Search book index -> top-k_book books
          2. For each book, search chapter index -> top-k_chapter chapters
             (deduplicate across books)
          3. For each chapter, search topic index -> top-k_topic topics
             (deduplicate)
          4. For each topic, search paragraph index -> top-k_paragraph
             paragraphs (deduplicate)
          5. Topic expansion: include full_text for each retrieved topic
             (spec §7)
          6. Build and return RetrievalResult

        Parameters
        ----------
        query : str
            User query.
        k_book : int or None
            Override for number of books to retrieve.
        k_chapter : int or None
            Override for number of chapters per book.
        k_topic : int or None
            Override for number of topics per chapter.
        k_paragraph : int or None
            Override for number of paragraphs per topic.

        Returns
        -------
        RetrievalResult
            Contains books, chapters, topics, paragraphs, expanded_topics.
        """
        k_book = k_book or self._config.k_book
        k_chapter = k_chapter or self._config.k_chapter
        k_topic = k_topic or self._config.k_topic
        k_paragraph = k_paragraph or self._config.k_paragraph

        # Step 1: Book retrieval
        book_results = self._manager.search_book(query, k=k_book)
        if not book_results:
            logger.info("Hierarchical retrieve: no books matched query '%s'", query)
            return RetrievalResult(query=query)

        top_books = [br for br, _ in book_results]
        top_book_ids = [b.id for b in top_books]
        logger.info("Hierarchical retrieve: %d books matched", len(top_books))

        # Step 2: Chapter retrieval (restricted to top books, deduplicated)
        chapter_results = self._search_chapters_for_books(query, top_book_ids,
                                                          k_chapter)
        if not chapter_results:
            logger.info("Hierarchical retrieve: no chapters matched")
            return RetrievalResult(query=query, books=top_books)

        top_chapters = [cr for cr, _ in chapter_results]
        top_chapter_ids = [c.id for c in top_chapters]
        logger.info("Hierarchical retrieve: %d chapters matched", len(top_chapters))

        # Step 3: Topic retrieval (restricted to top chapters, deduplicated)
        topic_results = self._search_topics_for_chapters(query, top_chapter_ids,
                                                         k_topic)
        if not topic_results:
            logger.info("Hierarchical retrieve: no topics matched")
            return RetrievalResult(query=query, books=top_books,
                                   chapters=top_chapters)

        top_topics = [tr for tr, _ in topic_results]
        top_topic_ids = [t.id for t in top_topics]
        logger.info("Hierarchical retrieve: %d topics matched", len(top_topics))

        # Step 4: Paragraph retrieval (restricted to top topics, deduplicated)
        paragraph_results = self._search_paragraphs_for_topics(query,
                                                               top_topic_ids,
                                                               k_paragraph)
        if not paragraph_results:
            logger.info("Hierarchical retrieve: no paragraphs matched")
            return RetrievalResult(query=query, books=top_books,
                                   chapters=top_chapters,
                                   topics=top_topics)

        top_paragraphs = [pr for pr, _ in paragraph_results]
        logger.info("Hierarchical retrieve: %d paragraphs matched",
                     len(top_paragraphs))

        # Step 5: Topic expansion (spec §7) — include full_text for each topic
        expanded_topics = self._expand_topics(top_topics)

        return RetrievalResult(
            query=query,
            books=top_books,
            chapters=top_chapters,
            topics=top_topics,
            paragraphs=top_paragraphs,
            expanded_topics=expanded_topics,
        )

    def _search_chapters_for_books(self, query: str, book_ids: List[str],
                                   k_per_book: int
                                   ) -> List[Tuple[Chapter, float]]:
        """Search chapter index restricted to given book_ids, deduplicating.

        When a chapter appears under multiple books (unlikely but possible),
        keep only the highest-scoring match.
        """
        all_results: List[Tuple[Chapter, float]] = []
        seen_ids: Set[str] = set()

        for book_id in book_ids:
            ch_results = self._manager.search_chapter(query,
                                                      book_ids=[book_id],
                                                      k=k_per_book)
            for ch, score in ch_results:
                if ch.id not in seen_ids:
                    seen_ids.add(ch.id)
                    all_results.append((ch, score))
                # Skip duplicates — prefer first (highest-scoring) match

        # Re-sort by score after deduplication
        all_results.sort(key=lambda x: x[1], reverse=True)
        return all_results

    def _search_topics_for_chapters(self, query: str, chapter_ids: List[str],
                                    k_per_chapter: int
                                    ) -> List[Tuple[Topic, float]]:
        """Search topic index restricted to given chapter_ids, deduplicating."""
        all_results: List[Tuple[Topic, float]] = []
        seen_ids: Set[str] = set()

        for ch_id in chapter_ids:
            t_results = self._manager.search_topic(query,
                                                   chapter_ids=[ch_id],
                                                   k=k_per_chapter)
            for t, score in t_results:
                if t.id not in seen_ids:
                    seen_ids.add(t.id)
                    all_results.append((t, score))

        all_results.sort(key=lambda x: x[1], reverse=True)
        return all_results

    def _search_paragraphs_for_topics(self, query: str, topic_ids: List[str],
                                      k_per_topic: int
                                      ) -> List[Tuple[Paragraph, float]]:
        """Search paragraph index restricted to given topic_ids, deduplicating."""
        all_results: List[Tuple[Paragraph, float]] = []
        seen_ids: Set[str] = set()

        for topic_id in topic_ids:
            p_results = self._manager.search_paragraph(query,
                                                       topic_ids=[topic_id],
                                                       k=k_per_topic)
            for p, score in p_results:
                if p.id not in seen_ids:
                    seen_ids.add(p.id)
                    all_results.append((p, score))

        all_results.sort(key=lambda x: x[1], reverse=True)
        return all_results

    def _expand_topics(self, topics: List[Topic]) -> List[Topic]:
        """Expand topics with full_text (spec §7).

        Returns topics that have full_text set. Topics without full_text
        are returned as-is (no expansion needed).
        """
        expanded = []
        for topic in topics:
            if topic.full_text and topic.full_text.strip():
                # Create a copy with full_text included
                expanded_topic = Topic(
                    id=topic.id,
                    title=topic.title,
                    level=topic.level,
                    parent_id=topic.parent_id,
                    children=list(topic.children),
                    chapter_id=topic.chapter_id,
                    book_id=topic.book_id,
                    full_text=topic.full_text,
                )
                expanded.append(expanded_topic)
            else:
                expanded.append(topic)
        return expanded

    # ------------------------------------------------------------------
    # Flat retrieval baseline (spec §87 Phase 2)
    # ------------------------------------------------------------------

    def retrieve_flat(self, query: str, k: Optional[int] = None
                      ) -> RetrievalResult:
        """Flat baseline retrieval — no hierarchy, just paragraph search.

        Uses the paragraph index from the multi-index manager directly.

        Parameters
        ----------
        query : str
            User query.
        k : int or None
            Number of results. Defaults to config.flat_k.

        Returns
        -------
        RetrievalResult
            Contains only paragraphs (and query).
        """
        k = k or self._config.flat_k

        paragraph_results = self._manager.search_paragraph(query, k=k)
        paragraphs = [pr for pr, _ in paragraph_results]

        logger.info("Flat retrieve: %d paragraphs matched", len(paragraphs))

        return RetrievalResult(query=query, paragraphs=paragraphs)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def manager(self) -> MultiIndexManager:
        return self._manager

    @property
    def config(self) -> HierarchicalConfig:
        return self._config
