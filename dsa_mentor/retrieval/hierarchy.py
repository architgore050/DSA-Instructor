"""Knee-aware hierarchical retriever.

Implements spec §12 (knee detection at every index level) and spec §13
(threshold fallback). Replaces the fixed top-k pipeline from Phase 3 with
dynamic evidence selection while preserving the same coarse-to-fine flow:

    query -> book -> chapter -> topic -> paragraph

Each level searches with a larger candidate pool, then applies
``detect_knee()`` to determine the actual selection.

When ``knee_enabled=False``, falls back to fixed top-k behaviour for
ablation studies (spec §50).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from ..config import Config, KneeParams
from ..embeddings import EmbeddingClient
from ..index.base import FAISSIndex
from ..index.flat import FlatRetriever
from ..index.hierarchy import HierarchicalConfig, HierarchicalRetriever
from ..index.multi import MultiIndexManager
from ..models import Book, Chapter, KneeData, Paragraph, RetrievalResult, Subtopic, Topic
from .expand import (
    ParagraphNeighborExpander,
    QueryBroadtherClassifier,
    TopicExpander,
    apply_context_budget,
    estimate_tokens,
)
from .knee import detect_knee

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Knee-aware config
# ---------------------------------------------------------------------------

@dataclass
class KneeHierarchicalConfig:
    """Configuration for the knee-aware hierarchical retriever.

    Combines the fixed top-k defaults (for ablation) with per-level knee
    parameters from the global config.
    """
    # Fixed top-k overrides (used when knee_enabled=False)
    k_book: int = 5
    k_chapter: int = 8
    k_topic: int = 6
    k_subtopic: int = 10
    k_paragraph: int = 15
    flat_k: int = 15

    # Knee params per level (from config.json)
    book_knee: KneeParams = field(
        default_factory=lambda: KneeParams(candidate_k=10, minimum=1, maximum=4))
    chapter_knee: KneeParams = field(
        default_factory=lambda: KneeParams(candidate_k=15, minimum=1, maximum=6))
    topic_knee: KneeParams = field(
        default_factory=lambda: KneeParams(candidate_k=20, minimum=1, maximum=8))
    subtopic_knee: KneeParams = field(
        default_factory=lambda: KneeParams(candidate_k=30, minimum=1, maximum=12))
    paragraph_knee: KneeParams = field(
        default_factory=lambda: KneeParams(candidate_k=40, minimum=3, maximum=20))

    # Global fallback threshold
    similarity_threshold: float = 0.15
    knee_strongness_threshold: float = 0.02

    # Expansion settings (spec §14, §16, §17, §18)
    neighbor_window: int = 2
    max_context_tokens: int = 20000


def _build_knee_config(config: Any) -> KneeHierarchicalConfig:
    """Build KneeHierarchicalConfig from global Config or dict."""
    if isinstance(config, Config):
        r = config.retrieval
        return KneeHierarchicalConfig(
            book_knee=r.book_knee,
            chapter_knee=r.chapter_knee,
            topic_knee=r.topic_knee,
            subtopic_knee=r.subtopic_knee,
            paragraph_knee=r.paragraph_knee,
            similarity_threshold=r.similarity_threshold,
            knee_strongness_threshold=getattr(r, "knee_strongness_threshold", 0.02),
            neighbor_window=r.neighbor_window,
            max_context_tokens=r.max_context_tokens,
        )
    elif isinstance(config, dict):
        return KneeHierarchicalConfig(**{
            k: v for k, v in config.items() if k in KneeHierarchicalConfig.__dataclass_fields__
        })
    return KneeHierarchicalConfig()


# ---------------------------------------------------------------------------
# KneeHierarchicalRetriever
# ---------------------------------------------------------------------------

class KneeHierarchicalRetriever:
    """Hierarchical retriever with knee-based dynamic selection.

    Parameters
    ----------
    multi_index_manager : MultiIndexManager
        Pre-built multi-level index manager.
    embedding_client : EmbeddingClient or None
        Embedding client for query embedding.
    config : Config, KneeHierarchicalConfig, dict, or None
        Global config (preferred) or knee-specific config.
    """

    def __init__(
        self,
        multi_index_manager: MultiIndexManager,
        embedding_client: Optional[EmbeddingClient] = None,
        config: Any = None,
    ) -> None:
        self._manager = multi_index_manager

        if isinstance(config, Config):
            self._knee_config = _build_knee_config(config)
            # Also build a fallback HierarchicalConfig for knee_enabled=False
            self._fixed_config = HierarchicalConfig()
        elif isinstance(config, KneeHierarchicalConfig):
            self._knee_config = config
            self._fixed_config = HierarchicalConfig()
        elif isinstance(config, dict):
            self._knee_config = _build_knee_config(config)
            self._fixed_config = HierarchicalConfig(**{
                k: v for k, v in config.items()
                if k in HierarchicalConfig.__dataclass_fields__
            })
        else:
            self._knee_config = KneeHierarchicalConfig()
            self._fixed_config = HierarchicalConfig()

        if embedding_client is not None:
            self._embedding_client = embedding_client
        else:
            self._embedding_client = None

        # Expansion components (spec §14, §16, §17)
        self._breadth_classifier = QueryBroadtherClassifier()
        self._topic_expander = TopicExpander()
        self._neighbor_expander = ParagraphNeighborExpander(
            neighbor_window=self._knee_config.neighbor_window,
        )

    @property
    def embedding_client(self) -> EmbeddingClient:
        if self._embedding_client is None:
            self._embedding_client = self._manager.embedding_client
        return self._embedding_client

    # ------------------------------------------------------------------
    # Hierarchical retrieval with knee detection
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        knee_enabled: bool = True,
    ) -> RetrievalResult:
        """Execute the full coarse-to-fine hierarchical pipeline with knee detection.

        Parameters
        ----------
        query : str
            User query.
        knee_enabled : bool
            If True, use knee detection at every level.
            If False, fall back to fixed top-k (ablation).

        Returns
        -------
        RetrievalResult
            Contains books, chapters, topics, paragraphs, expanded_topics,
            and per-level knee metadata.
        """
        if knee_enabled:
            return self._retrieve_knee(query)
        else:
            return self._retrieve_fixed(query)

    def _retrieve_knee(self, query: str) -> RetrievalResult:
        """Knee-aware hierarchical retrieval pipeline."""
        knees: Dict[str, KneeData] = {}

        # Step 1: Book retrieval with knee detection
        book_results = self._manager.search_book(
            query, k=self._knee_config.book_knee.candidate_k)
        book_similarities = [s for _, s in book_results]
        if book_similarities:
            book_knee = detect_knee(
                book_similarities,
                candidate_k=self._knee_config.book_knee.candidate_k,
                minimum=self._knee_config.book_knee.minimum,
                maximum=self._knee_config.book_knee.maximum,
                similarity_threshold=self._knee_config.similarity_threshold,
                knee_strongness_threshold=self._knee_config.knee_strongness_threshold,
            )
            book_knee = KneeData(
                index="book",
                candidate_k=book_knee.candidate_k,
                selected_k=book_knee.selected_k,
                knee_index=book_knee.knee_index,
                threshold=book_knee.threshold,
            )
            knees["book"] = book_knee
            top_books = [br for br, _ in book_results[:book_knee.selected_k]]
            top_book_ids = [b.id for b in top_books]
        else:
            book_knee = KneeData(
                index="book", candidate_k=0, selected_k=0,
                knee_index=0, threshold=self._knee_config.similarity_threshold,
            )
            knees["book"] = book_knee
            top_books = []
            top_book_ids = []
            logger.info("Knee retrieve: no books matched query '%s'", query)
            return RetrievalResult(query=query, knees=knees)

        logger.info("Knee retrieve: %d books selected (knee at rank %d of %d)",
                     len(top_books), book_knee.knee_index, book_knee.candidate_k)

        # Step 2: Chapter retrieval with knee detection
        chapter_results = self._search_chapters_for_books_knee(
            query, top_book_ids, self._knee_config.chapter_knee.candidate_k)
        chapter_similarities = [s for _, s in chapter_results]
        if chapter_similarities:
            chapter_knee = detect_knee(
                chapter_similarities,
                candidate_k=self._knee_config.chapter_knee.candidate_k,
                minimum=self._knee_config.chapter_knee.minimum,
                maximum=self._knee_config.chapter_knee.maximum,
                similarity_threshold=self._knee_config.similarity_threshold,
                knee_strongness_threshold=self._knee_config.knee_strongness_threshold,
            )
            chapter_knee = KneeData(
                index="chapter",
                candidate_k=chapter_knee.candidate_k,
                selected_k=chapter_knee.selected_k,
                knee_index=chapter_knee.knee_index,
                threshold=chapter_knee.threshold,
            )
            knees["chapter"] = chapter_knee
            top_chapters = [cr for cr, _ in chapter_results[:chapter_knee.selected_k]]
            top_chapter_ids = [c.id for c in top_chapters]
        else:
            chapter_knee = KneeData(
                index="chapter", candidate_k=0, selected_k=0,
                knee_index=0, threshold=self._knee_config.similarity_threshold,
            )
            knees["chapter"] = chapter_knee
            top_chapters = []
            top_chapter_ids = []
            logger.info("Knee retrieve: no chapters matched")

        logger.info("Knee retrieve: %d chapters selected (knee at rank %d of %d)",
                     len(top_chapters), chapter_knee.knee_index, chapter_knee.candidate_k)

        # Step 3: Topic retrieval with knee detection
        topic_results = self._search_topics_for_chapters_knee(
            query, top_chapter_ids, self._knee_config.topic_knee.candidate_k)
        topic_similarities = [s for _, s in topic_results]
        if topic_results:
            topic_knee = detect_knee(
                topic_similarities,
                candidate_k=self._knee_config.topic_knee.candidate_k,
                minimum=self._knee_config.topic_knee.minimum,
                maximum=self._knee_config.topic_knee.maximum,
                similarity_threshold=self._knee_config.similarity_threshold,
                knee_strongness_threshold=self._knee_config.knee_strongness_threshold,
            )
            topic_knee = KneeData(
                index="topic",
                candidate_k=topic_knee.candidate_k,
                selected_k=topic_knee.selected_k,
                knee_index=topic_knee.knee_index,
                threshold=topic_knee.threshold,
            )
            knees["topic"] = topic_knee
            top_topics = [tr for tr, _ in topic_results[:topic_knee.selected_k]]
            top_topic_ids = [t.id for t in top_topics]
        else:
            topic_knee = KneeData(
                index="topic", candidate_k=0, selected_k=0,
                knee_index=0, threshold=self._knee_config.similarity_threshold,
            )
            knees["topic"] = topic_knee
            top_topics = []
            top_topic_ids = []
            logger.info("Knee retrieve: no topics matched")

        logger.info("Knee retrieve: %d topics selected (knee at rank %d of %d)",
                      len(top_topics), topic_knee.knee_index, topic_knee.candidate_k)

        # Step 4: Subtopic retrieval with knee detection
        subtopic_results = self._search_subtopics_for_topics_knee(
            query, top_topic_ids, self._knee_config.subtopic_knee.candidate_k)
        subtopic_similarities = [s for _, s in subtopic_results]
        if subtopic_results:
            subtopic_knee = detect_knee(
                subtopic_similarities,
                candidate_k=self._knee_config.subtopic_knee.candidate_k,
                minimum=self._knee_config.subtopic_knee.minimum,
                maximum=self._knee_config.subtopic_knee.maximum,
                similarity_threshold=self._knee_config.similarity_threshold,
                knee_strongness_threshold=self._knee_config.knee_strongness_threshold,
            )
            subtopic_knee = KneeData(
                index="subtopic",
                candidate_k=subtopic_knee.candidate_k,
                selected_k=subtopic_knee.selected_k,
                knee_index=subtopic_knee.knee_index,
                threshold=subtopic_knee.threshold,
            )
            knees["subtopic"] = subtopic_knee
            top_subtopics = [sr for sr, _ in subtopic_results[:subtopic_knee.selected_k]]
            top_subtopic_ids = [s.id for s in top_subtopics]
        else:
            subtopic_knee = KneeData(
                index="subtopic", candidate_k=0, selected_k=0,
                knee_index=0, threshold=self._knee_config.similarity_threshold,
            )
            knees["subtopic"] = subtopic_knee
            top_subtopics = []
            top_subtopic_ids = []
            logger.info("Knee retrieve: no subtopics matched")

        logger.info("Knee retrieve: %d subtopics selected (knee at rank %d of %d)",
                      len(top_subtopics), subtopic_knee.knee_index, subtopic_knee.candidate_k)

        # Step 5: Paragraph retrieval with knee detection
        paragraph_results = self._search_paragraphs_for_subtopics_knee(
            query, top_subtopic_ids, self._knee_config.paragraph_knee.candidate_k)
        paragraph_similarities = [s for _, s in paragraph_results]
        if paragraph_results:
            paragraph_knee = detect_knee(
                paragraph_similarities,
                candidate_k=self._knee_config.paragraph_knee.candidate_k,
                minimum=self._knee_config.paragraph_knee.minimum,
                maximum=self._knee_config.paragraph_knee.maximum,
                similarity_threshold=self._knee_config.similarity_threshold,
                knee_strongness_threshold=self._knee_config.knee_strongness_threshold,
            )
            paragraph_knee = KneeData(
                index="paragraph",
                candidate_k=paragraph_knee.candidate_k,
                selected_k=paragraph_knee.selected_k,
                knee_index=paragraph_knee.knee_index,
                threshold=paragraph_knee.threshold,
            )
            knees["paragraph"] = paragraph_knee
            top_paragraphs = [pr for pr, _ in paragraph_results[:paragraph_knee.selected_k]]
        else:
            paragraph_knee = KneeData(
                index="paragraph", candidate_k=0, selected_k=0,
                knee_index=0, threshold=self._knee_config.similarity_threshold,
            )
            knees["paragraph"] = paragraph_knee
            top_paragraphs = []

        logger.info("Knee retrieve: %d paragraphs selected (knee at rank %d of %d)",
                      len(top_paragraphs), paragraph_knee.knee_index,
                      paragraph_knee.candidate_k)

        # Step 6: Flat search as complementary fallback (spec §87 Phase 2)
        # Searches entire corpus, deduplicates against hierarchical results
        hierarchical_ids: Set[str] = {p.id for p in top_paragraphs}
        flat_paragraphs: List[Tuple[Paragraph, float]] = []

        if not top_paragraphs or len(top_paragraphs) < 3:
            flat_results = self._manager.search_paragraph(
                query, k=self._knee_config.paragraph_knee.candidate_k)
            for p, score in flat_results:
                if p.id not in hierarchical_ids:
                    flat_paragraphs.append((p, score))
                    hierarchical_ids.add(p.id)

        flat_paragraphs.sort(key=lambda x: x[1], reverse=True)
        logger.info("Knee retrieve: flat fallback added %d paragraphs", len(flat_paragraphs))

        # Merge flat results into paragraphs
        all_paragraphs = list(top_paragraphs) + [p for p, _ in flat_paragraphs]

        # Step 7: Query breadth classification → topic expansion → neighbor expansion → context budget
        breadth = self._breadth_classifier.classify(query)
        logger.info("Knee retrieve: query breadth = %s (query='%s')", breadth, query)

        # Topic expansion (spec §14) — expand topics with full_text
        expanded_topics = self._topic_expander.expand(top_topics, breadth)
        logger.info("Knee retrieve: expanded %d topics (breadth=%s)", len(expanded_topics), breadth)

        # Build subtopic → paragraph id map for neighbor expansion (spec §17)
        subtopic_paragraph_map: Dict[str, List[str]] = {}
        for para in all_paragraphs:
            sid = para.subtopic_id
            if sid:
                subtopic_paragraph_map.setdefault(sid, []).append(para.id)

        # Also build a full id → paragraph map from all retrieved paragraphs
        all_para_by_id: Dict[str, Paragraph] = {p.id: p for p in all_paragraphs}

        # Paragraph neighbor expansion (spec §17)
        expanded_paras = self._neighbor_expander.expand(
            all_paragraphs,
            subtopic_paragraph_map,
            all_para_by_id,
        )
        logger.info("Knee retrieve: neighbor-expanded %d → %d paragraphs",
                      len(all_paragraphs), len(expanded_paras))

        # Compute context tokens: paragraphs + expanded topic full_text
        para_tokens = sum(estimate_tokens(p.content or "") for p in expanded_paras)
        topic_tokens = 0
        for t in expanded_topics:
            if t.full_text and t.full_text.strip():
                topic_tokens += estimate_tokens(t.full_text)
        total_tokens = para_tokens + topic_tokens
        logger.info("Knee retrieve: total context tokens = %d (para=%d, topics=%d, budget=%d)",
                      total_tokens, para_tokens, topic_tokens, self._knee_config.max_context_tokens)

        # Context budget enforcement (spec §18)
        if total_tokens > self._knee_config.max_context_tokens:
            expanded_paras = apply_context_budget(
                expanded_paras, self._knee_config.max_context_tokens,
            )
            logger.info("Knee retrieve: truncated to %d paragraphs (budget=%d)",
                          len(expanded_paras), self._knee_config.max_context_tokens)

        # Recompute final token count
        final_tokens = sum(estimate_tokens(p.content or "") for p in expanded_paras)
        # Add topic full_text tokens (not subject to paragraph budget truncation)
        for t in expanded_topics:
            if t.full_text and t.full_text.strip():
                final_tokens += estimate_tokens(t.full_text)

        return RetrievalResult(
            query=query,
            books=top_books,
            chapters=top_chapters,
            topics=top_topics,
            subtopics=top_subtopics,
            paragraphs=expanded_paras,
            expanded_topics=expanded_topics,
            knee=paragraph_knee,  # paragraph-level knee for backward compat
            knees=knees,
            context_tokens=final_tokens,
        )

    def _search_chapters_for_books_knee(
        self, query: str, book_ids: List[str], k_per_book: int
    ) -> List[Tuple[Chapter, float]]:
        """Search chapters for given books, deduplicated, sorted by score."""
        all_results: List[Tuple[Chapter, float]] = []
        seen_ids: Set[str] = set()

        for book_id in book_ids:
            ch_results = self._manager.search_chapter(
                query, book_ids=[book_id], k=k_per_book)
            for ch, score in ch_results:
                if ch.id not in seen_ids:
                    seen_ids.add(ch.id)
                    all_results.append((ch, score))

        all_results.sort(key=lambda x: x[1], reverse=True)
        return all_results

    def _search_topics_for_chapters_knee(
        self, query: str, chapter_ids: List[str], k_per_chapter: int
    ) -> List[Tuple[Topic, float]]:
        """Search topics for given chapters, deduplicated, sorted by score."""
        all_results: List[Tuple[Topic, float]] = []
        seen_ids: Set[str] = set()

        for ch_id in chapter_ids:
            t_results = self._manager.search_topic(
                query, chapter_ids=[ch_id], k=k_per_chapter)
            for t, score in t_results:
                if t.id not in seen_ids:
                    seen_ids.add(t.id)
                    all_results.append((t, score))

        all_results.sort(key=lambda x: x[1], reverse=True)
        return all_results

    def _search_paragraphs_for_topics_knee(
        self, query: str, topic_ids: List[str], k_per_topic: int
    ) -> List[Tuple[Paragraph, float]]:
        """Search paragraphs for given topics, deduplicated, sorted by score."""
        all_results: List[Tuple[Paragraph, float]] = []
        seen_ids: Set[str] = set()

        for topic_id in topic_ids:
            p_results = self._manager.search_paragraph(
                query, topic_ids=[topic_id], k=k_per_topic)
            for p, score in p_results:
                if p.id not in seen_ids:
                    seen_ids.add(p.id)
                    all_results.append((p, score))

        all_results.sort(key=lambda x: x[1], reverse=True)
        return all_results

    def _search_subtopics_for_topics_knee(
        self, query: str, topic_ids: List[str], k_per_topic: int
    ) -> List[Tuple[Subtopic, float]]:
        """Search subtopics for given topics, deduplicated, sorted by score."""
        all_results: List[Tuple[Subtopic, float]] = []
        seen_ids: Set[str] = set()

        for topic_id in topic_ids:
            st_results = self._manager.search_subtopic(
                query, topic_ids=[topic_id], k=k_per_topic)
            for st, score in st_results:
                if st.id not in seen_ids:
                    seen_ids.add(st.id)
                    all_results.append((st, score))

        all_results.sort(key=lambda x: x[1], reverse=True)
        return all_results

    def _search_paragraphs_for_subtopics_knee(
        self, query: str, subtopic_ids: List[str], k_per_subtopic: int
    ) -> List[Tuple[Paragraph, float]]:
        """Search paragraphs for given subtopics, deduplicated, sorted by score."""
        all_results: List[Tuple[Paragraph, float]] = []
        seen_ids: Set[str] = set()

        # Map subtopic_ids to parent topic_ids for scoped FAISS search
        topic_ids: Set[str] = set()
        for st in self._manager.subtopics.values():
            if st.id in subtopic_ids and st.topic_id:
                topic_ids.add(st.topic_id)

        if topic_ids:
            p_results = self._manager.search_paragraph(
                query, topic_ids=list(topic_ids),
                k=k_per_subtopic * max(len(topic_ids), 1))
            for p, score in p_results:
                if p.id not in seen_ids and p.subtopic_id in subtopic_ids:
                    seen_ids.add(p.id)
                    all_results.append((p, score))

        all_results.sort(key=lambda x: x[1], reverse=True)
        return all_results

    def _expand_topics(self, topics: List[Topic]) -> List[Topic]:
        """Expand topics with full_text (spec §7)."""
        expanded = []
        for topic in topics:
            if topic.full_text and topic.full_text.strip():
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
    # Fixed top-k retrieval (ablation fallback, spec §50)
    # ------------------------------------------------------------------

    def _retrieve_fixed(self, query: str) -> RetrievalResult:
        """Fixed top-k hierarchical retrieval (no knee detection).

        Delegates to the existing HierarchicalRetriever for ablation parity.
        """
        base_retriever = HierarchicalRetriever(
            multi_index_manager=self._manager,
            embedding_client=self._embedding_client,
            config=self._fixed_config,
        )
        result = base_retriever.retrieve(
            query,
            k_book=self._fixed_config.k_book,
            k_chapter=self._fixed_config.k_chapter,
            k_topic=self._fixed_config.k_topic,
            k_paragraph=self._fixed_config.k_paragraph,
        )
        return result

    # ------------------------------------------------------------------
    # Flat retrieval with knee detection
    # ------------------------------------------------------------------

    def retrieve_flat(self, query: str, knee_enabled: bool = True
                      ) -> RetrievalResult:
        """Flat baseline retrieval with optional knee detection.

        Parameters
        ----------
        query : str
            User query.
        knee_enabled : bool
            If True, use knee detection on paragraph scores.
            If False, use fixed top-k.

        Returns
        -------
        RetrievalResult
            Contains only paragraphs (and query, knee metadata).
        """
        if knee_enabled:
            return self._retrieve_flat_knee(query)
        else:
            return self._retrieve_flat_fixed(query)

    def _retrieve_flat_knee(self, query: str) -> RetrievalResult:
        """Flat retrieval with knee detection on paragraph scores."""
        paragraph_results = self._manager.search_paragraph(
            query, k=self._knee_config.paragraph_knee.candidate_k)
        paragraph_similarities = [s for _, s in paragraph_results]

        if paragraph_similarities:
            paragraph_knee = detect_knee(
                paragraph_similarities,
                candidate_k=self._knee_config.paragraph_knee.candidate_k,
                minimum=self._knee_config.paragraph_knee.minimum,
                maximum=self._knee_config.paragraph_knee.maximum,
                similarity_threshold=self._knee_config.similarity_threshold,
                knee_strongness_threshold=self._knee_config.knee_strongness_threshold,
            )
            paragraph_knee = KneeData(
                index="paragraph",
                candidate_k=paragraph_knee.candidate_k,
                selected_k=paragraph_knee.selected_k,
                knee_index=paragraph_knee.knee_index,
                threshold=paragraph_knee.threshold,
            )
            knees = {"paragraph": paragraph_knee}
            top_paragraphs = [pr for pr, _ in paragraph_results[:paragraph_knee.selected_k]]
        else:
            paragraph_knee = KneeData(
                index="paragraph", candidate_k=0, selected_k=0,
                knee_index=0, threshold=self._knee_config.similarity_threshold,
            )
            knees = {"paragraph": paragraph_knee}
            top_paragraphs = []

        # Apply expansion pipeline (spec §14, §16, §17, §18)
        breadth = self._breadth_classifier.classify(query)
        expanded_paras = self._neighbor_expander.expand(
            top_paragraphs,
            topic_paragraph_map={},
            all_paragraphs_by_id={p.id: p for p in top_paragraphs},
        )

        total_tokens = sum(estimate_tokens(p.content or "") for p in expanded_paras)
        if total_tokens > self._knee_config.max_context_tokens:
            expanded_paras = apply_context_budget(
                expanded_paras, self._knee_config.max_context_tokens,
            )

        final_tokens = sum(estimate_tokens(p.content or "") for p in expanded_paras)

        return RetrievalResult(
            query=query,
            paragraphs=expanded_paras,
            knee=paragraph_knee,
            knees=knees,
            context_tokens=final_tokens,
        )

    def _retrieve_flat_fixed(self, query: str) -> RetrievalResult:
        """Fixed top-k flat retrieval (no knee detection)."""
        k = self._fixed_config.flat_k
        paragraph_results = self._manager.search_paragraph(query, k=k)
        paragraphs = [pr for pr, _ in paragraph_results]
        return RetrievalResult(query=query, paragraphs=paragraphs)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def manager(self) -> MultiIndexManager:
        return self._manager

    @property
    def knee_config(self) -> KneeHierarchicalConfig:
        return self._knee_config
