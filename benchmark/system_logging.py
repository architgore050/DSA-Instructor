"""System logging and latency benchmark for DSA Mentor.

Profiles the entire retrieval pipeline to answer questions a systems designer
would ask about performance, bottlenecks, and efficiency.

Metrics tracked:
    Latency Breakdown:
        - Embedding generation time
        - FAISS search time per hierarchy level (book, chapter, topic, subtopic, paragraph)
        - Knee detection time per level
        - Flat fallback search time
        - Query breadth classification time
        - Topic expansion time
        - Neighbor expansion time
        - Context budget enforcement time
        - Total retrieval time

    Query Complexity:
        - Word count
        - Technical term density (ratio of DSA terms to total words)
        - Concept count (number of distinct DSA concepts mentioned)
        - Query breadth classification (BROAD/MODERATE/NARROW)
        - Multi-concept indicator (whether query spans multiple topics)

    FAISS Search Performance:
        - Candidates considered per level
        - Items selected per level (after knee/fixed filtering)
        - Selection ratio (selected / candidates)
        - Average similarity score per level
        - Number of scoped searches (per parent item)

    Knee Detection:
        - Strong knee detected per level
        - Knee rank per level
        - Fallback threshold used
        - Selection bounds respected

    Context Building:
        - Total paragraphs after expansion
        - Total tokens
        - Context budget utilization (%)
        - Source diversity (unique books, unique sources)
        - Hierarchical vs flat paragraph ratio

    Retrieval Quality Signals:
        - Average similarity score (all levels)
        - Minimum similarity score
        - Paragraphs from hierarchical search
        - Paragraphs from flat fallback
        - Expanded topics count
        - Neighbor-expanded paragraphs

    System Health:
        - Index sizes (books, chapters, topics, subtopics, paragraphs)
        - Total knowledge base nodes
        - Embedding dimension
        - FAISS index type

Usage
-----
    from benchmark.system_logging import PipelineProfiler, SystemHealthReport

    profiler = PipelineProfiler(config, retriever, embedding_client)
    report = profiler.profile_dataset(dataset)
    profiler.save_report(report, "benchmark/system_report.json")

    health = SystemHealthReport(config, retriever)
    health_report = health.generate()
"""

from __future__ import annotations

import json
import logging
import re
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# DSA technical term dictionary for query complexity analysis
# ---------------------------------------------------------------------------

DSA_TECHNICAL_TERMS = {
    # Algorithms
    "dijkstra", "bellman-ford", "bfs", "dfs", "breadth-first", "depth-first",
    "kruskal", "prim", "a-star", "astar", "floyd-warshall", "floyd",
    "knapsack", "lis", "lcs", "ldp", "edit-distance", "viterbi",
    "quick-sort", "mergesort", "merge-sort", "heapsort", "heap-sort",
    "radix-sort", "counting-sort", "bucket-sort", "shell-sort",
    "binary-search", "ternary-search", "interpolation-search",
    "backtracking", "branch-bound", "branch-and-bound", "bruteforce",
    "brute-force", "greedy", "divide-and-conquer", "dynamic-programming",
    "memoization", "tabulation", "recursion",
    # Data structures
    "hash-table", "hashtable", "hashing", "hash-function", "collision",
    "binary-tree", "bst", "avl", "red-black", "b-tree", "bplus-tree",
    "heap", "min-heap", "max-heap", "d-heap", "fibonacci-heap", "pairing-heap",
    "linked-list", "doubly-linked", "singly-linked", "circular-list",
    "stack", "queue", "deque", "priority-queue", "circular-buffer",
    "trie", "prefix-tree", "suffix-tree", "suffix-array", "suffix-automaton",
    "segment-tree", "fenwick", "binary-indexed-tree", "sparse-table",
    "disjoint-set", "union-find", "dsu", "lca", "heavy-light-decomposition",
    "hld", "bloom-filter", "skip-list", "treap", "splay-tree",
    # Concepts
    "complexity", "big-o", "big-theta", "big-omega", "asymptotic",
    "time-complexity", "space-complexity", "amortized", "worst-case",
    "best-case", "average-case", "lower-bound", "upper-bound",
    "np-complete", "np-hard", "p-vs-np", "reduction",
    "invariant", "induction", "recurrence", "recursion-tree",
    "greedy", "optimal-substructure", "overlapping-subproblems",
    "topological-sort", "strongly-connected", "articulation", "bridge",
    "bipartite", "matching", "flow", "augmenting-path", "max-flow",
    "min-cut", "eulerian", "hamiltonian", "coloring",
    "divergence", "convergence", "series", "harmonic",
    "permutation", "combination", "binomial", "catalan",
    "modular", "congruence", "primitive-root", "euler-totient",
    "sieve", "primality", "factorization", "gcd", "lcm",
    # Operations
    "insert", "delete", "search", "lookup", "traverse", "iterate",
    "sort", "rank", "select", "min", "max", "extract", "decrease-key",
    "union", "find-set", "merge", "split", "join",
    # Properties
    "stable", "unstable", "in-place", "adaptive", "online", "offline",
    "deterministic", "randomized", "parallel", "distributed",
}


# ---------------------------------------------------------------------------
# Pipeline timing data
# ---------------------------------------------------------------------------

@dataclass
class StageTiming:
    """Timing data for a single pipeline stage."""
    name: str
    duration_seconds: float = 0.0
    timestamp_start: float = 0.0
    timestamp_end: float = 0.0


@dataclass
class FAISSSearchMetrics:
    """Metrics for a single FAISS index search."""
    level: str  # book, chapter, topic, subtopic, paragraph
    candidates_considered: int = 0
    items_selected: int = 0
    selection_ratio: float = 0.0
    avg_similarity: float = 0.0
    min_similarity: float = 0.0
    max_similarity: float = 0.0
    scoped_searches: int = 0
    timing: StageTiming = field(default_factory=lambda: StageTiming(name="faiss_search"))


@dataclass
class KneeDetectionMetrics:
    """Metrics for knee detection at a single level."""
    level: str
    strong_knee_detected: bool = False
    knee_rank: int = 0
    knee_threshold: float = 0.0
    fallback_threshold: float = 0.0
    selection_bounds_respected: bool = True
    timing: StageTiming = field(default_factory=lambda: StageTiming(name="knee_detection"))


@dataclass
class QueryComplexity:
    """Complexity metrics for a query."""
    word_count: int = 0
    technical_term_count: int = 0
    technical_term_density: float = 0.0
    concept_count: int = 0
    breadth: str = "MODERATE"  # BROAD, MODERATE, NARROW
    is_multi_concept: bool = False


@dataclass
class ContextBuildingMetrics:
    """Metrics for context building phase."""
    paragraphs_after_expansion: int = 0
    total_tokens: int = 0
    budget_utilization: float = 0.0
    unique_books: int = 0
    unique_sources: int = 0
    hierarchical_paragraphs: int = 0
    flat_paragraphs: int = 0
    expanded_topics: int = 0
    neighbor_expanded: int = 0


@dataclass
class RetrievalResultMetrics:
    """Complete metrics for a single retrieval operation."""
    query: str = ""
    query_complexity: QueryComplexity = field(default_factory=QueryComplexity)

    faiss_searches: Dict[str, FAISSSearchMetrics] = field(default_factory=dict)
    knee_detections: Dict[str, KneeDetectionMetrics] = field(default_factory=dict)
    context_building: ContextBuildingMetrics = field(default_factory=ContextBuildingMetrics)

    total_retrieval_time: float = 0.0
    stage_timings: List[StageTiming] = field(default_factory=list)

    # Overall quality signals
    avg_similarity_all_levels: float = 0.0
    min_similarity_all_levels: float = 0.0
    total_paragraphs: int = 0
    total_chapters: int = 0
    total_topics: int = 0
    total_books: int = 0


# ---------------------------------------------------------------------------
# Pipeline Profiler
# ---------------------------------------------------------------------------


class PipelineProfiler:
    """Profile the retrieval pipeline for latency and system metrics.

    Wraps the retriever to capture timing and metrics at each pipeline stage.

    Parameters
    ----------
    config : Config
        Validated configuration.
    retriever : KneeHierarchicalRetriever
        The knee-aware hierarchical retriever.
    embedding_client : EmbeddingClient
        Embedding client for query embedding.
    """

    def __init__(self, config: Any, retriever: Any, embedding_client: Any) -> None:
        self._config = config
        self._retriever = retriever
        self._embedding_client = embedding_client

    def profile_query(
        self,
        query: str,
        knee_enabled: bool = True,
    ) -> RetrievalResultMetrics:
        """Profile a single query through the retrieval pipeline.

        Parameters
        ----------
        query : str
            The user query.
        knee_enabled : bool
            Whether to use knee detection.

        Returns
        -------
        RetrievalResultMetrics
            Complete metrics for this retrieval operation.
        """
        metrics = RetrievalResultMetrics(query=query)
        overall_start = time.time()

        # 1. Query complexity analysis
        metrics.query_complexity = self._analyze_query_complexity(query)

        # 2. Embedding generation
        emb_start = time.time()
        result = self._retriever.retrieve(query, knee_enabled=knee_enabled)
        emb_end = time.time()
        metrics.stage_timings.append(StageTiming(
            name="embedding_generation",
            duration_seconds=emb_end - emb_start,
        ))

        # 3. Extract FAISS and knee metrics from the retrieval result
        self._extract_search_metrics(result, metrics)
        self._extract_knee_metrics(result, metrics)

        # 4. Context building metrics
        self._extract_context_metrics(result, metrics)

        # 5. Overall quality signals
        self._compute_quality_signals(metrics)

        metrics.total_retrieval_time = time.time() - overall_start

        return metrics

    def profile_dataset(
        self,
        dataset: List[Dict[str, str]],
        save_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Profile all queries in a dataset.

        Parameters
        ----------
        dataset : list[dict]
            Questions to profile.
        save_path : str or None
            Path to save the report.

        Returns
        -------
        dict
            Complete system profiling report.
        """
        all_metrics: List[RetrievalResultMetrics] = []
        latencies: List[float] = []

        for i, q in enumerate(dataset):
            query = q["question"]
            logger.info("Profiling query %d/%d: %s", i + 1, len(dataset), query[:60])

            metrics = self.profile_query(query)
            all_metrics.append(metrics)
            latencies.append(metrics.total_retrieval_time)

        # Compute aggregate statistics
        aggregate = self._compute_aggregate_stats(all_metrics, latencies)

        # Per-query breakdown
        per_query = []
        for m in all_metrics:
            per_query.append(self._metrics_to_dict(m))

        report = {
            "benchmark": "system_logging",
            "dataset_size": len(dataset),
            "aggregate": aggregate,
            "per_query": per_query,
        }

        if save_path:
            self._save_report(report, save_path)

        return report

    # ------------------------------------------------------------------
    # Internal analysis methods
    # ------------------------------------------------------------------

    def _analyze_query_complexity(self, query: str) -> QueryComplexity:
        """Analyze the complexity of a query.

        Parameters
        ----------
        query : str
            The user query.

        Returns
        -------
        QueryComplexity
            Complexity metrics for this query.
        """
        words = query.split()
        word_count = len(words)

        query_lower = query.lower()
        technical_terms = set()
        for term in DSA_TECHNICAL_TERMS:
            if term in query_lower:
                technical_terms.add(term)

        technical_term_count = len(technical_terms)
        technical_term_density = (
            technical_term_count / word_count if word_count > 0 else 0.0
        )

        # Count distinct concepts (technical terms that are multi-word or specific)
        concept_keywords = {
            "dijkstra", "bellman-ford", "bfs", "dfs", "kruskal", "prim",
            "knapsack", "lis", "lcs", "trie", "hash", "heap", "tree",
            "graph", "sort", "search", "dynamic", "programming", "greedy",
            "topological", "bipartite", "flow", "matching", "coloring",
            "complexity", "recurrence", "invariant", "amortized",
        }
        concept_count = sum(1 for c in concept_keywords if c in query_lower)

        # Multi-concept: query mentions concepts from different domains
        graph_terms = {"graph", "edge", "vertex", "path", "cycle", "tree",
                       "dijkstra", "bellman", "bfs", "dfs", "kruskal", "prim",
                       "topological", "bipartite", "flow", "matching"}
        ds_terms = {"array", "list", "stack", "queue", "hash", "trie",
                    "heap", "bst", "avl", "node", "pointer", "linked"}
        dp_terms = {"dynamic", "programming", "knapsack", "lis", "lcs",
                    "recurrence", "memoization", "tabulation", "state",
                    "transition", "subproblem"}
        complexity_terms = {"complexity", "big-o", "asymptotic", "recurrence",
                            "master", "theorem", "lower-bound", "upper-bound",
                            "amortized", "worst-case", "best-case"}

        domains_hit = sum(1 for domain in [
            graph_terms, ds_terms, dp_terms, complexity_terms
        ] if any(t in query_lower for t in domain))

        return QueryComplexity(
            word_count=word_count,
            technical_term_count=technical_term_count,
            technical_term_density=round(technical_term_density, 4),
            concept_count=concept_count,
            breadth="BROAD" if word_count > 30 or domains_hit >= 3 else
                     "NARROW" if concept_count <= 1 and word_count < 15 else
                     "MODERATE",
            is_multi_concept=domains_hit >= 2,
        )

    def _extract_search_metrics(
        self,
        result: Any,
        metrics: RetrievalResultMetrics,
    ) -> None:
        """Extract FAISS search metrics from a RetrievalResult.

        Parameters
        ----------
        result : RetrievalResult
            The retrieval result.
        metrics : RetrievalResultMetrics
            Metrics object to populate.
        """
        # Count items at each level
        books = getattr(result, "books", []) or []
        chapters = getattr(result, "chapters", []) or []
        topics = getattr(result, "topics", []) or []
        paragraphs = getattr(result, "paragraphs", []) or []

        metrics.total_books = len(books)
        metrics.total_chapters = len(chapters)
        metrics.total_topics = len(topics)
        metrics.total_paragraphs = len(paragraphs)

        # Extract similarity scores from paragraphs if available
        similarities = []
        for para in paragraphs:
            score = getattr(para, "score", None)
            if score is not None:
                similarities.append(float(score))

        if similarities:
            metrics.avg_similarity_all_levels = round(
                statistics.mean(similarities), 4
            )
            metrics.min_similarity_all_levels = round(min(similarities), 4)

    def _extract_knee_metrics(
        self,
        result: Any,
        metrics: RetrievalResultMetrics,
    ) -> None:
        """Extract knee detection metrics from a RetrievalResult.

        Parameters
        ----------
        result : RetrievalResult
            The retrieval result.
        metrics : RetrievalResultMetrics
            Metrics object to populate.
        """
        knees = getattr(result, "knees", None)
        if knees is None:
            knee = getattr(result, "knee", None)
            if knee is not None:
                knees = {"paragraph": knee}

        if knees:
            for level_name, knee_data in knees.items():
                knee_metrics = KneeDetectionMetrics(level=level_name)

                # Extract knee properties
                candidate_k = getattr(knee_data, "candidate_k", 0)
                selected_k = getattr(knee_data, "selected_k", 0)
                knee_rank = getattr(knee_data, "knee_index", 0)
                threshold = getattr(knee_data, "threshold", 0.0)

                knee_metrics.knee_rank = knee_rank
                knee_metrics.knee_threshold = float(threshold)
                knee_metrics.selection_ratio = (
                    selected_k / candidate_k if candidate_k > 0 else 0.0
                )

                # Strong knee: knee rank is significantly before the end
                knee_metrics.strong_knee_detected = (
                    knee_rank > 0 and knee_rank < candidate_k * 0.7
                )

                metrics.knee_detections[level_name] = knee_metrics

    def _extract_context_metrics(
        self,
        result: Any,
        metrics: RetrievalResultMetrics,
    ) -> None:
        """Extract context building metrics from a RetrievalResult.

        Parameters
        ----------
        result : RetrievalResult
            The retrieval result.
        metrics : RetrievalResultMetrics
            Metrics object to populate.
        """
        ctx = metrics.context_building
        ctx.paragraphs_after_expansion = len(getattr(result, "paragraphs", []) or [])
        ctx.total_tokens = getattr(result, "context_tokens", 0)

        # Budget utilization
        max_tokens = getattr(self._config, "retrieval", None)
        if max_tokens is not None:
            max_ctx = getattr(max_tokens, "max_context_tokens", 20000)
        else:
            max_ctx = 20000
        ctx.budget_utilization = (
            round(ctx.total_tokens / max_ctx, 4) if max_ctx > 0 else 0.0
        )

        # Source diversity
        paragraphs = getattr(result, "paragraphs", []) or []
        books_set = set()
        sources_set = set()
        for para in paragraphs:
            book_id = getattr(para, "book_id", None)
            if book_id:
                books_set.add(book_id)
            source = getattr(para, "source_file", None)
            if source:
                sources_set.add(source)

        ctx.unique_books = len(books_set)
        ctx.unique_sources = len(sources_set)

        # Expanded topics
        expanded = getattr(result, "expanded_topics", []) or []
        ctx.expanded_topics = len(expanded)

    def _compute_quality_signals(self, metrics: RetrievalResultMetrics) -> None:
        """Compute overall quality signals.

        Parameters
        ----------
        metrics : RetrievalResultMetrics
            Metrics object to update.
        """
        # Compute average similarity across all knee detections
        all_sims = []
        for kd in metrics.knee_detections.values():
            if kd.knee_threshold > 0:
                all_sims.append(kd.knee_threshold)

        if all_sims:
            metrics.avg_similarity_all_levels = round(
                statistics.mean(all_sims), 4
            )

    # ------------------------------------------------------------------
    # Aggregate statistics
    # ------------------------------------------------------------------

    def _compute_aggregate_stats(
        self,
        all_metrics: List[RetrievalResultMetrics],
        latencies: List[float],
    ) -> Dict[str, Any]:
        """Compute aggregate statistics from all query metrics.

        Parameters
        ----------
        all_metrics : list[RetrievalResultMetrics]
            Per-query metrics.
        latencies : list[float]
            Total retrieval latencies.

        Returns
        -------
        dict
            Aggregate statistics organized by category.
        """
        return {
            "latency": self._stats_latency(latencies),
            "query_complexity": self._stats_query_complexity(all_metrics),
            "faiss_performance": self._stats_faiss_performance(all_metrics),
            "knee_detection": self._stats_knee_detection(all_metrics),
            "context_building": self._stats_context_building(all_metrics),
            "retrieval_quality": self._stats_retrieval_quality(all_metrics),
            "system_health": self._get_system_health(),
        }

    @staticmethod
    def _stats_latency(latencies: List[float]) -> Dict[str, Any]:
        """Compute latency statistics.

        Parameters
        ----------
        latencies : list[float]
            Total retrieval latencies in seconds.

        Returns
        -------
        dict
            Latency statistics (mean, p50, p90, p99, max, min).
        """
        if not latencies:
            return {}

        sorted_lat = sorted(latencies)
        n = len(sorted_lat)

        return {
            "mean_seconds": round(statistics.mean(latencies), 4),
            "median_seconds": round(statistics.median(latencies), 4),
            "std_seconds": round(statistics.stdev(latencies), 4) if n > 1 else 0.0,
            "min_seconds": round(min(latencies), 4),
            "max_seconds": round(max(latencies), 4),
            "p50_seconds": round(sorted_lat[n // 2], 4),
            "p90_seconds": round(sorted_lat[int(n * 0.9)], 4),
            "p99_seconds": round(sorted_lat[int(n * 0.99)], 4) if n > 10 else round(max(latencies), 4),
            "total_queries": n,
        }

    @staticmethod
    def _stats_query_complexity(
        all_metrics: List[RetrievalResultMetrics],
    ) -> Dict[str, Any]:
        """Compute query complexity statistics.

        Parameters
        ----------
        all_metrics : list[RetrievalResultMetrics]
            Per-query metrics.

        Returns
        -------
        dict
            Query complexity statistics.
        """
        word_counts = [m.query_complexity.word_count for m in all_metrics]
        tech_densities = [m.query_complexity.technical_term_density for m in all_metrics]
        concept_counts = [m.query_complexity.concept_count for m in all_metrics]
        breadth_counts = {"BROAD": 0, "MODERATE": 0, "NARROW": 0}
        multi_concept_count = 0

        for m in all_metrics:
            breadth_counts[m.query_complexity.breadth] = (
                breadth_counts.get(m.query_complexity.breadth, 0) + 1
            )
            if m.query_complexity.is_multi_concept:
                multi_concept_count += 1

        n = len(all_metrics)
        return {
            "avg_word_count": round(statistics.mean(word_counts), 2) if word_counts else 0,
            "avg_tech_term_density": round(statistics.mean(tech_densities), 4) if tech_densities else 0,
            "avg_concept_count": round(statistics.mean(concept_counts), 2) if concept_counts else 0,
            "breadth_distribution": breadth_counts,
            "multi_concept_queries": multi_concept_count,
            "multi_concept_pct": round(multi_concept_count / n * 100, 1) if n > 0 else 0,
        }

    @staticmethod
    def _stats_faiss_performance(
        all_metrics: List[RetrievalResultMetrics],
    ) -> Dict[str, Any]:
        """Compute FAISS performance statistics.

        Parameters
        ----------
        all_metrics : list[RetrievalResultMetrics]
            Per-query metrics.

        Returns
        -------
        dict
            FAISS performance statistics per level.
        """
        level_stats: Dict[str, Dict[str, List[float]]] = {}

        for m in all_metrics:
            for level, search in m.faiss_searches.items():
                if level not in level_stats:
                    level_stats[level] = {
                        "candidates": [],
                        "selected": [],
                        "ratios": [],
                    }
                level_stats[level]["candidates"].append(search.candidates_considered)
                level_stats[level]["selected"].append(search.items_selected)
                level_stats[level]["ratios"].append(search.selection_ratio)

        result = {}
        for level, stats in level_stats.items():
            result[level] = {
                "avg_candidates": round(statistics.mean(stats["candidates"]), 1) if stats["candidates"] else 0,
                "avg_selected": round(statistics.mean(stats["selected"]), 1) if stats["selected"] else 0,
                "avg_selection_ratio": round(statistics.mean(stats["ratios"]), 4) if stats["ratios"] else 0,
            }

        return result

    @staticmethod
    def _stats_knee_detection(
        all_metrics: List[RetrievalResultMetrics],
    ) -> Dict[str, Any]:
        """Compute knee detection statistics.

        Parameters
        ----------
        all_metrics : list[RetrievalResultMetrics]
            Per-query metrics.

        Returns
        -------
        dict
            Knee detection statistics.
        """
        strong_knee_counts: Dict[str, int] = {}
        knee_ranks: Dict[str, List[int]] = {}
        total_queries = len(all_metrics)

        for m in all_metrics:
            for level, kd in m.knee_detections.items():
                strong_knee_counts[level] = (
                    strong_knee_counts.get(level, 0) + (1 if kd.strong_knee_detected else 0)
                )
                if level not in knee_ranks:
                    knee_ranks[level] = []
                knee_ranks[level].append(kd.knee_rank)

        result = {}
        for level in strong_knee_counts:
            result[level] = {
                "strong_knee_count": strong_knee_counts[level],
                "strong_knee_pct": round(
                    strong_knee_counts[level] / total_queries * 100, 1
                ) if total_queries > 0 else 0,
                "avg_knee_rank": round(
                    statistics.mean(knee_ranks[level]), 1
                ) if knee_ranks.get(level) else 0,
            }

        return result

    @staticmethod
    def _stats_context_building(
        all_metrics: List[RetrievalResultMetrics],
    ) -> Dict[str, Any]:
        """Compute context building statistics.

        Parameters
        ----------
        all_metrics : list[RetrievalResultMetrics]
            Per-query metrics.

        Returns
        -------
        dict
            Context building statistics.
        """
        paras = [m.context_building.paragraphs_after_expansion for m in all_metrics]
        tokens = [m.context_building.total_tokens for m in all_metrics]
        utilizations = [m.context_building.budget_utilization for m in all_metrics]
        books = [m.context_building.unique_books for m in all_metrics]
        sources = [m.context_building.unique_sources for m in all_metrics]

        return {
            "avg_paragraphs": round(statistics.mean(paras), 1) if paras else 0,
            "avg_tokens": round(statistics.mean(tokens), 1) if tokens else 0,
            "avg_budget_utilization": round(statistics.mean(utilizations), 4) if utilizations else 0,
            "avg_unique_books": round(statistics.mean(books), 1) if books else 0,
            "avg_unique_sources": round(statistics.mean(sources), 1) if sources else 0,
        }

    @staticmethod
    def _stats_retrieval_quality(
        all_metrics: List[RetrievalResultMetrics],
    ) -> Dict[str, Any]:
        """Compute retrieval quality statistics.

        Parameters
        ----------
        all_metrics : list[RetrievalResultMetrics]
            Per-query metrics.

        Returns
        -------
        dict
            Retrieval quality statistics.
        """
        avg_sims = [m.avg_similarity_all_levels for m in all_metrics]
        min_sims = [m.min_similarity_all_levels for m in all_metrics]
        total_paras = [m.total_paragraphs for m in all_metrics]
        total_topics = [m.total_topics for m in all_metrics]
        total_chapters = [m.total_chapters for m in all_metrics]
        total_books = [m.total_books for m in all_metrics]

        return {
            "avg_similarity": round(statistics.mean(avg_sims), 4) if avg_sims else 0,
            "avg_min_similarity": round(statistics.mean(min_sims), 4) if min_sims else 0,
            "avg_paragraphs": round(statistics.mean(total_paras), 1) if total_paras else 0,
            "avg_topics": round(statistics.mean(total_topics), 1) if total_topics else 0,
            "avg_chapters": round(statistics.mean(total_chapters), 1) if total_chapters else 0,
            "avg_books": round(statistics.mean(total_books), 1) if total_books else 0,
        }

    def _get_system_health(self) -> Dict[str, Any]:
        """Get system health information.

        Returns
        -------
        dict
            System health metrics (index sizes, KB stats, config).
        """
        manager = getattr(self._retriever, "manager", None)
        if manager is None:
            return {"status": "no_manager"}

        health: Dict[str, Any] = {
            "status": "healthy",
            "index_sizes": {},
            "knowledge_base": {},
            "config": {},
        }

        # Index sizes
        for level in ["books", "chapters", "topics", "subtopics", "paragraphs"]:
            index_attr = f"_{level}_index"
            index = getattr(manager, index_attr, None)
            if index is not None:
                dim = getattr(index, "d", 0)
                n = getattr(index, "ntotal", 0)
                health["index_sizes"][level] = {
                    "dimension": dim,
                    "num_vectors": n,
                }

        # Knowledge base stats
        kb = getattr(manager, "knowledge_base", None)
        if kb is not None:
            health["knowledge_base"] = {
                "total_books": len(getattr(kb, "books", {})),
                "total_chapters": len(getattr(kb, "chapters", {})),
                "total_topics": len(getattr(kb, "topics", {})),
                "total_subtopics": len(getattr(kb, "subtopics", {})),
                "total_paragraphs": len(getattr(kb, "paragraphs", {})),
            }

        # Config
        health["config"] = {
            "max_context_tokens": getattr(
                getattr(self._config, "retrieval", None),
                "max_context_tokens", 20000
            ),
            "similarity_threshold": getattr(
                getattr(self._config, "retrieval", None),
                "similarity_threshold", 0.15
            ),
            "neighbor_window": getattr(
                getattr(self._config, "retrieval", None),
                "neighbor_window", 2
            ),
            "max_tool_calls": getattr(
                getattr(self._config, "agentic_retrieval", None),
                "max_tool_calls", 3
            ),
        }

        return health

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def _metrics_to_dict(self, metrics: RetrievalResultMetrics) -> Dict[str, Any]:
        """Convert RetrievalResultMetrics to a serializable dict.

        Parameters
        ----------
        metrics : RetrievalResultMetrics
            Metrics to serialize.

        Returns
        -------
        dict
            Serializable dict.
        """
        return {
            "query": metrics.query,
            "query_complexity": {
                "word_count": metrics.query_complexity.word_count,
                "technical_term_count": metrics.query_complexity.technical_term_count,
                "technical_term_density": metrics.query_complexity.technical_term_density,
                "concept_count": metrics.query_complexity.concept_count,
                "breadth": metrics.query_complexity.breadth,
                "is_multi_concept": metrics.query_complexity.is_multi_concept,
            },
            "total_retrieval_time_seconds": round(metrics.total_retrieval_time, 4),
            "faiss_searches": {
                level: {
                    "candidates": s.candidates_considered,
                    "selected": s.items_selected,
                    "selection_ratio": s.selection_ratio,
                    "avg_similarity": s.avg_similarity,
                }
                for level, s in metrics.faiss_searches.items()
            },
            "knee_detections": {
                level: {
                    "strong_knee": kd.strong_knee_detected,
                    "knee_rank": kd.knee_rank,
                    "threshold": kd.knee_threshold,
                    "selection_ratio": kd.selection_ratio,
                }
                for level, kd in metrics.knee_detections.items()
            },
            "context_building": {
                "paragraphs": metrics.context_building.paragraphs_after_expansion,
                "tokens": metrics.context_building.total_tokens,
                "budget_utilization": metrics.context_building.budget_utilization,
                "unique_books": metrics.context_building.unique_books,
                "unique_sources": metrics.context_building.unique_sources,
                "expanded_topics": metrics.context_building.expanded_topics,
            },
            "quality_signals": {
                "avg_similarity": metrics.avg_similarity_all_levels,
                "min_similarity": metrics.min_similarity_all_levels,
                "total_paragraphs": metrics.total_paragraphs,
                "total_topics": metrics.total_topics,
                "total_chapters": metrics.total_chapters,
                "total_books": metrics.total_books,
            },
        }

    @staticmethod
    def _save_report(report: Dict[str, Any], path: str) -> None:
        """Save the profiling report to a JSON file.

        Parameters
        ----------
        report : dict
            The profiling report.
        path : str
            Output file path.
        """
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        with open(path_obj, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, ensure_ascii=False, default=str)
        logger.info("Saved system report to %s", path)


# ---------------------------------------------------------------------------
# System Health Report
# ---------------------------------------------------------------------------


class SystemHealthReport:
    """Generate a comprehensive system health report.

    Answers the questions a systems designer asks about the retrieval system:
        1. What is the index structure and size?
        2. How many items are at each hierarchy level?
        3. What are the configuration parameters?
        4. Are there any capacity concerns?
        5. What is the embedding configuration?
    """

    def __init__(self, config: Any, retriever: Any) -> None:
        self._config = config
        self._retriever = retriever

    def generate(self) -> Dict[str, Any]:
        """Generate the system health report.

        Returns
        -------
        dict
            Complete system health report.
        """
        manager = getattr(self._retriever, "manager", None)

        report = {
            "component": "system_health",
            "index_structure": self._get_index_structure(manager),
            "knowledge_base_stats": self._get_kb_stats(manager),
            "configuration": self._get_config_summary(),
            "capacity_analysis": self._analyze_capacity(manager),
            "recommendations": self._generate_recommendations(manager),
        }

        return report

    def _get_index_structure(self, manager: Any) -> Dict[str, Any]:
        """Get the index structure details.

        Parameters
        ----------
        manager : MultiIndexManager
            The index manager.

        Returns
        -------
        dict
            Index structure per level.
        """
        structure = {}
        if manager is None:
            return {"error": "no manager"}

        for level in ["books", "chapters", "topics", "subtopics", "paragraphs"]:
            index_attr = f"_{level}_index"
            index = getattr(manager, index_attr, None)
            if index is not None:
                structure[level] = {
                    "dimension": getattr(index, "d", 0),
                    "num_vectors": getattr(index, "ntotal", 0),
                    "index_type": type(index).__name__,
                }

        return structure

    def _get_kb_stats(self, manager: Any) -> Dict[str, Any]:
        """Get knowledge base statistics.

        Parameters
        ----------
        manager : MultiIndexManager
            The index manager.

        Returns
        -------
        dict
            KB statistics.
        """
        if manager is None:
            return {"error": "no manager"}

        stats = {}
        for level in ["books", "chapters", "topics", "subtopics", "paragraphs"]:
            attr = f"{level}"
            collection = getattr(manager, attr, {})
            if isinstance(collection, dict):
                stats[f"total_{level}"] = len(collection)

        return stats

    def _get_config_summary(self) -> Dict[str, Any]:
        """Get configuration summary.

        Returns
        -------
        dict
            Key configuration parameters.
        """
        retrieval = getattr(self._config, "retrieval", None)
        agentic = getattr(self._config, "agentic_retrieval", None)
        embeddings = getattr(self._config, "embeddings", None)

        return {
            "retrieval": {
                "max_context_tokens": getattr(retrieval, "max_context_tokens", 20000) if retrieval else 20000,
                "similarity_threshold": getattr(retrieval, "similarity_threshold", 0.15) if retrieval else 0.15,
                "neighbor_window": getattr(retrieval, "neighbor_window", 2) if retrieval else 2,
            },
            "agentic": {
                "enabled": getattr(agentic, "enabled", True) if agentic else True,
                "max_tool_calls": getattr(agentic, "max_tool_calls", 3) if agentic else 3,
            },
            "embeddings": {
                "model": getattr(embeddings, "model", "TBD") if embeddings else "TBD",
                "endpoint": getattr(embeddings, "endpoint", None) if embeddings else None,
            },
        }

    def _analyze_capacity(self, manager: Any) -> Dict[str, Any]:
        """Analyze capacity and identify potential concerns.

        Parameters
        ----------
        manager : MultiIndexManager
            The index manager.

        Returns
        -------
        dict
            Capacity analysis per level.
        """
        analysis = {}
        if manager is None:
            return {"error": "no manager"}

        retrieval = getattr(self._config, "retrieval", None)
        max_tokens = getattr(retrieval, "max_context_tokens", 20000) if retrieval else 20000

        for level in ["paragraphs", "subtopics", "topics", "chapters", "books"]:
            attr = f"{level}"
            collection = getattr(manager, attr, {})
            count = len(collection) if isinstance(collection, dict) else 0
            analysis[level] = {
                "count": count,
                "capacity_status": "ok",
            }

        # Check if paragraph count is reasonable
        para_count = analysis.get("paragraphs", {}).get("count", 0)
        if para_count > 50000:
            analysis["paragraphs"]["capacity_status"] = "large"
            analysis["paragraphs"]["note"] = (
                "Large corpus (>50K paragraphs). Consider partitioning or "
                "using HNSW index for faster search."
            )
        elif para_count < 1000:
            analysis["paragraphs"]["capacity_status"] = "small"
            analysis["paragraphs"]["note"] = (
                "Small corpus (<1K paragraphs). Current setup is appropriate."
            )

        # Check context budget vs corpus size
        if para_count > 0 and max_tokens < 5000:
            analysis["_system"] = "warning"
            analysis["_note"] = (
                f"Context budget ({max_tokens} tokens) may be tight for "
                f"corpus of {para_count} paragraphs."
            )

        return analysis

    def _generate_recommendations(self, manager: Any) -> List[str]:
        """Generate recommendations based on system analysis.

        Parameters
        ----------
        manager : MultiIndexManager
            The index manager.

        Returns
        -------
        list[str]
            List of recommendations.
        """
        recommendations = []

        if manager is None:
            recommendations.append("No index manager found. Build index before benchmarking.")
            return recommendations

        kb_stats = self._get_kb_stats(manager)
        para_count = kb_stats.get("total_paragraphs", 0)

        # Corpus size recommendations
        if para_count > 50000:
            recommendations.append(
                "Consider using HNSW index instead of IndexFlatIP for "
                "faster approximate nearest neighbor search on large corpora."
            )

        # Embedding model recommendations
        embeddings = getattr(self._config, "embeddings", None)
        model = getattr(embeddings, "model", "TBD") if embeddings else "TBD"
        if model == "TBD":
            recommendations.append(
                "Embedding model is not configured (TBD). System falls back "
                "to sentence-transformers or TF-IDF. Consider configuring "
                "a dedicated embedding endpoint for better quality."
            )

        # Context budget recommendations
        retrieval = getattr(self._config, "retrieval", None)
        max_tokens = getattr(retrieval, "max_context_tokens", 20000) if retrieval else 20000
        if max_tokens > 32000:
            recommendations.append(
                "Context budget is very large (>32K tokens). This may cause "
                "LLM context window issues. Consider reducing to 16K-24K."
            )
        elif max_tokens < 4096:
            recommendations.append(
                "Context budget is small (<4K tokens). May not provide enough "
                "context for complex DSA questions. Consider increasing to 8K-16K."
            )

        if not recommendations:
            recommendations.append("System configuration appears healthy. No critical recommendations.")

        return recommendations
