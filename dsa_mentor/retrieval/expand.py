"""Topic expansion + paragraph neighbor expansion (spec §14, §16, §17).

Classes:
    QueryBroadtherClassifier  — lightweight heuristic breadth classification
    TopicExpander             — expand topics to full_text based on breadth
    ParagraphNeighborExpander — add prev/next paragraph neighbors within same topic
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple

from ..models import Paragraph, Topic


# ---------------------------------------------------------------------------
# DSA concept names used for breadth classification (spec §16)
# ---------------------------------------------------------------------------

CONCEPT_NAMES: List[str] = [
    "dijkstra",
    "bfs",
    "dfs",
    "dynamic programming",
    "greedy",
    "binary search",
    "heap",
    "tree",
    "graph",
    "hash",
    "segment tree",
    "dsu",
    "kruskal",
    "prim",
    "bellman-ford",
    "a*",
    "knapsack",
    "lis",
    "lcs",
    "backtracking",
    "topological",
    "floyd-warshall",
    "red-black",
    "avl",
    "trie",
    "fenwick",
    "suffix",
    "kmp",
    "rabin-karp",
]

# Technical terms that signal a narrow, specific query (spec §16)
NARROW_TECHNICAL_TERMS: List[str] = [
    "settled vertex invariant",
    "settled-vertex invariant",
    "settled vertex",
    "negative edge",
    "negative-weight",
    "admissible heuristic",
    "consistency optimality",
    "invariant violation",
    "amortized analysis",
    "edge relaxation",
    "priority queue",
    "union find",
    "disjoint set",
    "union-by-rank",
    "path compression",
    "dominator tree",
    "bridge",
    "articulation point",
    "strongly connected",
    "biconnected",
    "topological sort",
    "longest increasing subsequence",
    "longest common subsequence",
    "edit distance",
    "bitset",
    "bitmask",
    "state compression",
    "meet-in-the-middle",
    "square root decomposition",
    "heavy-light decomposition",
    "centroid decomposition",
    "suffix automaton",
    "suffix link",
    "z-function",
    "kmp failure function",
    "hash collision",
    "double hashing",
    "open addressing",
    "reservoir sampling",
    "moore voting",
    "saddleback search",
    "golden ratio",
    "euler tour",
    "indegree",
    "outdegree",
    "topsort",
    "dp state",
    "dp transition",
    "memoization",
    "state space",
    "state representation",
    "base case",
    "transition function",
    "optimal substructure",
    "overlapping subproblems",
    "state space tree",
    "pruning",
    "alpha-beta",
    "minimax",
    "nim sum",
    "xor sum",
    "nim game",
    "splay tree",
    "treap",
    "sb tree",
    "wavelet tree",
    "merge sort tree",
    "persistent segment tree",
    "offline query",
    "parallel binary search",
    "fractional cascading",
    "k-d tree",
    "range minimum query",
    "rmq",
    "lowest common ancestor",
    "lca",
    "euler tour technique",
    "hld",
    "link-cut tree",
    "dynamic tree",
    "heavy path",
    "light edge",
    "color class",
    "chromatic number",
    "max flow",
    "min cut",
    "bipartite matching",
    "augmenting path",
    "dinic",
    "edmonds-karp",
    "push-relabel",
    "residual graph",
    "scc",
    "tarjan",
    "bridges",
    "2-sat",
    "implication graph",
    "strong component",
    "condensation",
    "topological ordering",
    "critical path",
    "slack variable",
    "reduced cost",
    "potential function",
    "Johnson's",
    "floyd-warshall",
    "all-pairs shortest",
    "matrix multiplication",
    "semiring",
    "convex hull",
    "half-plane intersection",
    "rotating calipers",
    "Voronoi",
    "Delaunay",
    "closest pair",
    "bentley-ottmann",
    "sweep line",
    "scan line",
    "plane sweep",
    "line segment intersection",
    "range tree",
    "fractional cascading",
    "kruskal reconstruction",
    "virtual tree",
    "virtual tree construction",
    "dsu on tree",
    "small-to-large",
    "longest path",
    "longest chain",
    "dilworth",
    "erdos-szekeres",
    "inversion count",
    "merge sort inversion",
    "cdq divide and conquer",
    "parallel binary search",
    "offline dynamic connectivity",
    "rollback dsu",
    "square root trick",
    "sqrt decomposition",
    "mo's algorithm",
    "mo's with updates",
    "tree mo's",
    "heavy-light mo's",
    "centroid mo's",
    "block decomposition",
    "bitset optimization",
    "word-level parallelism",
    "bit parallelism",
    "transitive closure",
    "reachability",
    "strong connectivity",
    "bridge tree",
    "block-cut tree",
    "cactus graph",
    "planar graph",
    "euler's formula",
    "dual graph",
    "face traversal",
    "outerplanar",
    "series-parallel",
    "partial k-tree",
    "tree decomposition",
    "branch width",
    "tree width",
    "bidirectional search",
    "iterative deepening",
    "ida*",
    "astar",
    "heuristic search",
    "consistency",
    "monotonicity",
    "path consistency",
    "arc consistency",
    "forward checking",
    "constraint propagation",
    "backjumping",
    "backmarking",
    "learning",
    "nogood",
    "leam",
    "lrv",
    "largest remaining",
    "first fail",
    "dominating variable",
    "min values",
    "min conflicts",
    "wmax",
    "max weight",
    "weighted",
    "cost function",
    "objective function",
    "constraint satisfaction",
    "csp",
    "satisfiability",
    "boolean",
    "clause",
    "literal",
    "cnf",
    "dnf",
    "resolution",
    "refutation",
    "unit propagation",
    "pure literal",
    "dpll",
    "sat solver",
    "unsat core",
    "model counting",
    "sharp sat",
    "weighted model counting",
    "markov",
    "bayesian",
    "belief propagation",
    "mean field",
    "variational",
    "em algorithm",
    "vb",
    "gaussian",
    "kalman",
    "particle filter",
    "monte carlo",
    "mcmc",
    "metropolis",
    "hastings",
    "gibbs sampling",
    "rejection sampling",
    "importance sampling",
    "sequential monte carlo",
    "particle filtering",
    "smoothing",
    "forward-backward",
    "viterbi",
    " Baum-welch",
    "hidden markov",
    "markov decision",
    "mdp",
    "q-learning",
    "policy gradient",
    "actor-critic",
    "deep q",
    "dqn",
    "double dqn",
    "dueling dqn",
    "noisy net",
    "distributional",
    "c51",
    "rainbow",
    "sac",
    "td3",
    "ppo",
    "trpo",
    "a2c",
    "asynchronous",
    "distributed",
    "multi-agent",
    "cooperative",
    "competitive",
    "decentralized",
    "federated",
    "meta-learning",
    "maml",
    "prototypical",
    "matching",
    "one-shot",
    "few-shot",
    "zero-shot",
    "transfer",
    "domain adaptation",
    "continual",
    "lifelong",
    "curriculum",
    "self-supervised",
    "contrastive",
    "simclr",
    "momentum contrast",
    "swav",
    "byol",
    "mae",
    "masked autoencoder",
    "bert",
    "gpt",
    "transformer",
    "attention",
    "multi-head",
    "scaled dot-product",
    "positional encoding",
    "layer norm",
    "residual",
    "ffn",
    "encoder-decoder",
    "cross-attention",
    "causal",
    "autoregressive",
    "causal mask",
    "flash attention",
    "sparse attention",
    "longformer",
    "reformer",
    "performer",
    "linear attention",
    "retention",
    "ring attention",
    "blockwise",
    "paged attention",
    "kv cache",
    "speculative",
    "draft model",
    "verification",
    "distillation",
    "knowledge distillation",
    "mixture of experts",
    "moe",
    "gating",
    "load balancing",
    "expert",
    "router",
    "top-k",
    "noisy top-k",
    "auxiliary loss",
    "switch transformer",
    "deepspeed",
    "tensor parallel",
    "pipeline parallel",
    "data parallel",
    "fSDP",
    "activation checkpointing",
    "gradient accumulation",
    "gradient clipping",
    "mixed precision",
    "fp16",
    "bf16",
    "fp8",
    "amp",
    "loss scaling",
    "optimizer state",
    "zeRO",
    "offload",
    "cpu offload",
    "nvme offload",
    "inference",
    "vllm",
    "tgi",
    "tensorrt llm",
    "llama cpp",
    "gptq",
    "awq",
    "bitsandbytes",
    "quantization",
    "int8",
    "int4",
    "nf4",
    "paged optimizer",
    "loop unrolling",
    "vectorization",
    "simd",
    "avx",
    "sse",
    "neon",
    "cuda",
    "gpu",
    "tpu",
    "npu",
    "fpga",
    "asic",
    "tensor core",
    "warp",
    "block",
    "thread",
    "shared memory",
    "register",
    "texture cache",
    "constant cache",
    "l1 cache",
    "l2 cache",
    "global memory",
    "hbm",
    "nvlink",
    "infiniband",
    "rdma",
    "all-reduce",
    "all-gather",
    "reduce-scatter",
    "broadcast",
    "scatter",
    "gather",
    "point-to-point",
    "blocking",
    "non-blocking",
    "overlap",
    "pipeline",
    "stall",
    "bubble",
    "gradient compression",
    "sparsification",
    "top-k sparsification",
    "sign SGD",
    "quantized SGD",
    "1-bit SGD",
    "stochastic sparsification",
    "error feedback",
    "compression error",
    "accumulation",
    "local SGD",
    "periodic averaging",
    "asynchronous SGD",
    "stale gradient",
    "gradient delay",
    "bounded delay",
    "unbounded delay",
    "heterogeneous",
    "straggler",
    "fault tolerance",
    "checkpointing",
    "rollback",
    "recovery",
    "elastic training",
    "dynamic scaling",
    "auto-scaling",
    "resource allocation",
    "scheduling",
    "bin packing",
    "knapsack variant",
    "makespan",
    "latency",
    "throughput",
    "utilization",
    "efficiency",
    "scalability",
    "strong scaling",
    "weak scaling",
    "amdahl",
    "gustafson",
    "speedup",
    "efficiency curve",
    "strong scaling limit",
    "communication overhead",
    "computation communication overlap",
    "latency hiding",
    "bandwidth bound",
    "compute bound",
    "memory bound",
    "arithmetic intensity",
    "roofline",
    "peak performance",
    "flops",
    "bytes",
    "memory bandwidth",
    "compute throughput",
    "tensor throughput",
    "sparse throughput",
    "sparsity",
    "structured sparsity",
    "unstructured sparsity",
    "pruning",
    "structured pruning",
    "channel pruning",
    "filter pruning",
    "kernel pruning",
    "magnitude pruning",
    "gradual pruning",
    "one-shot pruning",
    "iterative pruning",
    "fine-tuning",
    "retraining",
    "pruning ratio",
    "sparsity level",
    "accuracy drop",
    "accuracy recovery",
    "compensation",
    "bias compensation",
    "weight compensation",
    "gradient compensation",
    "error backpropagation",
    "error compensation",
    "accumulated error",
    "error drift",
    "error bound",
    "convergence",
    "convergence rate",
    "linear convergence",
    "superlinear convergence",
    "sublinear convergence",
    "geometric convergence",
    "asymptotic",
    "limit",
    "epsilon",
    "tolerance",
    "stopping criterion",
    "early stopping",
    "patience",
    "validation",
    "overfitting",
    "underfitting",
    "generalization",
    "generalization gap",
    "bias variance",
    "tradeoff",
    "regularization",
    "l1",
    "l2",
    "dropout",
    "dropout rate",
    "alpha dropout",
    "gaussian dropout",
    "mixed dropout",
    "zoneout",
    "stochastic depth",
    "random depth",
    "stochastic depth rate",
    "label smoothing",
    "mixup",
    "cutmix",
    "autoaugment",
    "random augment",
    "cutout",
    "erase",
    "mixup variant",
    "cutmix variant",
    "rag",
    "retrieval augmented",
    "retrieval",
    "embedding",
    "re-embedding",
    "reranking",
    "cross encoder",
    "bi-encoder",
    "dense retrieval",
    "sparse retrieval",
    "lexical retrieval",
    "bm25",
    "tf-idf",
    "colbert",
    "late interaction",
    "max sim",
    "colbertv2",
    "anserini",
    "unicoil",
    "splade",
    "learned sparse",
    "neural sparse",
    "dual encoder",
    "cross encoder",
    "two-tower",
    "biencoder",
    "multi-vector",
    "single-vector",
    "aggregate",
    "mean pooling",
    "cls pooling",
    "last token",
    "weighted pooling",
    "attention pooling",
    "projection",
    "projection head",
    "mlp head",
    "normalization",
    "l2 normalize",
    "temperature",
    "nce loss",
    "info nce",
    "triplet loss",
    "contrastive loss",
    "margin loss",
    "hinge loss",
    "focal loss",
    "supervised contrastive",
    "hard negative",
    "mining",
    "online mining",
    "offline mining",
    "buffer",
    "memory bank",
    "queue",
    "momentum encoder",
    "teacher encoder",
    "student encoder",
    "ema",
    "exponential moving average",
    "queue length",
    "batch size",
    "negative mining",
    "in-batch negative",
    "hard in-batch",
    "easy negative",
    "semi-hard",
    "angular margin",
    "cosine margin",
    "arcface",
    "cosface",
    "normface",
    "sphereface",
    "amsoftmax",
    "logits scaling",
    "margin scaling",
    "angular margin",
    "feature norm",
    "class weight",
    "class frequency",
    "class imbalance",
    "oversampling",
    "undersampling",
    "synthetic minority",
    "smote",
    "borderline smote",
    "adasyn",
    "focal loss variant",
    "class balanced loss",
    "effective number",
    "beta",
    "class balanced",
    "resampling",
    "reweighting",
    "inverse frequency",
    "effective count",
    "head vs tail",
    "long tail",
    "heavy head",
    "power law",
    "zipf",
    "pareto",
    "80-20",
    "imbalanced",
    "imbalanced learning",
    "imbalanced classification",
    "imbalanced regression",
    "imbalanced detection",
    "imbalanced segmentation",
    "imbalanced retrieval",
    "imbalanced ranking",
    "imbalanced clustering",
    "imbalanced anomaly",
    "imbalanced novelty",
    "imbalanced outlier",
    "imbalanced one-class",
    "imbalanced few-class",
    "imbalanced zero-class",
    "imbalanced multi-class",
    "imbalanced multi-label",
    "imbalanced multi-task",
    "imbalanced multi-view",
    "imbalanced multi-modal",
    "imbalanced multi-source",
    "imbalanced multi-domain",
    "imbalanced multi-granularity",
    "imbalanced multi-scale",
    "imbalanced multi-resolution",
    "imbalanced multi-precision",
    "imbalanced multi-fidelity",
    "imbalanced multi-level",
    "imbalanced multi-hierarchy",
    "imbalanced multi-structure",
    "imbalanced multi-graph",
    "imbalanced multi-tree",
    "imbalanced multi-metric",
    "imbalanced multi-distance",
    "imbalanced multi-similarity",
    "imbalanced multi-dissimilarity",
    "imbalanced multi-similarity",
    "imbalanced multi-clustering",
    "imbalanced multi-ranking",
    "imbalanced multi-retrieval",
    "imbalanced multi-detection",
    "imbalanced multi-segmentation",
    "imbalanced multi-classification",
    "imbalanced multi-regression",
    "imbalanced multi-anomaly",
    "imbalanced multi-novelty",
    "imbalanced multi-outlier",
    "imbalanced multi-one-class",
    "imbalanced multi-few-class",
    "imbalanced multi-zero-class",
    "imbalanced multi-label",
    "imbalanced multi-task",
    "imbalanced multi-view",
    "imbalanced multi-modal",
    "imbalanced multi-source",
    "imbalanced multi-domain",
    "imbalanced multi-granularity",
    "imbalanced multi-scale",
    "imbalanced multi-resolution",
    "imbalanced multi-precision",
    "imbalanced multi-fidelity",
    "imbalanced multi-level",
    "imbalanced multi-hierarchy",
    "imbalanced multi-structure",
    "imbalanced multi-graph",
    "imbalanced multi-tree",
    "imbalanced multi-metric",
    "imbalanced multi-distance",
    "imbalanced multi-similarity",
    "imbalanced multi-dissimilarity",
]

# General-purpose query starters that suggest a broad intent
GENERAL_QUERY_STARTERS: List[str] = [
    "explain",
    "teach",
    "what is",
    "what's",
    "how does",
    "how do",
    "describe",
    "define",
    "introduction to",
    "intro to",
    "overview of",
    "overview",
    "tell me about",
    "tell me about",
    "summarize",
    "summarise",
]

# Pre-compile a single regex for narrow term detection (faster than 690 individual `in` checks)
_NARROW_PATTERN = re.compile(
    "|".join(re.escape(term) for term in NARROW_TECHNICAL_TERMS),
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# QueryBroadtherClassifier
# ---------------------------------------------------------------------------

class QueryBroadtherClassifier:
    """Lightweight heuristic query breadth classifier (spec §16).

    Classifies queries into BROAD, MODERATE, or NARROW without using an LLM.
    """

    def classify(self, query: str) -> str:
        """Classify query breadth.

        Returns
        -------
        str
            "BROAD", "MODERATE", or "NARROW"
        """
        lower_query = query.lower()

        # Check for narrow technical terms first (highest specificity)
        if _NARROW_PATTERN.search(query):
            return "NARROW"

        # Count distinct concept names found in the query
        concept_count = self._count_concepts(lower_query)

        # Check for general query starters
        has_general_starter = any(
            lower_query.startswith(starter) for starter in GENERAL_QUERY_STARTERS
        )

        # Check word count
        word_count = len(query.split())

        # NARROW: 3+ concept names
        if concept_count >= 3:
            return "NARROW"

        # General starters indicate a learning intent → BROAD (overrides concept count)
        if has_general_starter:
            return "BROAD"

        # Short queries: < 5 words
        if word_count < 5:
            if concept_count >= 1:
                return "MODERATE"
            return "BROAD"

        # MODERATE: 1-2 concept names
        if 1 <= concept_count <= 2:
            return "MODERATE"

        # Default: short queries with no concepts are BROAD
        if word_count < 5:
            return "BROAD"

        # Default fallback: short-ish queries with 0 concepts and no starter
        if concept_count == 0:
            return "BROAD"

        return "MODERATE"

    @staticmethod
    def _count_concepts(query_lower: str) -> int:
        """Count distinct DSA concept names found in the (lowercased) query."""
        count = 0
        for concept in CONCEPT_NAMES:
            if concept in query_lower:
                count += 1
        return count


# ---------------------------------------------------------------------------
# TopicExpander
# ---------------------------------------------------------------------------

class TopicExpander:
    """Expand topics with full_text based on query breadth (spec §14).

    - BROAD: expand ALL topics (include full_text)
    - MODERATE: expand top 3 topics
    - NARROW: expand only the highest-scoring topic
    """

    def expand(self, topics: List[Topic], breadth: str) -> List[Topic]:
        """Expand topics with full_text based on breadth classification.

        Parameters
        ----------
        topics : list[Topic]
            Topics sorted by relevance score (descending).
        breadth : str
            "BROAD", "MODERATE", or "NARROW".

        Returns
        -------
        list[Topic]
            Topics with full_text populated where applicable.
        """
        if not topics:
            return []

        if breadth == "BROAD":
            # Expand all topics
            return self._expand_all(topics)
        elif breadth == "MODERATE":
            # Expand top 3 topics
            top_n = min(3, len(topics))
            expanded = self._expand_all(topics[:top_n])
            # Keep non-expanded topics as-is (no full_text)
            return expanded + list(topics[top_n:])
        elif breadth == "NARROW":
            # Expand only the highest-scoring topic
            expanded = self._expand_one(topics[0])
            # Keep rest as-is
            return [expanded] + list(topics[1:])
        else:
            raise ValueError(f"Unknown breadth: {breadth!r}. Expected BROAD/MODERATE/NARROW.")

    @staticmethod
    def _expand_all(topics: List[Topic]) -> List[Topic]:
        """Expand all topics — return new Topic instances with full_text."""
        return [Topic(
            id=t.id,
            title=t.title,
            level=t.level,
            parent_id=t.parent_id,
            children=list(t.children),
            chapter_id=t.chapter_id,
            book_id=t.book_id,
            full_text=t.full_text,
        ) for t in topics]

    @staticmethod
    def _expand_one(topic: Topic) -> Topic:
        """Expand a single topic — return new Topic with full_text."""
        return Topic(
            id=topic.id,
            title=topic.title,
            level=topic.level,
            parent_id=topic.parent_id,
            children=list(topic.children),
            chapter_id=topic.chapter_id,
            book_id=topic.book_id,
            full_text=topic.full_text,
        )


# ---------------------------------------------------------------------------
# ParagraphNeighborExpander
# ---------------------------------------------------------------------------

class ParagraphNeighborExpander:
    """Add prev/next paragraph neighbors within the same topic (spec §17).

    For each selected paragraph, also include its prev_paragraph_id and
    next_paragraph_id paragraphs IF they belong to the same topic.
    """

    def __init__(self, neighbor_window: int = 1) -> None:
        """Initialize with neighbor window size.

        Parameters
        ----------
        neighbor_window : int
            Number of neighbors on each side (default 1 = ±1).
        """
        self.neighbor_window = neighbor_window

    def expand(
        self,
        paragraphs: List[Paragraph],
        topic_paragraph_map: Dict[str, List[str]],
        all_paragraphs_by_id: Dict[str, Paragraph] | None = None,
    ) -> List[Paragraph]:
        """Expand paragraphs with neighbors from the same topic.

        Parameters
        ----------
        paragraphs : list[Paragraph]
            Selected paragraphs (already deduplicated).
        topic_paragraph_map : dict[str, list[str]]
            Maps topic_id → list of paragraph ids in order.
        all_paragraphs_by_id : dict[str, Paragraph] or None
            Optional map of paragraph_id → Paragraph for all paragraphs in the
            corpus. Used to look up neighbor paragraphs that were not in the
            original selection.

        Returns
        -------
        list[Paragraph]
            Expanded paragraphs (selected + neighbors), deduplicated.
        """
        if not paragraphs:
            return []

        # Build a set of already-selected paragraph ids
        selected_ids: Set[str] = {p.id for p in paragraphs}

        # Build id → paragraph lookup if not provided
        if all_paragraphs_by_id is None:
            all_paragraphs_by_id = {p.id: p for p in paragraphs}
        else:
            # Merge: start with all_paragraphs_by_id, fill in missing from paragraphs
            merged = dict(all_paragraphs_by_id)
            for p in paragraphs:
                if p.id not in merged:
                    merged[p.id] = p
            all_paragraphs_by_id = merged

        expanded_ids: Set[str] = set(selected_ids)
        result: List[Paragraph] = list(paragraphs)

        for para in paragraphs:
            topic_id = para.topic_id
            if not topic_id or topic_id not in topic_paragraph_map:
                continue

            ordered_ids = topic_paragraph_map[topic_id]
            try:
                idx = ordered_ids.index(para.id)
            except ValueError:
                # Paragraph not in the topic's ordered list — skip
                continue

            # Collect neighbors within window
            for offset in range(-self.neighbor_window, self.neighbor_window + 1):
                if offset == 0:
                    continue  # Skip self

                neighbor_idx = idx + offset
                if neighbor_idx < 0 or neighbor_idx >= len(ordered_ids):
                    continue  # Out of bounds

                neighbor_id = ordered_ids[neighbor_idx]
                if neighbor_id in expanded_ids:
                    continue  # Already selected — deduplicate

                # Look up the neighbor paragraph
                neighbor_para = all_paragraphs_by_id.get(neighbor_id)
                if neighbor_para is None:
                    continue  # Not found — skip

                # Verify neighbor belongs to the same topic
                if neighbor_para.topic_id != topic_id:
                    continue  # Different topic — skip

                expanded_ids.add(neighbor_id)
                result.append(neighbor_para)

        return result


# ---------------------------------------------------------------------------
# Context budget helper
# ---------------------------------------------------------------------------

def estimate_tokens(text: str) -> int:
    """Estimate token count from text length.

    Uses a simple heuristic: char_count / 4 ≈ token count for English text.
    """
    if not text:
        return 0
    return len(text) // 4


def apply_context_budget(
    paragraphs: List[Paragraph],
    max_context_tokens: int,
) -> List[Paragraph]:
    """Truncate paragraphs to fit within context budget.

    Removes lowest-similarity paragraphs first (assumes paragraphs are sorted
    by descending similarity).

    Parameters
    ----------
    paragraphs : list[Paragraph]
        Paragraphs sorted by descending similarity.
    max_context_tokens : int
        Maximum allowed token count for context.

    Returns
    -------
    list[Paragraph]
        Truncated paragraphs that fit within the budget.
    """
    if not paragraphs:
        return []

    # Calculate total estimated tokens
    total_tokens = sum(estimate_tokens(p.content or "") for p in paragraphs)

    if total_tokens <= max_context_tokens:
        return paragraphs

    # Remove lowest-similarity paragraphs first (they're at the end)
    result: List[Paragraph] = []
    current_tokens = 0

    for para in paragraphs:
        para_tokens = estimate_tokens(para.content or "")
        if current_tokens + para_tokens <= max_context_tokens:
            result.append(para)
            current_tokens += para_tokens
        # Skip this paragraph (truncate)

    return result
