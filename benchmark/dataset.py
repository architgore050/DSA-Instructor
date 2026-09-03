"""Benchmark dataset loader.

Implements spec §35 (reasoning-focused questions), §36 (difficulty dimensions),
§37 (categories), §38 (hard-question patterns), and §41 (gold answers with
required_claims / forbidden_claims / complexity / edge_cases / algorithm).

Usage
-----
    from benchmark.dataset import BenchmarkDataset, create_sample_dataset

    # Load an existing JSONL dataset
    ds = BenchmarkDataset()
    questions = ds.load("benchmark/questions.jsonl")

    # Filter by category / difficulty
    graph_qs = ds.filter_by_category(questions, "shortest_paths")
    expert_qs = ds.filter_by_difficulty(questions, "expert")

    # Random sample for quick testing
    sample = ds.sample(questions, n=3, seed=42)

    # Create a small sample dataset for testing
    create_sample_dataset("benchmark/sample_questions.jsonl")
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional


class BenchmarkDataset:
    """Load and filter benchmark question datasets (JSONL format).

    Each line in the JSONL file is a question dict with fields:
        id, category, difficulty, requires, question, gold_answer,
        required_claims, forbidden_claims, complexity, edge_cases, algorithm.
    """

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self, path: str) -> list[dict]:
        """Load a JSONL file and return a list of question dicts.

        Parameters
        ----------
        path : str
            Path to a JSONL file (one JSON object per line).

        Returns
        -------
        list[dict]
            List of question dicts.

        Raises
        ------
        FileNotFoundError
            If the file does not exist.
        ValueError
            If a line is not valid JSON.
        """
        path_obj = Path(path)
        if not path_obj.is_file():
            raise FileNotFoundError(f"Dataset file not found: {path}")

        questions: list[dict] = []
        with open(path_obj, "r", encoding="utf-8") as fh:
            for line_no, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    q = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSON on line {line_no} of {path}: {exc}"
                    ) from exc
                # Validate required fields
                self._validate_question(q, line_no)
                questions.append(q)

        return questions

    # ------------------------------------------------------------------
    # Filtering
    # ------------------------------------------------------------------

    def filter_by_category(
        self, questions: list[dict], category: str
    ) -> list[dict]:
        """Return questions matching *category* (case-insensitive)."""
        cat_lower = category.lower()
        return [q for q in questions if q.get("category", "").lower() == cat_lower]

    def filter_by_difficulty(
        self, questions: list[dict], difficulty: str
    ) -> list[dict]:
        """Return questions matching *difficulty* (case-insensitive)."""
        diff_lower = difficulty.lower()
        return [q for q in questions if q.get("difficulty", "").lower() == diff_lower]

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample(
        self, questions: list[dict], n: int, seed: int = 42
    ) -> list[dict]:
        """Return a random sample of *n* questions.

        Parameters
        ----------
        questions : list[dict]
            Full question list.
        n : int
            Number of questions to sample.
        seed : int
            Random seed for reproducibility.

        Returns
        -------
        list[dict]
            Random sample (no replacement).
        """
        rng = random.Random(seed)
        actual_n = min(n, len(questions))
        if actual_n <= 0:
            return []
        return rng.sample(questions, actual_n)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_question(q: dict, line_no: int) -> None:
        """Validate that a question dict has the required fields (spec §41)."""
        required = ("id", "category", "difficulty", "question", "gold_answer")
        for field_name in required:
            if field_name not in q:
                raise ValueError(
                    f"Question on line {line_no} is missing required field '{field_name}'"
                )
        # Optional but expected fields — provide defaults
        q.setdefault("requires", [])
        q.setdefault("required_claims", [])
        q.setdefault("forbidden_claims", [])
        q.setdefault("complexity", "")
        q.setdefault("edge_cases", [])
        q.setdefault("algorithm", "")


# ---------------------------------------------------------------------------
# Sample dataset factory
# ---------------------------------------------------------------------------

def create_sample_dataset(path: str) -> None:
    """Write a small sample dataset (10 questions) for testing.

    Covers categories: shortest_paths, complexity, graph_traversal, dp,
    adversarial.  Each question has realistic DSA content with full gold_answer,
    required_claims, forbidden_claims, complexity, edge_cases, and algorithm.

    Parameters
    ----------
    path : str
        Output JSONL file path.
    """
    questions: list[dict] = [
        {
            "id": "Q001",
            "category": "shortest_paths",
            "difficulty": "expert",
            "requires": ["Dijkstra", "greedy correctness", "counterexample"],
            "question": (
                "A graph contains one negative edge that does not lie on any "
                "shortest path from the source vertex. Is Dijkstra's algorithm "
                "guaranteed to produce correct shortest-path distances? "
                "Prove or give a counterexample."
            ),
            "gold_answer": (
                "Dijkstra's algorithm is NOT guaranteed to be correct even if the "
                "negative edge is not on the shortest path. The algorithm relies on "
                "the greedy invariant that once a vertex is settled (extracted from "
                "the priority queue), its distance is final. A negative edge can "
                "cause a previously-settled vertex to later have its distance "
                "reduced, violating this invariant. Even if the negative edge does "
                "not lie on the true shortest path from source to any vertex, the "
                "algorithm may settle vertices in an incorrect order because it "
                "assumes nonnegative edge weights throughout. A concrete counterexample: "
                "consider a graph with vertices s, a, b and edges (s,a,1), (s,b,2), "
                "(a,b,-2). Dijkstra settles a (dist=1), then b (dist=2). But the "
                "true shortest path to b is s→a→b with cost -1. The negative edge "
                "(a,b) was not on the shortest path from s to any vertex as computed "
                "by Dijkstra, yet it still causes an incorrect result."
            ),
            "required_claims": [
                "Dijkstra requires nonnegative edge weights",
                "settled vertex cannot later be improved",
                "negative edges violate the greedy invariant",
                "a negative edge not on the shortest path can still cause incorrect results",
            ],
            "forbidden_claims": [
                "negative edges only cause problems when on the shortest path",
                "Dijkstra works with negative edges if they are not on the shortest path",
            ],
            "complexity": "O((V+E) log V) for Dijkstra, but incorrect with negative edges",
            "edge_cases": [
                "negative self-loop",
                "negative edge between two non-source vertices",
                "graph where negative edge is reachable only through a settled vertex",
            ],
            "algorithm": "Dijkstra's algorithm with priority queue",
        },
        {
            "id": "Q002",
            "category": "complexity",
            "difficulty": "expert",
            "requires": ["heap construction", "amortized analysis", "recurrence"],
            "question": (
                "Explain why bottom-up heap construction (build_heap) runs in "
                "O(n) time, despite each individual heapify operation taking "
                "O(log n) time. Provide a rigorous analysis."
            ),
            "gold_answer": (
                "The key insight is that most nodes in a binary heap are leaves or "
                "near the bottom of the tree, where heapify is very fast. We "
                "analyze by summing over all nodes at each height h: there are at "
                "most n/2^(h+1) nodes at height h, and each heapify at height h "
                "takes O(h) time. The total work is sum_{h=0}^{log n} "
                "(n/2^(h+1)) * O(h) = O(n) * sum_{h=0}^{log n} h/2^(h+1). The "
                "infinite series sum h/2^(h+1) converges to a constant (specifically "
                "1), so the total is O(n). This is an amortized analysis — while "
                "the root heapify takes O(log n), the vast majority of nodes take "
                "O(1) time."
            ),
            "required_claims": [
                "most nodes are near the bottom of the heap",
                "there are at most n/2^(h+1) nodes at height h",
                "the series sum h/2^(h+1) converges to a constant",
                "total work is O(n) not O(n log n)",
            ],
            "forbidden_claims": [
                "build_heap is O(n log n) because we call heapify n times",
                "each heapify is O(1) on average",
            ],
            "complexity": "O(n) total, with individual heapify operations ranging from O(1) to O(log n)",
            "edge_cases": [
                "single-element heap",
                "heap with exactly 2 levels",
                "non-perfect binary heap shape",
            ],
            "algorithm": "Bottom-up heap construction (Floyd's algorithm)",
        },
        {
            "id": "Q003",
            "category": "graph_traversal",
            "difficulty": "intermediate",
            "requires": ["BFS", "DFS", "shortest path", "unweighted graphs"],
            "question": (
                "Give a concrete example of a graph where DFS finds a target "
                "vertex but does not find the shortest path to it first. "
                "Explain why this happens and under what conditions BFS would "
                "be preferred."
            ),
            "gold_answer": (
                "Consider a graph with vertices s, a, b, t and edges "
                "(s,a), (s,b), (a,t), (b,t). DFS starting from s might visit "
                "a first, then immediately visit t through a, finding the path "
                "s→a→t of length 2. BFS would also find this same-length path. "
                "To make DFS fail at finding the shortest path, consider: s has "
                "edges to a and b; a has edge to c; c has edge to t; b has "
                "direct edge to t. The shortest path is s→b→t (length 2), but "
                "DFS might go s→a→c→t (length 3) first, exploring the deeper "
                "branch before discovering the shorter one. BFS is preferred "
                "for shortest-path in unweighted graphs because it explores "
                "layer by layer (by distance from source), guaranteeing the "
                "first time a vertex is reached, it is via the shortest path."
            ),
            "required_claims": [
                "DFS does not guarantee shortest path",
                "DFS explores depth-first, potentially finding longer paths first",
                "BFS explores layer by layer by distance from source",
                "BFS guarantees shortest path in unweighted graphs",
            ],
            "forbidden_claims": [
                "DFS always finds the shortest path",
                "BFS and DFS are equivalent for shortest-path problems",
            ],
            "complexity": "DFS: O(V+E), BFS: O(V+E), but only BFS guarantees shortest path",
            "edge_cases": [
                "graph with a single path",
                "complete graph",
                "graph with cycles",
            ],
            "algorithm": "DFS and BFS graph traversal",
        },
        {
            "id": "Q004",
            "category": "dp",
            "difficulty": "expert",
            "requires": ["knapsack", "state transition", "memoization", "tabulation"],
            "question": (
                "In the 0/1 knapsack DP formulation, the recurrence is "
                "dp[i][w] = max(dp[i-1][w], dp[i-1][w-weight[i]] + value[i]). "
                "Why does the include transition use dp[i-1][w-weight[i]] "
                "instead of dp[i][w-weight[i]]? What would go wrong if we used "
                "the latter, and which variant of knapsack would it correspond to?"
            ),
            "gold_answer": (
                "The include transition uses dp[i-1][w-weight[i]] because in the "
                "0/1 knapsack problem, each item can be used at most once. By "
                "referencing dp[i-1], we ensure that item i has not already been "
                "included in the sub-solution — we are deciding whether to include "
                "item i for the first time. If we used dp[i][w-weight[i]], we would "
                "be allowing item i to be included multiple times, which corresponds "
                "to the unbounded (complete) knapsack problem, not the 0/1 variant. "
                "In the unbounded knapsack, dp[i][w-weight[i]] represents the optimal "
                "solution for capacity w-weight[i] using items 1..i, which already "
                "may include item i, so adding item i again is valid."
            ),
            "required_claims": [
                "0/1 knapsack uses dp[i-1] to prevent reusing the same item",
                "dp[i][w-weight[i]] would allow multiple uses of item i",
                "dp[i][w-weight[i]] corresponds to the unbounded knapsack variant",
                "the difference is between using an item once vs. unlimited times",
            ],
            "forbidden_claims": [
                "both transitions are equivalent for 0/1 knapsack",
                "using dp[i][w-weight[i]] in 0/1 knapsack is correct",
            ],
            "complexity": "O(n*W) time and space for tabulation, O(n*W) time for memoization",
            "edge_cases": [
                "zero-weight item",
                "zero-value item",
                "capacity of zero",
                "item weight greater than capacity",
            ],
            "algorithm": "0/1 Knapsack dynamic programming",
        },
        {
            "id": "Q005",
            "category": "adversarial",
            "difficulty": "expert",
            "requires": ["binary search", "off-by-one", "invariant", "edge cases"],
            "question": (
                "Consider this binary search implementation for finding the "
                "first occurrence of a value in a sorted array:\n\n"
                "    lo = 0, hi = n\n"
                "    while lo < hi:\n"
                "        mid = (lo + hi) // 2\n"
                "        if arr[mid] < target:\n"
                "            lo = mid + 1\n"
                "        else:\n"
                "            hi = mid\nn    return lo\n\n"
                "This works correctly on most inputs but returns n (out of bounds) "
                "when the target is larger than all elements. Is this a bug? "
                "What invariant does this implementation maintain, and what "
                "postcondition should the caller check?"
            ),
            "gold_answer": (
                "This is not a bug — it is correct behavior for a lower_bound "
                "implementation. The invariant maintained is: the first occurrence "
                "of target (if it exists) is always in the range [lo, hi). When "
                "lo == hi, the loop terminates and lo points to either the first "
                "occurrence of target or the insertion point (the position where "
                "target would be inserted to maintain sorted order). If target is "
                "larger than all elements, lo = n, which correctly indicates "
                "that target is not found. The caller must check that lo < n AND "
                "arr[lo] == target to confirm the element exists. This is the "
                "standard lower_bound / first_occurrence pattern, consistent with "
                "C++'s std::lower_bound."
            ),
            "required_claims": [
                "the invariant is that the answer (if exists) is in [lo, hi)",
                "lo == hi means the search space is empty",
                "returning n when target exceeds all elements is correct",
                "the caller must verify lo < n and arr[lo] == target",
            ],
            "forbidden_claims": [
                "this is a bug because it returns an out-of-bounds index",
                "hi should be initialized to n-1",
                "the function should return -1 when not found",
            ],
            "complexity": "O(log n) time, O(1) space",
            "edge_cases": [
                "empty array",
                "target not in array",
                "target appears multiple times",
                "target is the first element",
                "target is larger than all elements",
            ],
            "algorithm": "Binary search (lower_bound variant)",
        },
        {
            "id": "Q006",
            "category": "shortest_paths",
            "difficulty": "intermediate",
            "requires": ["Bellman-Ford", "negative cycles", "relaxation"],
            "question": (
                "Why does Bellman-Ford detect negative cycles, and why does it "
                "require exactly V-1 relaxation passes? What happens if you run "
                "it for V passes and still find an edge that can be relaxed?"
            ),
            "gold_answer": (
                "In a graph with V vertices, any simple path has at most V-1 edges. "
                "Bellman-Ford works by relaxing all edges repeatedly. After k passes, "
                "it guarantees to have found the shortest path to every vertex that "
                "uses at most k edges. After V-1 passes, all shortest simple paths "
                "are guaranteed to be found (if no negative cycle exists). If after "
                "V-1 passes, another edge can still be relaxed, it means there exists "
                "a path with V or more edges that is shorter — which is only possible "
                "if there is a negative cycle reachable from the source (since any "
                "path with V+ edges must revisit a vertex, creating a cycle). "
                "Traversing this negative cycle repeatedly would make the path cost "
                "arbitrarily small, so no finite shortest path exists."
            ),
            "required_claims": [
                "simple paths have at most V-1 edges",
                "after k passes, shortest paths with up to k edges are found",
                "V-1 passes guarantee all simple shortest paths",
                "a relaxable edge after V-1 passes indicates a negative cycle",
            ],
            "forbidden_claims": [
                "Bellman-Ford detects negative cycles because distances become negative",
                "V passes are needed for correctness",
            ],
            "complexity": "O(V * E) time, O(V) space",
            "edge_cases": [
                "disconnected graph",
                "negative cycle not reachable from source",
                "graph with only negative edges",
                "single vertex",
            ],
            "algorithm": "Bellman-Ford algorithm",
        },
        {
            "id": "Q007",
            "category": "complexity",
            "difficulty": "intermediate",
            "requires": ["recurrence", "master theorem", "substitution method"],
            "question": (
                "Solve the recurrence T(n) = 2T(n/2) + n/log(n) for n > 1, "
                "T(1) = 1. Show that the Master Theorem does not directly apply "
                "and provide a tight asymptotic bound."
            ),
            "gold_answer": (
                "The Master Theorem does not directly apply because the non-recursive "
                "part f(n) = n/log(n) is not polynomially related to n^(log_b(a)) = "
                "n^1 = n. Specifically, f(n) = n/log(n) = n * n^(-log n / log log n) "
                "which is smaller than n by a factor of log(n), but not by a "
                "polynomial factor n^epsilon. Using the recursion tree method: at "
                "level i, there are 2^i subproblems of size n/2^i, each contributing "
                "(n/2^i)/log(n/2^i) = (n/2^i)/(log n - i). The total at level i is "
                "n/(log n - i). There are log n levels. Summing: T(n) = sum_{i=0}^{log n - 1} "
                "n/(log n - i) = n * sum_{j=1}^{log n} 1/j = n * H(log n) = "
                "n * Theta(log(log n)) = Theta(n log log n). So T(n) = Theta(n log log n)."
            ),
            "required_claims": [
                "Master Theorem case 2 requires f(n) = Theta(n^log_b(a) * log^k(n)) for k >= 0",
                "n/log(n) = n * log^(-1)(n) which is not log^k(n) for k >= 0",
                "the recursion tree has log n levels",
                "the harmonic series sum gives Theta(n log log n)",
            ],
            "forbidden_claims": [
                "T(n) = O(n log n) by Master Theorem case 2",
                "the answer is Theta(n) because f(n) is smaller than n",
            ],
            "complexity": "Theta(n log log n)",
            "edge_cases": [
                "n not a power of 2",
                "base case T(1) = 1 vs T(1) = 0",
                "floor/ceiling in n/2",
            ],
            "algorithm": "Recurrence solving (recursion tree + harmonic series)",
        },
        {
            "id": "Q008",
            "category": "graph_traversal",
            "difficulty": "expert",
            "requires": ["topological sort", "DAG", "DP on DAG", "longest path"],
            "question": (
                "Why can the longest path in a DAG be found in O(V+E) time using "
                "topological sort, while the longest path in a general graph is "
                "NP-hard? What is the key difference that makes the DAG case "
                "tractable?"
            ),
            "gold_answer": (
                "In a DAG, a topological ordering guarantees that for every edge "
                "(u,v), u appears before v. This means we can process vertices in "
                "topological order and compute the longest path to each vertex "
                "using only already-computed values — there are no cycles, so no "
                "infinite loops. The algorithm: compute topological sort (O(V+E)), "
                "then for each vertex u in topo order, relax all outgoing edges "
                "(u,v): dist[v] = max(dist[v], dist[u] + w(u,v)). This is O(V+E) "
                "total. In a general graph, cycles create dependencies that prevent "
                "such a linear ordering. The longest path problem in general graphs "
                "is NP-hard because a Hamiltonian path (visit every vertex exactly "
                "once) is a special case, and finding a Hamiltonian path is NP-complete. "
                "The key difference is the absence of cycles in a DAG, which provides "
                "a valid processing order."
            ),
            "required_claims": [
                "topological order ensures all predecessors are processed before a vertex",
                "no cycles means no infinite dependency loops",
                "longest path in general graphs is NP-hard via Hamiltonian path reduction",
                "the DAG algorithm runs in O(V+E) using topo sort + single pass",
            ],
            "forbidden_claims": [
                "longest path in general graphs can be solved with Bellman-Ford",
                "topological sort works for graphs with cycles",
            ],
            "complexity": "O(V+E) for DAG; NP-hard for general graphs",
            "edge_cases": [
                "disconnected DAG",
                "graph with negative edge weights",
                "single vertex",
                "graph with no edges",
            ],
            "algorithm": "Topological sort + longest path on DAG",
        },
        {
            "id": "Q009",
            "category": "adversarial",
            "difficulty": "expert",
            "requires": ["hashing", "collision", "adversarial input", "load factor"],
            "question": (
                "A hash table uses chaining with a hash function h(k) = k mod m. "
                "Construct an adversarial input set of size n that causes all "
                "elements to collide in the same bucket, degrading performance "
                "from O(1) expected to O(n) worst case per operation. What "
                "countermeasures prevent this?"
            ),
            "gold_answer": (
                "For h(k) = k mod m, any set of keys {a, a+m, a+2m, ..., a+(n-1)m} "
                "will all hash to the same bucket (hash value a mod m). For example, "
                "with m=100 and a=0, the set {0, 100, 200, ..., 100*(n-1)} all "
                "collide. This makes the chaining degenerate to a linked list, "
                "giving O(n) worst-case for search/insert/delete. Countermeasures: "
                "(1) Universal hashing — pick h randomly from a universal family "
                "so no adversarial input can target a specific hash function. "
                "(2) Double hashing — use a second hash function when collisions occur. "
                "(3) Dynamic resizing — resize the table when load factor exceeds a "
                "threshold, rehashing with a new random hash function. "
                "(4) Balanced tree fallback — use a BST in each bucket (as in "
                "Java's HashMap) for O(log n) worst-case per bucket."
            ),
            "required_claims": [
                "keys {a, a+m, a+2m, ...} all collide under h(k) = k mod m",
                "collision degrades to O(n) per operation",
                "universal hashing prevents adversarial targeting",
                "dynamic resizing with rehashing is a countermeasure",
            ],
            "forbidden_claims": [
                "this cannot happen with a good hash function",
                "chaining always guarantees O(1) performance",
            ],
            "complexity": "O(n) worst case with adversarial input; O(1) expected with universal hashing",
            "edge_cases": [
                "m is prime vs composite",
                "n equals m (load factor = 1)",
                "n > m (load factor > 1)",
            ],
            "algorithm": "Hash table with chaining",
        },
        {
            "id": "Q010",
            "category": "dp",
            "difficulty": "expert",
            "requires": ["LIS", "patience sorting", "binary search", "O(n log n)"],
            "question": (
                "The naive LIS (Longest Increasing Subsequence) algorithm runs in "
                "O(n^2) time. Explain how to solve it in O(n log n) using the "
                "patience sorting approach with binary search. What does the "
                '"tails" array represent, and why does its length equal the LIS length?'
            ),
            "gold_answer": (
                "The O(n log n) algorithm maintains a 'tails' array where tails[i] "
                "stores the smallest tail element of all increasing subsequences of "
                "length i+1 found so far. For each element x in the input: if x is "
                "greater than all elements in tails, append it (extending the longest "
                "subsequence). Otherwise, find the smallest element in tails that is "
                ">= x (using binary search / lower_bound) and replace it with x. "
                "This replacement does not change any existing subsequence length but "
                "makes the subsequence of that length end with a smaller element, "
                "making it easier to extend later. The length of tails always equals "
                "the LIS length because: (1) tails is always sorted (provable by "
                "induction), and (2) tails[i] being the minimum tail for length i+1 "
                "means we have found a subsequence of length i+1. Binary search "
                "(lower_bound) gives O(log k) per element where k is the current "
                "LIS length, yielding O(n log n) total."
            ),
            "required_claims": [
                "tails[i] stores the smallest tail of all increasing subsequences of length i+1",
                "tails is always sorted (enabling binary search)",
                "replacing tails[k] with x maintains the invariant while making future extensions easier",
                "len(tails) equals the LIS length",
                "binary search (lower_bound) gives O(log n) per element",
            ],
            "forbidden_claims": [
                "tails stores the actual LIS elements",
                "the algorithm finds the longest non-decreasing subsequence",
                "tails can contain duplicate values",
            ],
            "complexity": "O(n log n) time, O(n) space",
            "edge_cases": [
                "empty array",
                "all elements equal",
                "strictly decreasing array",
                "strictly increasing array",
                "single element",
            ],
            "algorithm": "LIS via patience sorting with binary search",
        },
    ]

    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    with open(path_obj, "w", encoding="utf-8") as fh:
        for q in questions:
            fh.write(json.dumps(q, ensure_ascii=False) + "\n")
