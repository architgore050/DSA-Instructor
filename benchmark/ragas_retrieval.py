"""RAGAS-based retrieval evaluation for DSA Mentor.

Evaluates the retrieval pipeline using RAGAS metrics tailored to DSA tutoring:
    - Context Precision: Of all retrieved paragraphs, how many contain information
      actually needed to answer the DSA question?
    - Context Recall: Does the retrieved context contain all information needed
      to fully answer the question (complexity, edge cases, algorithm mechanics)?
    - Faithfulness: Is the generated answer grounded in the retrieved context?

DSA Retrieval Philosophy
------------------------
For a DSA tutoring system, good retrieval must satisfy:

1. Technical precision: Retrieved paragraphs must contain correct algorithm
   mechanics, not just related content. A wrong complexity analysis in retrieved
   context is worse than no context.

2. Multi-concept coverage: DSA questions span multiple concepts (e.g., comparing
   Dijkstra vs Bellman-Ford requires retrieval from both algorithms).

3. Edge case awareness: DSA answers depend on edge cases (empty input, single
   element, negative values, cycles). Retrieval should include edge case
   discussions.

4. Complexity analysis: Every DSA answer requires time/space complexity.
   Retrieved context must include complexity information.

5. Algorithmic reasoning over code: For tutoring, understanding invariants,
   greedy choices, and recurrences matters more than code snippets.

6. Source diversity: Drawing from multiple sources (textbooks, tutorials, CP
   resources) provides different perspectives and depths.

Usage
-----
    from benchmark.ragas_retrieval import RAGASRetrievalBenchmark

    benchmark = RAGASRetrievalBenchmark(config, retriever, embedding_client)
    results = benchmark.run()
    benchmark.save_report(results, "benchmark/ragas_results.json")
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# DSA Retrieval Dataset
# ---------------------------------------------------------------------------
# Each entry: {question, gold_answer, gold_context}
# - question: The user query
# - gold_answer: Reference answer for faithfulness evaluation
# - gold_context: Description of information that SHOULD be present in retrieved
#   paragraphs. RAGAS uses this to evaluate context precision and recall.

DSA_RETRIEVAL_DATASET: List[Dict[str, str]] = [
    {
        "id": "RAG001",
        "category": "graph_algorithms",
        "difficulty": "expert",
        "question": (
            "A graph contains one negative edge that does not lie on any shortest "
            "path from the source vertex. Is Dijkstra's algorithm guaranteed to "
            "produce correct shortest-path distances? Explain why or why not."
        ),
        "gold_answer": (
            "Dijkstra's algorithm is NOT guaranteed to be correct even if the "
            "negative edge is not on the shortest path. The algorithm relies on "
            "the greedy invariant that once a vertex is settled (extracted from "
            "the priority queue), its distance is final. A negative edge can "
            "cause a previously-settled vertex to later have its distance "
            "reduced, violating this invariant. Dijkstra requires all edge "
            "weights to be non-negative. The algorithm uses a greedy approach "
            "with a min-priority queue and runs in O((V+E) log V) time. Even "
            "if the negative edge does not lie on the true shortest path, the "
            "algorithm may settle vertices in an incorrect order because it "
            "assumes nonnegative edge weights throughout. A counterexample: "
            "vertices s, a, b with edges (s,a,1), (s,b,2), (a,b,-2). Dijkstra "
            "settles a (dist=1), then b (dist=2), but the true shortest path "
            "to b is s->a->b with cost -1."
        ),
        "gold_context": (
            "Information about Dijkstra's algorithm requirements and mechanics: "
            "Dijkstra requires non-negative edge weights; uses a greedy approach "
            "with a min-priority queue; time complexity O((V+E) log V); the "
            "greedy invariant states that once a vertex is settled, its distance "
            "is final; negative edges violate this invariant because they can "
            "reduce distances of already-settled vertices; a negative edge not "
            "on the shortest path can still cause incorrect vertex ordering; "
            "counterexample with vertices s, a, b and edges (s,a,1), (s,b,2), "
            "(a,b,-2) where Dijkstra computes dist[b]=2 but true shortest is -1."
        ),
    },
    {
        "id": "RAG002",
        "category": "graph_algorithms",
        "difficulty": "intermediate",
        "question": (
            "Compare BFS and DFS for finding the shortest path in an unweighted "
            "graph. Why does BFS guarantee shortest path while DFS does not?"
        ),
        "gold_answer": (
            "BFS (Breadth-First Search) guarantees shortest path in unweighted "
            "graphs because it explores vertices layer by layer, ordered by "
            "distance from the source. The first time a vertex is visited, it "
            "is via the shortest path. BFS uses a queue and runs in O(V+E) time "
            "and space. DFS (Depth-First Search) explores as far as possible "
            "along each branch before backtracking. It does NOT guarantee "
            "shortest path because it may reach a vertex through a longer path "
            "before discovering the shorter one. DFS uses a stack (or recursion) "
            "and also runs in O(V+E) time, but the path found may not be optimal. "
            "Example: s->a->c->t (length 3) vs s->b->t (length 2). DFS might "
            "find the longer path first."
        ),
        "gold_context": (
            "Information about BFS and DFS for shortest path: BFS explores "
            "layer by layer by distance from source; BFS guarantees shortest "
            "path in unweighted graphs because first visit = shortest distance; "
            "BFS uses a queue data structure; BFS time complexity O(V+E); DFS "
            "explores depth-first along branches; DFS does NOT guarantee shortest "
            "path; DFS may find longer paths before shorter ones; DFS uses a "
            "stack or recursion; DFS time complexity O(V+E); counterexample "
            "where s->a->c->t (length 3) is found before s->b->t (length 2)."
        ),
    },
    {
        "id": "RAG003",
        "category": "graph_algorithms",
        "difficulty": "intermediate",
        "question": (
            "Why does Bellman-Ford require exactly V-1 relaxation passes? What "
            "does it mean if an edge can still be relaxed after V-1 passes?"
        ),
        "gold_answer": (
            "In a graph with V vertices, any simple path has at most V-1 edges. "
            "Bellman-Ford works by relaxing all edges repeatedly. After k passes, "
            "it guarantees to have found the shortest path to every vertex that "
            "uses at most k edges. After V-1 passes, all shortest simple paths "
            "are guaranteed to be found (if no negative cycle exists). If after "
            "V-1 passes, another edge can still be relaxed, it means there exists "
            "a path with V or more edges that is shorter — which is only possible "
            "if there is a negative cycle reachable from the source. Traversing "
            "this negative cycle repeatedly would make the path cost arbitrarily "
            "small, so no finite shortest path exists. Bellman-Ford runs in "
            "O(V*E) time and uses O(V) space."
        ),
        "gold_context": (
            "Information about Bellman-Ford algorithm: simple paths have at "
            "most V-1 edges in a graph with V vertices; after k relaxation "
            "passes, shortest paths with up to k edges are guaranteed; V-1 "
            "passes guarantee all simple shortest paths are found; an edge "
            "that can still be relaxed after V-1 passes indicates a negative "
            "cycle reachable from the source; negative cycles make shortest "
            "path undefined (arbitrarily small); Bellman-Ford time complexity "
            "O(V*E); space complexity O(V); edge relaxation updates distance "
            "estimates; negative cycle detection is a key feature of Bellman-Ford."
        ),
    },
    {
        "id": "RAG004",
        "category": "dynamic_programming",
        "difficulty": "expert",
        "question": (
            "In the 0/1 knapsack DP formulation, why does the include transition "
            "use dp[i-1][w-weight[i]] instead of dp[i][w-weight[i]]? What would "
            "go wrong with the latter?"
        ),
        "gold_answer": (
            "The include transition uses dp[i-1][w-weight[i]] because in the "
            "0/1 knapsack problem, each item can be used at most once. By "
            "referencing dp[i-1], we ensure that item i has not already been "
            "included in the sub-solution — we are deciding whether to include "
            "item i for the first time. If we used dp[i][w-weight[i]], we would "
            "be allowing item i to be included multiple times, which corresponds "
            "to the unbounded (complete) knapsack problem. The recurrence is "
            "dp[i][w] = max(dp[i-1][w], dp[i-1][w-weight[i]] + value[i]). "
            "The space can be optimized to O(W) using a 1D array with reverse "
            "iteration. Time complexity is O(n*W) where n is number of items."
        ),
        "gold_context": (
            "Information about 0/1 knapsack DP: each item can be used at most "
            "once; dp[i-1][w-weight[i]] ensures item i is not already included; "
            "dp[i][w-weight[i]] would allow multiple uses of item i; dp[i][w-weight[i]] "
            "corresponds to the unbounded knapsack variant; the recurrence is "
            "dp[i][w] = max(dp[i-1][w], dp[i-1][w-weight[i]] + value[i]); "
            "space optimization to O(W) using 1D array with reverse iteration; "
            "time complexity O(n*W) where n = number of items, W = capacity; "
            "the exclude transition uses dp[i-1][w] (item not included)."
        ),
    },
    {
        "id": "RAG005",
        "category": "dynamic_programming",
        "difficulty": "expert",
        "question": (
            "Explain how the O(n log n) LIS algorithm works using the patience "
            "sorting approach. What does the 'tails' array represent?"
        ),
        "gold_answer": (
            "The O(n log n) LIS algorithm maintains a 'tails' array where "
            "tails[i] stores the smallest tail element of all increasing "
            "subsequences of length i+1 found so far. For each element x: if x "
            "is greater than all elements in tails, append it (extending the "
            "longest subsequence). Otherwise, find the smallest element in "
            "tails >= x using binary search (lower_bound) and replace it. This "
            "makes the subsequence of that length end with a smaller element, "
            "making it easier to extend later. The length of tails always equals "
            "the LIS length because tails is always sorted (provable by "
            "induction) and tails[i] being the minimum tail for length i+1 "
            "means a subsequence of length i+1 exists. Binary search gives "
            "O(log k) per element, yielding O(n log n) total."
        ),
        "gold_context": (
            "Information about O(n log n) LIS algorithm: tails[i] stores the "
            "smallest tail of all increasing subsequences of length i+1; for "
            "each element, append if greater than all tails, otherwise replace "
            "using binary search (lower_bound); replacement makes future "
            "extensions easier by minimizing the tail value; tails is always "
            "sorted enabling binary search; len(tails) equals the LIS length; "
            "binary search (lower_bound) gives O(log k) per element; total "
            "time complexity O(n log n); space complexity O(n); tails does NOT "
            "store the actual LIS elements, only the minimum tail values."
        ),
    },
    {
        "id": "RAG006",
        "category": "dynamic_programming",
        "difficulty": "expert",
        "question": (
            "Why can the longest path in a DAG be found in O(V+E) time using "
            "topological sort, while it is NP-hard in general graphs?"
        ),
        "gold_answer": (
            "In a DAG, a topological ordering guarantees that for every edge "
            "(u,v), u appears before v. This means we can process vertices in "
            "topological order and compute the longest path to each vertex "
            "using only already-computed values — there are no cycles, so no "
            "infinite dependency loops. Algorithm: compute topological sort "
            "(O(V+E)), then for each vertex u in topo order, relax all outgoing "
            "edges: dist[v] = max(dist[v], dist[u] + w(u,v)). Total: O(V+E). "
            "In general graphs, cycles create dependencies preventing linear "
            "ordering. Longest path in general graphs is NP-hard because a "
            "Hamiltonian path is a special case, and finding a Hamiltonian path "
            "is NP-complete. The key difference is the absence of cycles in DAGs."
        ),
        "gold_context": (
            "Information about longest path in DAG: topological order ensures "
            "all predecessors are processed before a vertex; no cycles means "
            "no infinite dependency loops; algorithm is topo sort + single "
            "pass relaxation; dist[v] = max(dist[v], dist[u] + w(u,v)); "
            "time complexity O(V+E); in general graphs longest path is NP-hard "
            "via Hamiltonian path reduction; Hamiltonian path finding is NP-complete; "
            "the key difference is absence of cycles providing valid processing "
            "order; DAG longest path works with negative edge weights."
        ),
    },
    {
        "id": "RAG007",
        "category": "data_structures",
        "difficulty": "expert",
        "question": (
            "A hash table uses h(k) = k mod m. Construct an adversarial input "
            "that causes all elements to collide. What countermeasures prevent this?"
        ),
        "gold_answer": (
            "For h(k) = k mod m, any set {a, a+m, a+2m, ..., a+(n-1)m} all "
            "hash to the same bucket. With m=100 and a=0: {0, 100, 200, ...} "
            "all collide, degrading chaining to a linked list with O(n) worst "
            "case per operation. Countermeasures: (1) Universal hashing — pick "
            "h randomly from a universal family so no adversarial input can "
            "target a specific function. (2) Double hashing — use a second hash "
            "function when collisions occur. (3) Dynamic resizing — resize when "
            "load factor exceeds threshold, rehashing with new random function. "
            "(4) Balanced tree fallback — use BST in each bucket for O(log n) "
            "worst case per bucket. Expected time with universal hashing is O(1)."
        ),
        "gold_context": (
            "Information about hash table adversarial attacks: keys {a, a+m, "
            "a+2m, ...} all collide under h(k) = k mod m; collision degrades "
            "chaining to O(n) worst case per operation; universal hashing picks "
            "h randomly from a universal family to prevent adversarial targeting; "
            "double hashing uses a second hash function for collision resolution; "
            "dynamic resizing rehashes with new random function when load factor "
            "exceeds threshold; balanced tree fallback uses BST in each bucket "
            "for O(log n) worst case; expected time O(1) with good hashing; "
            "load factor = n/m where n = elements, m = table size."
        ),
    },
    {
        "id": "RAG008",
        "category": "data_structures",
        "difficulty": "expert",
        "question": (
            "Explain why bottom-up heap construction (build_heap) runs in O(n) "
            "time despite individual heapify taking O(log n). Provide the analysis."
        ),
        "gold_answer": (
            "Most nodes in a binary heap are near the bottom where heapify is "
            "fast. We sum over nodes at each height h: there are at most n/2^(h+1) "
            "nodes at height h, each taking O(h) time. Total work = "
            "sum_{h=0}^{log n} (n/2^(h+1)) * O(h) = O(n) * sum_{h=0}^{log n} "
            "h/2^(h+1). The series sum h/2^(h+1) converges to a constant (1), "
            "so total is O(n). This is amortized analysis — while root heapify "
            "takes O(log n), the vast majority of nodes take O(1) time. There "
            "are n/2 leaves (height 0), n/4 nodes at height 1, etc."
        ),
        "gold_context": (
            "Information about bottom-up heap construction (Floyd's algorithm): "
            "most nodes are near the bottom of the heap; at most n/2^(h+1) nodes "
            "at height h; heapify at height h takes O(h) time; total work = "
            "sum_{h=0}^{log n} (n/2^(h+1)) * O(h); the series sum h/2^(h+1) "
            "converges to a constant (specifically 1); total complexity O(n) "
            "not O(n log n); this is amortized analysis; root heapify takes "
            "O(log n) but most nodes take O(1); n/2 nodes are leaves at height 0."
        ),
    },
    {
        "id": "RAG009",
        "category": "sorting",
        "difficulty": "intermediate",
        "question": (
            "Prove that any comparison-based sorting algorithm requires Omega(n "
            "log n) comparisons in the worst case. What is the decision tree "
            "argument?"
        ),
        "gold_answer": (
            "Any comparison-based sorting algorithm can be modeled as a decision "
            "tree where each internal node represents a comparison and each leaf "
            "represents a permutation. For n elements, there are n! possible "
            "permutations, so the tree must have at least n! leaves. A binary "
            "tree of height h has at most 2^h leaves. So 2^h >= n!, meaning "
            "h >= log2(n!). Using Stirling's approximation: log2(n!) = "
            "log2(sqrt(2*pi*n) * (n/e)^n) = Theta(n log n). Therefore, the "
            "height (worst-case comparisons) is Omega(n log n). This lower bound "
            "applies to merge sort, heap sort, and any comparison-based sorter."
        ),
        "gold_context": (
            "Information about comparison sort lower bound: comparison-based "
            "sorting modeled as decision tree; internal nodes are comparisons, "
            "leaves are permutations; n! possible permutations for n elements; "
            "tree must have at least n! leaves; binary tree of height h has at "
            "most 2^h leaves; 2^h >= n! implies h >= log2(n!); Stirling's "
            "approximation: log2(n!) = Theta(n log n); worst-case height is "
            "Omega(n log n); applies to merge sort, heap sort, quick sort; "
            "non-comparison sorts (counting, radix) can beat this bound."
        ),
    },
    {
        "id": "RAG010",
        "category": "algorithms",
        "difficulty": "intermediate",
        "question": (
            "Consider binary search for first occurrence with lo=0, hi=n, "
            "returning lo. It returns n when target exceeds all elements. Is "
            "this a bug? What invariant does it maintain?"
        ),
        "gold_answer": (
            "This is not a bug — it is correct behavior for a lower_bound "
            "implementation. The invariant: the first occurrence of target (if "
            "it exists) is always in range [lo, hi). When lo == hi, the loop "
            "terminates and lo points to either the first occurrence or the "
            "insertion point. If target exceeds all elements, lo = n, correctly "
            "indicating not found. The caller must check lo < n AND arr[lo] == "
            "target to confirm existence. This is the standard lower_bound / "
            "first_occurrence pattern (C++ std::lower_bound). Time: O(log n), "
            "space: O(1). Edge cases: empty array, target not present, multiple "
            "occurrences, target is first/last element."
        ),
        "gold_context": (
            "Information about binary search lower_bound variant: invariant is "
            "that answer (if exists) is in [lo, hi); lo == hi means search "
            "space is empty; returning n when target exceeds all elements is "
            "correct behavior; caller must verify lo < n and arr[lo] == target; "
            "standard lower_bound / first_occurrence pattern; C++ std::lower_bound "
            "equivalent; time complexity O(log n); space complexity O(1); "
            "edge cases include empty array, target not in array, target appears "
            "multiple times, target is first element, target larger than all."
        ),
    },
    {
        "id": "RAG011",
        "category": "graph_algorithms",
        "difficulty": "expert",
        "question": (
            "Explain how Dijkstra's algorithm maintains correctness through its "
            "greedy invariant. What happens to this invariant when edge weights "
            "can be negative?"
        ),
        "gold_answer": (
            "Dijkstra's algorithm maintains the greedy invariant: once a vertex "
            "is extracted from the min-priority queue (settled), its shortest "
            "distance is final and will never change. This holds because with "
            "non-negative edges, any alternative path to the settled vertex "
            "must go through an unsettled vertex with distance >= the settled "
            "vertex's distance, making it no better. With negative edges, this "
            "invariant breaks: a negative edge from an unsettled vertex can "
            "create a shorter path to an already-settled vertex. The algorithm "
            "never revisits settled vertices, so it misses these improvements. "
            "This is why Bellman-Ford is needed for graphs with negative edges."
        ),
        "gold_context": (
            "Information about Dijkstra's greedy invariant: once a vertex is "
            "settled (extracted from min-priority queue), its distance is final; "
            "with non-negative edges, alternative paths through unsettled vertices "
            "cannot be shorter; negative edges break this invariant by creating "
            "shorter paths to settled vertices; Dijkstra never revisits settled "
            "vertices; Bellman-Ford handles negative edges by allowing multiple "
            "relaxations; Dijkstra uses min-priority queue; O((V+E) log V) time; "
            "the invariant proof relies on triangle inequality with non-negative weights."
        ),
    },
    {
        "id": "RAG012",
        "category": "dynamic_programming",
        "difficulty": "expert",
        "question": (
            "What are the two key properties that make a problem suitable for "
            "dynamic programming? Explain with examples from knapsack and LIS."
        ),
        "gold_answer": (
            "The two key properties are: (1) Optimal substructure — the optimal "
            "solution contains optimal solutions to subproblems. (2) Overlapping "
            "subproblems — the same subproblems are solved multiple times in "
            "recursion. For 0/1 knapsack: optimal substructure because dp[i][w] "
            "depends on dp[i-1][w] and dp[i-1][w-weight[i]]; overlapping because "
            "the same (i,w) state is reached through different inclusion/exclusion "
            "paths. For LIS: optimal substructure because LIS ending at index i "
            "depends on LIS ending at earlier indices j < i where arr[j] < arr[i]; "
            "overlapping because multiple longer subsequences extend the same "
            "shorter ones. DP memoizes these overlapping subproblems."
        ),
        "gold_context": (
            "Information about dynamic programming properties: optimal substructure "
            "means optimal solution contains optimal subproblem solutions; overlapping "
            "subproblems means same subproblems solved multiple times in recursion; "
            "0/1 knapsack optimal substructure: dp[i][w] depends on dp[i-1][w] and "
            "dp[i-1][w-weight[i]]; 0/1 knapsack overlapping: same (i,w) state reached "
            "through different paths; LIS optimal substructure: LIS at index i depends "
            "on LIS at earlier indices j where arr[j] < arr[i]; LIS overlapping: "
            "multiple subsequences extend same shorter ones; DP solves by memoization "
            "or tabulation; DP eliminates redundant computation."
        ),
    },
    {
        "id": "RAG013",
        "category": "data_structures",
        "difficulty": "intermediate",
        "question": (
            "How does a Trie (prefix tree) enable efficient string operations? "
            "What are its time and space complexity characteristics?"
        ),
        "gold_answer": (
            "A Trie is a tree where each node represents a character, and the path "
            "from root to a node forms a prefix. Operations: insert O(m), search "
            "O(m), prefix search O(m) where m = string length (independent of "
            "number of stored strings). Space: O(ALPHABET_SIZE * n * m) in worst "
            "case where n = number of strings, m = max length. Optimizations: "
            "compressed tries (Patricia trees) reduce space. Tries excel at "
            "prefix operations (autocomplete, longest common prefix), dictionary "
            "operations, and string matching. Unlike hash tables, Tries provide "
            "lexicographic ordering and prefix-based queries. Each node has "
            "ALPHABET_SIZE children pointers (or a map for sparse children)."
        ),
        "gold_context": (
            "Information about Trie data structure: tree where each node represents "
            "a character; path from root forms a prefix; insert O(m) where m = "
            "string length; search O(m); prefix search O(m); time independent of "
            "number of stored strings; space O(ALPHABET_SIZE * n * m) worst case; "
            "n = number of strings, m = max length; compressed tries (Patricia "
            "trees) reduce space; excellent for autocomplete and prefix queries; "
            "provides lexicographic ordering; each node has ALPHABET_SIZE children "
            "or a map for sparse children; marks end-of-word nodes."
        ),
    },
    {
        "id": "RAG014",
        "category": "sorting",
        "difficulty": "intermediate",
        "question": (
            "Why is merge sort considered stable while quick sort is not? How "
            "does stability affect sorting with composite keys?"
        ),
        "gold_answer": (
            "Merge sort is stable because when merging two sorted subarrays, if "
            "elements are equal, we always take from the left subarray first, "
            "preserving original order. Quick sort is not inherently stable "
            "because partitioning swaps elements arbitrarily — equal elements "
            "can change relative order during partition swaps. Stability matters "
            "for composite key sorting: to sort by (name, age), first sort by "
            "age (stable), then sort by name (stable) — result is sorted by "
            "name then age. This is radix sort for multiple keys. Without "
            "stability, the secondary key sort would destroy the primary key "
            "ordering. Merge sort: O(n log n) time, O(n) space, stable. "
            "Quick sort: O(n log n) avg, O(n^2) worst, typically unstable."
        ),
        "gold_context": (
            "Information about sort stability: merge sort is stable because "
            "merging takes from left subarray first on equality; quick sort is "
            "unstable because partitioning swaps change relative order of equal "
            "elements; stability preserves original order of equal elements; "
            "composite key sorting requires stability: sort by secondary key "
            "first (stable), then primary key (stable); radix sort for multiple "
            "keys uses stable sorts; without stability, secondary ordering is "
            "destroyed; merge sort O(n log n) time, O(n) space, stable; quick "
            "sort O(n log n) average, O(n^2) worst case, typically unstable."
        ),
    },
    {
        "id": "RAG015",
        "category": "complexity",
        "difficulty": "expert",
        "question": (
            "Solve T(n) = 2T(n/2) + n/log(n). Why doesn't the Master Theorem "
            "directly apply? Give a tight asymptotic bound."
        ),
        "gold_answer": (
            "Master Theorem doesn't directly apply because f(n) = n/log(n) is "
            "not polynomially related to n^(log_b(a)) = n^1 = n. f(n) = n * "
            "log^(-1)(n) which is smaller than n by log(n) factor, but not by "
            "n^epsilon (polynomial factor). Using recursion tree: level i has "
            "2^i subproblems of size n/2^i, each contributing (n/2^i)/log(n/2^i) "
            "= (n/2^i)/(log n - i). Total at level i: n/(log n - i). There are "
            "log n levels. Sum: T(n) = sum_{i=0}^{log n-1} n/(log n - i) = "
            "n * sum_{j=1}^{log n} 1/j = n * H(log n) = n * Theta(log(log n)) "
            "= Theta(n log log n). So T(n) = Theta(n log log n)."
        ),
        "gold_context": (
            "Information about solving T(n) = 2T(n/2) + n/log(n): Master Theorem "
            "doesn't apply because f(n) = n/log(n) is not polynomially related "
            "to n^1; f(n) = n * log^(-1)(n) which is log^(-1)(n) not log^k(n) "
            "for k >= 0; recursion tree has log n levels; level i has 2^i "
            "subproblems; each contributes n/(2^i * (log n - i)); total per "
            "level is n/(log n - i); sum gives n * H(log n); harmonic series "
            "H(log n) = Theta(log(log n)); final answer Theta(n log log n); "
            "this is a gap case between Master Theorem cases 1 and 2."
        ),
    },
    {
        "id": "RAG016",
        "category": "complexity",
        "difficulty": "expert",
        "question": (
            "What are the three cases of the Master Theorem for recurrences of "
            "the form T(n) = aT(n/b) + f(n)? When does each case apply?"
        ),
        "gold_answer": (
            "For T(n) = aT(n/b) + f(n) where a >= 1, b > 1: Let "
            "n^(log_b(a)) be the critical exponent. Case 1: If f(n) = "
            "O(n^(log_b(a) - epsilon)) for some epsilon > 0, then "
            "T(n) = Theta(n^(log_b(a))). The work is dominated by the leaves. "
            "Case 2: If f(n) = Theta(n^(log_b(a)) * log^k(n)) for k >= 0, "
            "then T(n) = Theta(n^(log_b(a)) * log^(k+1)(n)). Work is equal "
            "at each level. Case 3: If f(n) = Omega(n^(log_b(a) + epsilon)) "
            "for epsilon > 0, AND af(n/b) <= cf(n) for c < 1 and large n "
            "(regularity condition), then T(n) = Theta(f(n)). The work is "
            "dominated by the root. Example: T(n) = 2T(n/2) + n => Case 2, "
            "T(n) = Theta(n log n)."
        ),
        "gold_context": (
            "Information about Master Theorem cases: recurrence form T(n) = "
            "aT(n/b) + f(n) with a >= 1, b > 1; critical exponent is "
            "n^(log_b(a)); Case 1: f(n) = O(n^(log_b(a) - epsilon)), "
            "T(n) = Theta(n^(log_b(a))), work dominated by leaves; Case 2: "
            "f(n) = Theta(n^(log_b(a)) * log^k(n)) for k >= 0, "
            "T(n) = Theta(n^(log_b(a)) * log^(k+1)(n)), equal work per level; "
            "Case 3: f(n) = Omega(n^(log_b(a) + epsilon)) with regularity "
            "condition af(n/b) <= cf(n), T(n) = Theta(f(n)), work dominated "
            "by root; example T(n) = 2T(n/2) + n is Case 2, Theta(n log n)."
        ),
    },
    {
        "id": "RAG017",
        "category": "string_algorithms",
        "difficulty": "expert",
        "question": (
            "Explain how the KMP (Knuth-Morris-Pratt) algorithm achieves O(n+m) "
            "string matching. What is the prefix function and how is it computed?"
        ),
        "gold_answer": (
            "KMP achieves O(n+m) by preprocessing the pattern to build a prefix "
            "function (also called pi or failure function). The prefix function "
            "pi[i] stores the length of the longest proper prefix of pattern[0..i] "
            "that is also a suffix of pattern[0..i]. During matching, when a "
            "mismatch occurs at pattern[j], instead of moving the pattern start "
            "back (as in naive matching), KMP uses pi[j-1] to determine the new "
            "position to compare. This avoids re-examining characters already "
            "matched. The prefix function is computed in O(m) time using a "
            "self-matching approach. Matching then takes O(n) since the pattern "
            "index never decreases. Total: O(n+m) where n = text length, m = "
            "pattern length."
        ),
        "gold_context": (
            "Information about KMP algorithm: achieves O(n+m) string matching; "
            "preprocesses pattern to build prefix function pi; pi[i] = length "
            "of longest proper prefix of pattern[0..i] that is also a suffix; "
            "during mismatch at pattern[j], use pi[j-1] to shift pattern without "
            "backtracking text; pattern index never decreases during matching; "
            "prefix function computed in O(m) via self-matching; matching takes "
            "O(n); total time O(n+m); n = text length, m = pattern length; "
            "avoids O(n*m) worst case of naive matching; pi[0] = 0 always."
        ),
    },
    {
        "id": "RAG018",
        "category": "trees",
        "difficulty": "intermediate",
        "question": (
            "How do you find the diameter of a binary tree in O(n) time? Explain "
            "the recursive approach and what information each node returns."
        ),
        "gold_answer": (
            "The diameter of a binary tree is the number of edges on the longest "
            "path between any two nodes. O(n) approach: for each node, compute "
            "the height of left and right subtrees. The diameter through this "
            "node = left_height + right_height. Track the maximum across all "
            "nodes. Each node returns its height = 1 + max(left_height, "
            "right_height). The recursive function computes both height (returned) "
            "and diameter (tracked as global max). For a leaf: height = 0, "
            "diameter = 0. For a node with one child: height = 1 + child_height, "
            "diameter = max(child_diameter, 1 + child_height). Time: O(n) since "
            "each node visited once. Space: O(h) for recursion stack where h = tree height."
        ),
        "gold_context": (
            "Information about binary tree diameter: diameter = longest path "
            "between any two nodes (measured in edges); O(n) approach computes "
            "height of left and right subtrees for each node; diameter through "
            "node = left_height + right_height; track maximum across all nodes; "
            "each node returns height = 1 + max(left_height, right_height); "
            "leaf height = 0, diameter = 0; time complexity O(n) — each node "
            "visited once; space complexity O(h) for recursion stack; h = tree "
            "height; for balanced tree O(log n), for skewed tree O(n)."
        ),
    },
    {
        "id": "RAG019",
        "category": "data_structures",
        "difficulty": "expert",
        "question": (
            "Design an LRU (Least Recently Used) cache with O(1) get and put "
            "operations. What data structures enable this?"
        ),
        "gold_answer": (
            "LRU cache requires O(1) get and put using a combination of: "
            "(1) Hash map (dictionary) for O(1) key lookup, mapping keys to "
            "cache entries. (2) Doubly linked list for O(1) move-to-front and "
            "eviction. Most recently used items are at the head; least recently "
            "used are at the tail. On get(key): lookup in hash map O(1), move "
            "node to head of DLL O(1). On put(key, value): if key exists, "
            "update value and move to head O(1). If new key and cache is full, "
            "remove tail node (LRU) O(1), remove from hash map O(1), add new "
            "node at head O(1), add to hash map O(1). Capacity constraint: "
            "evict least recently used when full. Time: O(1) for both operations. "
            "Space: O(capacity) for storing at most capacity entries."
        ),
        "gold_context": (
            "Information about LRU cache design: requires O(1) get and put "
            "operations; uses hash map for O(1) key-to-node lookup; uses "
            "doubly linked list for O(1) move-to-front and tail eviction; "
            "most recently used at head of DLL; least recently used at tail; "
            "get(key): hash map lookup O(1), move to head O(1); put(key, value): "
            "if exists update and move to head O(1); if new and full, evict "
            "tail O(1), add to head O(1); capacity constraint with LRU eviction; "
            "time complexity O(1) for both operations; space O(capacity); "
            "doubly linked list enables O(1) removal from arbitrary position."
        ),
    },
    {
        "id": "RAG020",
        "category": "graph_algorithms",
        "difficulty": "expert",
        "question": (
            "Compare Prim's and Kruskal's algorithms for Minimum Spanning Tree. "
            "When should each be preferred based on graph density?"
        ),
        "gold_answer": (
            "Both find MST but use different approaches. Prim's: grows a single "
            "tree from a starting vertex, always adding the minimum weight edge "
            "connecting the tree to a new vertex. Uses min-priority queue. "
            "Time: O((V+E) log V) with binary heap, O(E + V log V) with "
            "Fibonacci heap. Best for dense graphs (E close to V^2). Kruskal's: "
            "sorts all edges by weight, adds edges in order if they don't create "
            "a cycle (using Union-Find/DSU). Time: O(E log E) = O(E log V) "
            "for sorting, nearly linear for Union-Find operations. Best for "
            "sparse graphs (E close to V). Both produce same total weight MST "
            "(unique if edge weights distinct). Prim's is vertex-centric, "
            "Kruskal's is edge-centric."
        ),
        "gold_context": (
            "Information about MST algorithms comparison: Prim's grows single "
            "tree from starting vertex; adds minimum weight edge connecting "
            "tree to new vertex; uses min-priority queue; O((V+E) log V) with "
            "binary heap, O(E + V log V) with Fibonacci heap; best for dense "
            "graphs; Kruskal's sorts all edges by weight; adds edges if no "
            "cycle (Union-Find/DSU); O(E log E) = O(E log V) for sorting; "
            "nearly linear Union-Find operations; best for sparse graphs; "
            "both produce same total weight MST; unique MST if edge weights "
            "distinct; Prim's is vertex-centric, Kruskal's is edge-centric."
        ),
    },
]


def get_dsa_retrieval_dataset() -> List[Dict[str, str]]:
    """Return the DSA retrieval benchmark dataset.

    Returns
    -------
    list[dict]
        List of question dicts with keys: id, category, difficulty,
        question, gold_answer, gold_context.
    """
    return DSA_RETRIEVAL_DATASET


# ---------------------------------------------------------------------------
# RAGAS Evaluation Runner
# ---------------------------------------------------------------------------


class RAGASRetrievalBenchmark:
    """Run RAGAS-based retrieval evaluation on the DSA benchmark dataset.

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

    def _run_retrieval(self, query: str) -> List[str]:
        """Run retrieval for a query and return list of paragraph texts.

        Parameters
        ----------
        query : str
            The user query.

        Returns
        -------
        list[str]
            List of retrieved paragraph content strings.
        """
        result = self._retriever.retrieve(query, knee_enabled=True)
        contexts = []
        for para in result.paragraphs:
            content = getattr(para, "content", "") or ""
            if content.strip():
                contexts.append(content.strip())
        return contexts

    def _check_ragas_available(self) -> bool:
        """Check if ragas package is available.

        Returns
        -------
        bool
            True if ragas is installed.
        """
        try:
            import ragas  # noqa: F401
            from ragas import evaluate  # noqa: F401
            from ragas.metrics import (
                answer_faithfulness,
                context_precision,
                context_recall,
            )
            return True
        except ImportError:
            return False

    def run(
        self,
        dataset: Optional[List[Dict[str, str]]] = None,
        save_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run the full RAGAS retrieval benchmark.

        For each question in the dataset:
        1. Run retrieval on the question
        2. Compute RAGAS metrics (context precision, recall, faithfulness)
        3. Collect per-question and aggregate results

        Parameters
        ----------
        dataset : list[dict] or None
            Custom dataset. Uses DSA_RETRIEVAL_DATASET if None.
        save_path : str or None
            Path to save results.

        Returns
        -------
        dict
            Benchmark results with per-question scores and aggregate metrics.
        """
        if dataset is None:
            dataset = get_dsa_retrieval_dataset()

        has_ragas = self._check_ragas_available()

        if not has_ragas:
            logger.warning("ragas not installed. Running in lightweight mode.")
            return self._run_lightweight(dataset, save_path)

        return self._run_full_ragas(dataset, save_path)

    def _run_lightweight(
        self,
        dataset: List[Dict[str, str]],
        save_path: Optional[str],
    ) -> Dict[str, Any]:
        """Run retrieval benchmark without RAGAS (lightweight mode).

        Computes custom precision/recall metrics based on keyword overlap
        between retrieved context and gold context.

        Parameters
        ----------
        dataset : list[dict]
            Benchmark questions.
        save_path : str or None
            Output path.

        Returns
        -------
        dict
            Benchmark results.
        """
        results: List[Dict[str, Any]] = []
        all_context_precision = []
        all_context_recall = []
        all_faithfulness = []

        for i, q in enumerate(dataset):
            start = time.time()
            query = q["question"]
            gold_context = q["gold_context"]
            gold_answer = q["gold_answer"]

            # Run retrieval
            retrieved_contexts = self._run_retrieval(query)
            retrieved_text = " ".join(retrieved_contexts)

            # Lightweight context precision: fraction of gold context concepts
            # found in retrieved text
            cp = _compute_lightweight_precision(gold_context, retrieved_text)
            all_context_precision.append(cp)

            # Lightweight context recall: fraction of retrieved info covered
            # by gold context concepts
            cr = _compute_lightweight_recall(gold_context, retrieved_text)
            all_context_recall.append(cr)

            # Lightweight faithfulness: check if answer claims appear in context
            faith = _compute_lightweight_faithfulness(gold_answer, retrieved_text)
            all_faithfulness.append(faith)

            latency = time.time() - start

            result = {
                "question_id": q["id"],
                "category": q.get("category", ""),
                "difficulty": q.get("difficulty", ""),
                "question": query,
                "retrieved_paragraphs": len(retrieved_contexts),
                "retrieved_tokens": sum(len(c) for c in retrieved_contexts),
                "scores": {
                    "context_precision": round(cp, 4),
                    "context_recall": round(cr, 4),
                    "faithfulness": round(faith, 4),
                },
                "latency": round(latency, 4),
            }
            results.append(result)

            logger.info(
                "[%d/%d] %s: cp=%.3f, cr=%.3f, faith=%.3f, paras=%d, latency=%.2fs",
                i + 1, len(dataset), q["id"], cp, cr, faith,
                len(retrieved_contexts), latency,
            )

        aggregate = _compute_aggregate_metrics(
            all_context_precision, all_context_recall, all_faithfulness
        )

        output = {
            "benchmark": "ragas_retrieval",
            "mode": "lightweight",
            "dataset_size": len(dataset),
            "aggregate_metrics": aggregate,
            "per_question": results,
        }

        if save_path:
            self._save_results(output, save_path)

        return output

    def _run_full_ragas(
        self,
        dataset: List[Dict[str, str]],
        save_path: Optional[str],
    ) -> Dict[str, Any]:
        """Run full RAGAS evaluation using the ragas library.

        Parameters
        ----------
        dataset : list[dict]
            Benchmark questions.
        save_path : str or None
            Output path.

        Returns
        -------
        dict
            Benchmark results with RAGAS scores.
        """
        from ragas import evaluate
        from ragas.datasets import Dataset
        from ragas.metrics import (
            answer_faithfulness,
            context_precision,
            context_recall,
        )

        # Build RAGAS-compatible samples
        samples = []
        for q in dataset:
            retrieved = self._run_retrieval(q["question"])
            samples.append({
                "question": q["question"],
                "answer": q["gold_answer"],
                "contexts": retrieved,
                "ground_truth": q["gold_context"],
            })

        # Create RAGAS dataset
        ragas_dataset = Dataset.from_list(samples)

        # Evaluate
        metrics = [context_precision, context_recall, answer_faithfulness]
        evaluation_result = evaluate(ragas_dataset, metrics=metrics)

        # Convert to our format
        df = evaluation_result.to_pandas()

        results: List[Dict[str, Any]] = []
        for i, q in enumerate(dataset):
            row = df.iloc[i] if i < len(df) else {}
            results.append({
                "question_id": q["id"],
                "category": q.get("category", ""),
                "difficulty": q.get("difficulty", ""),
                "question": q["question"],
                "retrieved_paragraphs": len(samples[i]["contexts"]),
                "scores": {
                    "context_precision": float(row.get("context_precision", 0.0))
                    if not pd_isna(row.get("context_precision"))
                    else 0.0,
                    "context_recall": float(row.get("context_recall", 0.0))
                    if not pd_isna(row.get("context_recall"))
                    else 0.0,
                    "faithfulness": float(row.get("answer_faithfulness", 0.0))
                    if not pd_isna(row.get("answer_faithfulness"))
                    else 0.0,
                },
            })

        aggregate = {
            "context_precision": float(df["context_precision"].mean())
            if "context_precision" in df.columns
            else 0.0,
            "context_recall": float(df["context_recall"].mean())
            if "context_recall" in df.columns
            else 0.0,
            "faithfulness": float(df["answer_faithfulness"].mean())
            if "answer_faithfulness" in df.columns
            else 0.0,
        }

        output = {
            "benchmark": "ragas_retrieval",
            "mode": "full_ragas",
            "dataset_size": len(dataset),
            "aggregate_metrics": aggregate,
            "per_question": results,
        }

        if save_path:
            self._save_results(output, save_path)

        return output

    @staticmethod
    def _save_results(output: Dict[str, Any], path: str) -> None:
        """Save benchmark results to JSON.

        Parameters
        ----------
        output : dict
            Benchmark results.
        path : str
            Output file path.
        """
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        with open(path_obj, "w", encoding="utf-8") as fh:
            json.dump(output, fh, indent=2, ensure_ascii=False, default=str)
        logger.info("Saved RAGAS results to %s", path)


# ---------------------------------------------------------------------------
# Lightweight metric computations (fallback when ragas is not installed)
# ---------------------------------------------------------------------------

def pd_isna(value: Any) -> bool:
    """Check if a value is NaN/NA without importing pandas."""
    if value is None:
        return True
    try:
        return value != value  # NaN check
    except (TypeError, ValueError):
        return False


def _extract_concepts(text: str) -> set:
    """Extract meaningful concept keywords from text.

    Parameters
    ----------
    text : str
        Input text.

    Returns
    -------
    set
        Set of lowercase concept keywords.
    """
    import re

    # Common stop words to filter
    stop_words = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "shall", "can", "to", "of", "in", "for",
        "on", "with", "at", "by", "from", "as", "into", "through", "during",
        "but", "and", "or", "if", "then", "than", "so", "this", "that",
        "these", "those", "it", "its", "what", "which", "who", "how",
        "where", "when", "why", "not", "no", "nor", "all", "each", "every",
        "both", "few", "more", "most", "other", "some", "such", "only",
        "own", "same", "too", "very", "just", "because", "while", "about",
        "between", "after", "before", "above", "below", "up", "down",
        "out", "off", "over", "under", "again", "further",
    }

    # Extract alphanumeric tokens (words and numbers)
    tokens = re.findall(r"[a-z0-9_]+", text.lower())
    # Filter: length > 2, not stop words, not purely numeric
    concepts = set()
    for t in tokens:
        if len(t) > 2 and t not in stop_words and not t.isdigit():
            concepts.add(t)
    return concepts


def _compute_lightweight_precision(
    gold_context: str, retrieved_text: str
) -> float:
    """Compute lightweight context precision.

    Measures: of the concepts mentioned in the gold context (what SHOULD be
    retrieved), what fraction are found in the retrieved text?

    Higher precision = retrieved paragraphs contain the right information.

    Parameters
    ----------
    gold_context : str
        Description of what should be retrieved.
    retrieved_text : str
        Concatenation of retrieved paragraph texts.

    Returns
    -------
    float
        Precision score in [0.0, 1.0].
    """
    gold_concepts = _extract_concepts(gold_context)
    if not gold_concepts:
        return 1.0

    retrieved_concepts = _extract_concepts(retrieved_text)
    matched = gold_concepts & retrieved_concepts
    return len(matched) / len(gold_concepts)


def _compute_lightweight_recall(
    gold_context: str, retrieved_text: str
) -> float:
    """Compute lightweight context recall.

    Measures: of the concepts found in the retrieved text, what fraction
    are relevant (mentioned in gold context)?

    Higher recall = retrieved text is comprehensive and not noisy.

    Parameters
    ----------
    gold_context : str
        Description of what should be retrieved.
    retrieved_text : str
        Concatenation of retrieved paragraph texts.

    Returns
    -------
    float
        Recall score in [0.0, 1.0].
    """
    gold_concepts = _extract_concepts(gold_context)
    if not gold_concepts:
        return 1.0

    retrieved_concepts = _extract_concepts(retrieved_text)
    if not retrieved_concepts:
        return 0.0

    matched = gold_concepts & retrieved_concepts
    return len(matched) / len(retrieved_concepts)


def _compute_lightweight_faithfulness(
    gold_answer: str, retrieved_text: str
) -> float:
    """Compute lightweight faithfulness.

    Measures: do the key claims in the answer appear in the retrieved context?

    Higher faithfulness = answer is grounded in retrieved context.

    Parameters
    ----------
    gold_answer : str
        The reference answer.
    retrieved_text : str
        Concatenation of retrieved paragraph texts.

    Returns
    -------
    float
        Faithfulness score in [0.0, 1.0].
    """
    # Extract key claims from answer (sentences with technical content)
    import re

    sentences = re.split(r"[.!?]+\s*", gold_answer)
    claims = [s.strip() for s in sentences if 30 <= len(s) <= 400]

    if not claims:
        return 1.0

    # Check how many claims have keyword overlap with retrieved text
    retrieved_lower = retrieved_text.lower()
    supported = 0

    for claim in claims[:10]:  # limit to first 10 claims
        claim_lower = claim.lower()
        claim_keywords = _extract_concepts(claim)
        if claim_keywords and claim_keywords & _extract_concepts(retrieved_lower):
            supported += 1

    return supported / len(claims) if claims else 1.0


def _compute_aggregate_metrics(
    context_precisions: List[float],
    context_recalls: List[float],
    faithfulness_scores: List[float],
) -> Dict[str, float]:
    """Compute aggregate metrics from per-question scores.

    Parameters
    ----------
    context_precisions : list[float]
        Per-question context precision scores.
    context_recalls : list[float]
        Per-question context recall scores.
    faithfulness_scores : list[float]
        Per-question faithfulness scores.

    Returns
    -------
    dict
        Aggregate metrics with mean and standard deviation.
    """
    import statistics

    def _stats(values: List[float]) -> Dict[str, float]:
        if not values:
            return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "count": 0}
        mean = statistics.mean(values)
        std = statistics.stdev(values) if len(values) > 1 else 0.0
        return {
            "mean": round(mean, 4),
            "std": round(std, 4),
            "min": round(min(values), 4),
            "max": round(max(values), 4),
            "count": len(values),
        }

    return {
        "context_precision": _stats(context_precisions),
        "context_recall": _stats(context_recalls),
        "faithfulness": _stats(faithfulness_scores),
    }
