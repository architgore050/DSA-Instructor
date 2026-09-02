Absolutely. I would throw away the implementation details for now and define the system as a **technical specification / research blueprint** first.

The central idea is:

> **A DSA instructor that uses hierarchical retrieval to provide broad topical context when the query is broad, precise evidence when the query is precise, and can autonomously perform additional searches when the model determines that the retrieved evidence is insufficient.**

And the benchmark is deliberately designed to answer:

> **How much does this retrieval architecture actually add over the model's own parametric knowledge, especially as model capability decreases?**

Below is the blueprint I would work from.

---

# DSA Mentor RAG

## Complete Technical Blueprint

---

# 1. Project definition

## 1.1 Objective

Build a DSA tutoring system whose knowledge comes primarily from a deliberately small corpus of high-quality DSA books and algorithm references.

The system should be able to:

* teach DSA concepts
* explain algorithms
* explain correctness and invariants
* reason about complexity
* help solve LeetCode/interview-style problems
* give progressive hints
* explain incorrect approaches
* explain code
* connect related concepts
* answer questions requiring material distributed across several sections or books

The system should also make the retrieval process observable.

A user should be able to see:

```text
Question
    ↓
Selected books
    ↓
Selected chapters
    ↓
Selected topics
    ↓
Selected paragraph evidence
    ↓
LLM answer
```

The retrieval system should therefore be a **first-class part of the product**, rather than something hidden behind an LLM API.

---

# 2. The research question

The actual interesting question is not:

> "Can I build a RAG chatbot?"

That is trivial.

The research question is:

> **When does retrieval materially improve DSA reasoning, and how does the effect vary with model capability?**

The experimental matrix should be:

| Model    | RAG OFF | RAG ON |
| -------- | ------: | -----: |
| Small    |       A |      B |
| Medium   |       C |      D |
| Frontier |       E |      F |

Your hypothesis is approximately:

```text
Frontier:
    E ≈ F

Smaller model:
    A << B
```

But the project should **not assume this is true**.

That is exactly what the experiment is supposed to establish.

Possible outcomes:

### Outcome A

```text
Frontier OFF = 95
Frontier ON  = 96

Small OFF    = 57
Small ON     = 83
```

Excellent result.

### Outcome B

```text
Frontier OFF = 87
Frontier ON  = 91

Small OFF    = 64
Small ON     = 67
```

Also interesting. It would mean retrieval isn't rescuing the smaller model as effectively as expected.

### Outcome C

```text
Frontier OFF = 96
Frontier ON  = 86
```

Also useful.

It could expose:

* bad retrieval
* context overload
* irrelevant evidence
* poor chunking
* model distraction
* incorrect hierarchy

A good engineering project should allow itself to fail.

---

# 3. Knowledge-base philosophy

Do **not** attempt to create a giant internet-scale DSA corpus.

That would create several problems:

* difficult evaluation
* excessive redundancy
* harder attribution
* retrieval ambiguity
* huge indexing overhead
* impossible debugging
* harder explanation of why a particular passage was retrieved

Instead, deliberately construct a **small but deep knowledge base**.

Target:

```text
2–4 primary books/resources
+
~10–25 carefully selected supplementary articles
```

Roughly:

```text
5–20 MB
```

rather than gigabytes.

The point is not quantity.

The point is:

> **high-quality knowledge + strong document structure.**

---

# 4. Knowledge hierarchy

The conceptual document tree should be:

```text
CORPUS
│
├── BOOK
│   │
│   ├── CHAPTER
│   │   │
│   │   ├── TOPIC
│   │   │   │
│   │   │   ├── SUBTOPIC
│   │   │   │   │
│   │   │   │   ├── PARAGRAPH
│   │   │   │   ├── PARAGRAPH
│   │   │   │   └── PARAGRAPH
│   │   │   │
│   │   │   └── ...
│   │   │
│   │   └── ...
│   │
│   └── ...
│
└── ...
```

This hierarchy should be preserved in metadata.

Every paragraph should know:

```text
corpus_id
book_id
chapter_id
topic_id
subtopic_id
paragraph_id

book_title
chapter_title
topic_title
subtopic_title

source_file
source_url
page_number
license
```

Potentially also:

```text
section_number
paragraph_index
parent_topic_id
previous_paragraph_id
next_paragraph_id
```

Those relationships become extremely valuable later.

---

# 5. Why paragraph-level retrieval?

The lowest retrieval unit should be the paragraph because arbitrary token or character chunking throws away document structure.

Bad:

```text
paragraph A
paragraph B
paragraph C
paragraph D
```

becomes:

```text
chunk 1 = end of paragraph A + beginning of B
chunk 2 = middle of B + C
```

This can separate:

* definition from qualification
* theorem from proof
* algorithm from complexity
* invariant from explanation
* caveat from claim

For DSA, this is particularly dangerous.

For example:

```text
Dijkstra works by...
```

might be followed by:

```text
This requires all edge weights to be nonnegative.
```

Those two paragraphs belong together conceptually even if they are separate retrieval units.

Therefore:

> **Paragraph is the atomic evidence unit.**

---

# 6. Handling long paragraphs

Paragraphs can occasionally be huge.

Example:

```text
4,000-character paragraph
```

A gigantic paragraph is undesirable as a vector unit.

Therefore:

```text
if paragraph <= P:
    preserve paragraph

if paragraph > P:
    split paragraph into overlapping subparagraph chunks
```

For example:

```text
P = 1800 characters
overlap = 250 characters
```

But this should happen **only when necessary**.

The hierarchy remains:

```text
topic
    ↓
paragraph
    ↓
oversized paragraph segments
```

rather than treating all text as arbitrary chunks.

---

# 7. Higher-level aggregation

The key design decision is that **chapter/topic retrieval should not return little snippets**.

Suppose:

```text
Chapter:
Graph Algorithms

Topic:
Dijkstra's Algorithm
```

The topic representation should contain the full topic text:

```text
Dijkstra's Algorithm

definition...

intuition...

algorithm...

proof...

negative-weight limitation...

complexity...

implementation details...
```

Then the topic embedding describes the entire concept.

---

# 8. Multiple vector indices

Instead of one vector store, conceptually maintain four logical collections.

### Index 1 — Book index

Each vector represents an entire book or book-level synopsis.

```text
BOOK_VECTOR
```

Used to answer:

> Which sources are likely to contain the necessary knowledge?

---

### Index 2 — Chapter index

One vector per chapter.

```text
CHAPTER_VECTOR
```

Used to determine:

> Where inside the selected books should we search?

---

### Index 3 — Topic index

One vector per topic/subtopic.

```text
TOPIC_VECTOR
```

Used to determine:

> What conceptual unit is relevant?

---

### Index 4 — Paragraph index

One vector per paragraph.

```text
PARAGRAPH_VECTOR
```

Used for:

> precise evidence retrieval.

This is fundamentally different from a flat vector DB.

---

# 9. Retrieval architecture

The nominal pipeline is:

```text
                         USER QUERY
                              │
                              ▼
                    Query Understanding
                              │
                              ▼
                     Book Retrieval
                              │
                              ▼
                    Chapter Retrieval
                              │
                              ▼
                     Topic Retrieval
                              │
                    ┌─────────┴─────────┐
                    │                   │
                    ▼                   ▼
             Full Topic Text      Global Paragraph
                                  Vector Search
                    │                   │
                    └─────────┬─────────┘
                              ▼
                       Context Builder
                              │
                              ▼
                              LLM
```

But there is an important nuance:

**The number of paragraph chunks should not be fixed.**

---

# 10. Dynamic retrieval via the similarity-score curve

Suppose paragraph search returns:

```text
rank    similarity

1       0.91
2       0.89
3       0.88
4       0.86
5       0.85
6       0.84
7       0.62
8       0.60
9       0.59
10      0.58
```

The first six are plausibly relevant.

Then there is a large drop:

```text
0.84
 ↓
0.62
```

A fixed:

```text
top_k = 10
```

would unnecessarily expose four irrelevant chunks.

Instead, let retrieval determine:

> **Where does the relevant cluster end?**

---

# 11. Similarity score normalization

Let paragraph similarity scores be:

$$
s_1 \geq s_2 \geq \dots \geq s_n
$$

Assuming normalized embeddings and cosine/IP similarity:

$$
s_i \in [-1,1]
$$

or, in practical embedding systems:

$$
s_i \approx [0,1]
$$

We want to identify a discontinuity.

Define first difference:

$$
\Delta_i=s_i-s_{i+1}
$$

A large:

$$
\Delta_i
$$

means there is a drop between candidates \(i\) and \(i+1\).

---

# 12. Knee detection

A simple and robust method for this project is:

1. retrieve a reasonably large candidate pool, e.g. 30–50;
2. sort descending by similarity;
3. normalize rank and score;
4. determine the point of maximum curvature / largest normalized drop;
5. use that as the retrieval boundary;
6. enforce sensible minimum and maximum bounds.

Conceptually:

```text
similarity

1.0 ┤ ●
    │  ●
0.9 ┤    ●
    │      ●
0.8 ┤         ●
    │           ●
0.7 ┤
    │              ●
0.6 ┤                 ● ● ● ●
    │
    └───────────────────────────── rank
      1  2  3  4  5  6  7  8  ...
```

The knee occurs around:

```text
rank = 6
```

---

# 13. Important caveat about "the knee"

You should **not blindly use the mathematical knee as truth**.

A similarity curve can look like:

```text
0.91
0.90
0.89
0.88
0.87
0.86
0.85
0.84
0.83
```

There may be no meaningful cutoff.

Or:

```text
0.92
0.91
0.90
0.89
0.62
0.61
```

There is an obvious cutoff.

Therefore the dynamic retriever should implement:

```text
knee detection
+
minimum relevance threshold
+
minimum evidence count
+
maximum context budget
```

Example:

```text
candidate_k = 40

minimum_chunks = 3
maximum_chunks = 20

if no strong knee:
    retain chunks above relevance threshold
    capped at 20

if knee exists:
    retain candidates through knee

always:
    at least 3
```

---

# 14. Topic-aware expansion

This is where your architecture becomes especially interesting.

Suppose the user asks:

> "Explain Dijkstra's algorithm."

This is a **broad conceptual query**.

The paragraph search might identify:

```text
Dijkstra definition
Dijkstra implementation
Dijkstra correctness
Dijkstra complexity
Dijkstra limitations
```

But giving only 5 paragraphs isn't ideal.

Instead, topic retrieval identifies:

```text
Topic:
Dijkstra's Algorithm
```

The system expands that topic to the **entire topic text**.

So the model receives the complete conceptual unit.

---

# 15. Narrow query behavior

Now consider:

> "Why does Dijkstra fail when an edge has negative weight even if that edge doesn't lie on the final shortest path?"

This is highly specific.

Topic retrieval still identifies:

```text
Shortest Paths
→ Dijkstra
```

But paragraph retrieval finds only specific relevant evidence:

```text
settling invariant
negative-weight counterexample
greedy assumption
```

So the context becomes:

```text
Topic context:
    partial/relevant Dijkstra topic material

Paragraph evidence:
    3–5 highly specific paragraphs
```

rather than dumping the entire textbook section.

---

# 16. Query breadth classification

The system should internally estimate query breadth.

Not necessarily with another LLM.

It can use a lightweight heuristic/model.

Categories:

```text
BROAD
MODERATE
NARROW
```

Examples:

### Broad

> Explain dynamic programming.

> Teach me graph algorithms.

### Moderate

> Explain Dijkstra's algorithm and when to use it.

### Narrow

> Why does Dijkstra's settled-vertex invariant fail with negative edges?

Then:

```text
BROAD
    → full topic expansion

MODERATE
    → topic + paragraphs

NARROW
    → primarily paragraph evidence
```

This is conceptually important because:

> **retrieval depth should depend on query intent.**

---

# 17. Neighbor expansion

Paragraph retrieval should have one additional mechanism.

If paragraph \(p_i\) is selected, retrieve its local neighbors:

```text
p(i-1)
p(i)
p(i+1)
```

but only when they belong to the same topic.

This handles cases where:

```text
paragraph 7:
    algorithm description

paragraph 8:
    proof

paragraph 9:
    complexity
```

and only paragraph 8 has high similarity.

The user asked about correctness, but paragraph 7 contains the algorithm definition required to understand the proof.

Thus:

```text
selected evidence
+
local structural context
```

---

# 18. Preventing context explosion

Dynamic retrieval should never become:

> "Broad query → give the model the entire 30-page textbook."

Therefore establish a hard token/character budget.

For example:

```text
maximum context = 16k tokens
```

or perhaps lower depending on model.

The retrieval system should optimize:

$$
\text{relevance per token}
$$

rather than:

$$
\text{number of retrieved chunks}
$$

---

# 19. Retrieval deduplication

Different retrieval levels may return the same information.

For example:

```text
Topic = Dijkstra
Paragraph = Dijkstra definition
```

If the full topic already contains the paragraph, don't send both verbatim.

Therefore the context builder needs:

```text
document identity
topic identity
paragraph identity
text-overlap detection
```

and should eliminate redundant context.

---

# 20. Source diversity

A query can accidentally produce:

```text
paragraph 1 → same book
paragraph 2 → same book
paragraph 3 → same book
paragraph 4 → same book
paragraph 5 → same book
```

That may be acceptable.

But multi-source corroboration can be useful.

Therefore metadata should allow optional diversity control:

```text
max paragraphs per source
```

But **do not force diversity blindly**.

If only one source contains the correct explanation, artificial source diversity would make retrieval worse.

Use source diversity as a soft preference, not a hard requirement.

---

# 21. LLM retrieval tool

This is the second major extension.

Normal RAG:

```text
query
 ↓
retrieval
 ↓
LLM
```

Tool-enabled RAG becomes:

```text
query
 ↓
initial retrieval
 ↓
LLM
 │
 ├── answer
 │
 └── tool call
       ↓
    vector search
       ↓
    evidence
       ↓
      LLM
```

The model itself can ask:

> "Search the corpus for the invariant used in Dijkstra's correctness proof."

This gives the LLM a second retrieval mechanism beyond the application's initial search.

---

# 22. Retrieval tool interface

Conceptually expose:

```text
search_knowledge(
    query: string,
    scope?: string,
    max_results?: integer
)
```

Potential scope values:

```text
all
book
chapter
topic
```

Optional parameters:

```text
book
chapter
topic
```

Example:

```json
{
  "query": "Dijkstra correctness invariant nonnegative edge weights",
  "scope": "all",
  "max_results": 8
}
```

The tool should return structured metadata:

```json
{
  "results": [
    {
      "source": "...",
      "book": "...",
      "chapter": "...",
      "topic": "...",
      "paragraph": "...",
      "similarity": 0.91
    }
  ]
}
```

---

# 23. Tool-call loop

The model should be allowed a small number of retrieval calls.

Example:

```text
MAX_TOOL_CALLS = 3
```

Loop:

```text
User question
       ↓
Initial retrieved context
       ↓
LLM
       ↓
Does it need more evidence?
      / \
    yes  no
    │     │
    ▼     ▼
tool     answer
call
    │
    ▼
new evidence
    │
    ▼
LLM
```

Eventually:

```text
answer
```

or:

```text
I don't have enough information in the knowledge base.
```

---

# 24. Why tool-based retrieval is useful

Suppose the initial query is:

> "Compare Dijkstra and A*."

Initial retrieval might produce:

```text
Dijkstra
shortest paths
```

The model realizes:

> I need information about the heuristic admissibility property of A*.

It can then issue:

```text
search_knowledge(
    "A* admissible heuristic consistency optimality"
)
```

The model is now performing **active retrieval**.

This matters particularly for multi-hop questions.

---

# 25. Tool-call guardrails

The model should not be able to search infinitely.

Rules:

```text
max tool calls = 3
```

Each call should be logged.

Tool output must remain inside the model context.

The model must distinguish:

```text
parametric knowledge
```

from:

```text
retrieved knowledge
```

The system prompt should explicitly say:

> You have access to a DSA knowledge retrieval tool. Use it when the supplied evidence is insufficient, when a claim needs verification, or when the question spans concepts that initial retrieval does not adequately cover.

---

# 26. System prompt philosophy

The system prompt should NOT say:

> You must answer entirely from the provided documents.

That creates a strange artificial limitation.

Instead:

> You are a DSA instructor with access to a curated DSA knowledge base. Use retrieved evidence to ground source-dependent claims. You may reason using general algorithmic knowledge, but do not pretend that unsupported claims came from the corpus.

This allows the crucial experiment:

```textRAG OFF
```

to represent:

> model using its own knowledge

while:

```textRAG ON
```

represents:

> model + knowledge retrieval.

---

# 27. Instructor behavior

The assistant should optimize for learning, not merely answer production.

For a problem:

```text
1. Identify the structural clue.
2. Identify candidate technique.
3. State invariant/intuition.
4. Establish correctness.
5. Analyze complexity.
6. Consider edge cases.
7. Give implementation guidance.
```

It should not immediately dump code unless asked.

---

# 28. Interaction modes

Three primary UI modes are enough.

## Learn

```text
"Explain segment trees."
```

The model gives a conceptual explanation.

---

## Hint

```text
"I'm stuck on this LeetCode problem."
```

Model should reveal progressively more.

Potential hint levels:

```text
Hint 1:
What property of the input could you exploit?

Hint 2:
The array is sorted, so consider...

Hint 3:
Try maintaining two pointers...

Hint 4:
Here is the full approach.
```

---

## Explain

Student submits:

```python
code...
```

The model analyzes:

* correctness
* bug
* invariant
* complexity
* alternatives

---

# 29. Chat UI

The interface should look roughly like:

```text
┌──────────────────────────────────────────────────────────┐
│ DSA Mentor                               🟢 RAG ENABLED   │
├──────────────────────────────────────────────────────────┤
│                                                          │
│ User                                                    │
│ Why can't Dijkstra handle negative edges?               │
│                                                          │
│ DSA Mentor                                               │
│ The issue is not merely that a negative edge can...     │
│                                                          │
│ ┌──────────────────────────────────────────────────────┐ │
│ │ Sources                                               │ │
│ │ Open Data Structures → Graphs → Shortest Paths       │ │
│ │ CP Algorithms → Dijkstra                             │ │
│ └──────────────────────────────────────────────────────┘ │
│                                                          │
│ Ask a DSA question...                          [ Send ]  │
└──────────────────────────────────────────────────────────┘
```

---

# 30. Retrieval inspector

The user should be able to expand:

```text
Retrieval Inspector
```

and see:

```text
Query
────────────────────────
"Why does Dijkstra fail..."

Books
────────────────────────
1. Open Data Structures       0.84
2. CP Algorithms              0.79

Chapters
────────────────────────
Graphs / Shortest Paths       0.91

Topics
────────────────────────
Dijkstra's Algorithm          0.94

Paragraph retrieval
────────────────────────
S1  0.93
S2  0.91
S3  0.88
S4  0.61  ← excluded by knee

Knee
────────────────────────
selected through rank 3
```

This makes the project **demonstrable**.

---

# 31. Important distinction: retrieval transparency

The UI should distinguish:

```text
retrieved evidence
```

from:

```text
LLM-generated explanation
```

This is important because otherwise a judge can't tell whether:

> "This model gave the right answer"

because:

1. the model knew the answer,
2. retrieval found it,
3. or the retrieval actually caused the answer.

Your inspector makes the causal mechanism visible.

---

# 32. Evaluation architecture

Do not mix the interactive tutor and benchmark.

They share:

```text
retrieval engine
LLM client
knowledge base
configuration
```

but evaluation remains independent.

The benchmark invokes the system like:

```text
question
   ↓
rag enabled?
   ↓
retrieval or no retrieval
   ↓
answer
   ↓
scoring
```

The UI is merely another consumer.

---

# 33. Critical experimental rule

For every model:

### RAG OFF

The model must receive:

```text
question
+
system prompt
```

and nothing from the knowledge base.

### RAG ON

The model receives:

```text
question
+
system prompt
+
retrieved knowledge
```

and possibly tool retrieval.

Nothing else should change.

---

# 34. Keep the configuration centralized

Something like:

```yaml
experiment:

  rag_enabled: true

  model:
    name: ...
    endpoint: ...

  retrieval:
    candidate_k: 40
    minimum_chunks: 3
    maximum_chunks: 20
    knee_method: ...
    similarity_threshold: ...
    neighbor_window: 1
    max_context_tokens: ...

  agentic_retrieval:
    enabled: true
    max_tool_calls: 3

  evaluation:
    dataset: ...
```

But your key switch remains:

```yaml
rag_enabled: false
```

---

# 35. Evaluation dataset philosophy

The benchmark should not be:

```text
What is a stack?
What is BFS?
What is a queue?
```

That's useless.

A frontier model will obviously ace it.

Instead, questions should require **reasoning around familiar algorithms**.

---

# 36. Difficulty dimensions

Every question should have metadata.

```json
{
  "id": "Q001",
  "category": "shortest_paths",
  "difficulty": "expert",
  "requires": [
    "Dijkstra",
    "greedy correctness",
    "counterexample"
  ]
}
```

Possible dimensions:

```text
concept recall
algorithm selection
invariant reasoning
proof
counterexample
complexity
implementation
edge cases
multi-hop
adversarial
```

---

# 37. Benchmark categories

Target approximately 10–15 categories.

### Algorithms

* sorting
* searching
* hashing
* trees
* heaps
* graphs
* shortest paths
* MST
* flow
* dynamic programming
* greedy
* strings
* range structures
* randomized algorithms
* amortized analysis
* complexity theory

---

# 38. Hard-question patterns

This is more important than simply making questions long.

A good hard question should attack assumptions.

Examples:

### Hidden assumption

> BFS finds shortest paths.

Question:

> Under exactly what edge-weight assumption?

---

### Plausible but false claim

> "Dijkstra fails only when the negative edge appears on the final shortest path."

Ask for a counterexample.

---

### Implementation trap

> Binary search returns the right answer on ordinary inputs but hangs for a one-element interval.

Ask for the invariant violation.

---

### Complexity trap

> Build heap is O(n log n) because heapify is O(log n).

Ask why that's wrong.

---

### Transfer

> When does an algorithmic idea from unweighted graphs survive after adding weights?

---

# 39. Multi-hop questions

These are especially valuable for RAG.

Example:

> "Why can't I replace BFS with Dijkstra here, and under what weight restrictions would a specialized shortest-path algorithm work?"

This requires:

```text
BFS
+
Dijkstra
+
0/1 BFS
```

A flat single-concept question is less informative.

---

# 40. Adversarial variants

Take common statements and slightly modify them.

Base:

> Dijkstra requires nonnegative edges.

Adversarial:

> "If a graph contains a negative edge that cannot lie on any shortest path from the source, can Dijkstra safely ignore it?"

This catches superficial memorization.

---

# 41. Gold answers

Every benchmark item should have:

```text
gold_answer
```

but also:

```text
required_claims
forbidden_claims
complexity
edge_cases
algorithm
```

Example:

```json
{
  "gold_answer": "...",
  "required_claims": [
    "Dijkstra requires nonnegative edge weights",
    "settled vertex cannot later be improved",
    "negative edges violate that assumption"
  ],
  "forbidden_claims": [
    "negative edges merely make Dijkstra slower"
  ],
  "complexity": "O((V+E) log V)"
}
```

This gives the evaluator structure rather than relying entirely on semantic similarity to the gold answer.

---

# 42. Do not rely on one metric

A RAG benchmark needs multiple metrics.

## Retrieval

### Recall@K

Did we retrieve the relevant evidence?

$$
Recall@K =
\frac{\text{relevant retrieved evidence}}
{\text{all relevant evidence}}
$$

---

# 43. Retrieval precision

Also useful:

$$
Precision@K=
\frac{\text{relevant retrieved evidence}}
{\text{retrieved evidence}}
$$

This matters because your dynamic knee system is specifically trying to eliminate irrelevant chunks.

---

# 44. Generation correctness

Score:

```text
0 = wrong
1 = major errors
2 = partially correct
3 = correct
4 = fully correct
```

This can be evaluated using a fixed judge model.

But the judge receives:

```text
question
gold answer
candidate answer
```

and should not have access to the model that generated the answer.

---

# 45. Groundedness

For RAG ON:

Ask:

> Are the source-dependent claims supported by retrieved evidence?

This is separate from correctness.

An answer can be:

```text
correct
but ungrounded
```

because the model knew it from its own weights.

That distinction is central to your experiment.

---

# 46. Parametric-vs-retrieved knowledge analysis

This is potentially your most interesting result.

For each question classify:

```text
Baseline correct?
RAG correct?
Retrieved relevant evidence?
```

Then you get:

| Baseline | Retrieval                                   | Result                       |
| -------- | ------------------------------------------- | ---------------------------- |
| Correct  | Relevant                                    | model already knew it        |
| Correct  | Irrelevant                                  | retrieval unnecessary        |
| Wrong    | Relevant                                    | retrieval rescued model      |
| Wrong    | Irrelevant                                  | retrieval failed             |
| Wrong    | Correct evidence retrieved but answer wrong | generation/reasoning failure |

This decomposition is much more informative than a single accuracy number.

---

# 47. The four-way rescue matrix

Especially report:

```text
                            RAG retrieves correctly

                       YES                     NO

Baseline YES       redundant retrieval      retrieval irrelevant
Baseline NO        RAG rescue               unavoidable failure
```

The most exciting cell is:

```text
Baseline WRONG
RAG RETRIEVAL CORRECT
RAG ANSWER CORRECT
```

That is your actual evidence that RAG helped.

---

# 48. Tool-call evaluation

For the agentic system, record:

```text
initial retrieval
tool calls
tool queries
tool results
final answer
```

Then determine:

### Did the tool help?

Compare:

```text
initial retrieval answer
```

vs.

```text
tool-assisted answer
```

Possible result:

```text
agentic retrieval:
+7% on multi-hop questions
+1% on simple conceptual questions
```

That would be interesting.

---

# 49. Measure retrieval efficiency

Track:

```text
number of paragraphs retrieved
number of tokens sent
number of tool calls
latency
```

Then compute:

$$
\text{retrieval efficiency}
=
\frac{\text{relevant evidence}}
{\text{tokens retrieved}}
$$

Your knee detector should hopefully improve this.

---

# 50. Evaluate dynamic retrieval itself

You should have an ablation study.

Compare:

```text
Fixed top-5
Fixed top-10
Fixed top-20
Knee-based
```

This tests whether your retrieval innovation is actually worthwhile.

Expected result might be:

```text
top-5:
high precision / lower recall

top-20:
high recall / noisy context

knee:
good recall / lower noise
```

That is a much stronger technical story.

---

# 51. Hierarchy ablation

Also test:

```text
Flat paragraph RAG
```

versus:

```text
Hierarchical RAG
```

Architecture:

```text
Flat:
query → paragraph vector search → LLM

Hierarchical:
query → book → chapter → topic
                  +
             global paragraph
                  ↓
                 LLM
```

This allows you to answer:

> Does document structure actually help?

---

# 52. Agentic ablation

Similarly:

```text
Hierarchical RAG
```

versus:

```text
Hierarchical RAG + tool calling
```

Now you have three increasingly sophisticated systems:

```text
Baseline LLM
      ↓
Flat RAG
      ↓
Hierarchical dynamic RAG
      ↓
Hierarchical dynamic agentic RAG
```

That is an excellent experiment.

---

# 53. Recommended final experimental table

You can eventually produce:

| System                           | Small | Medium | Frontier |
| -------------------------------- | ----: | -----: | -------: |
| No RAG                           |       |        |          |
| Flat RAG                         |       |        |          |
| Hierarchical RAG                 |       |        |          |
| Hierarchical + dynamic retrieval |       |        |          |
| Hierarchical + dynamic + tool    |       |        |          |

This becomes much more academically interesting than simply "RAG vs no RAG."

---

# 54. Knowledge-base construction

Recommended first corpus:

```text
Open Data Structures
+
selected CP-Algorithms
+
one additional structured DSA textbook/reference
```

You do not need:

```text
20 textbooks
```

A good initial target:

```text
Books:              3
Chapters:          ~30–60
Topics:            ~150–300
Paragraphs:        ~2,000–10,000
```

This is large enough to demonstrate retrieval but small enough for exhaustive evaluation.

---

# 55. Source metadata

Every node should carry provenance.

Example:

```json
{
  "source_file": "open_data_structures.pdf",
  "source_url": "...",
  "book": "Open Data Structures",
  "chapter": "Graphs",
  "topic": "Shortest Paths",
  "subtopic": "Dijkstra",
  "page": 267
}
```

This enables the UI to show:

```text
Open Data Structures
→ Graphs
→ Shortest Paths
→ Dijkstra
→ p. 267
```

---

# 56. Citation model

The LLM should cite retrieved sources in responses:

```text
[Dijkstra topic]
[Graph shortest paths]
```

These should map back to actual evidence objects.

Important:

> citations should only be produced for retrieved sources actually used.

Don't let the model invent page numbers.

---

# 57. Conversation context

Chat history introduces another retrieval problem.

User:

> Explain Dijkstra.

Assistant answers.

User:

> Why does that fail with negative edges?

The second query cannot be interpreted in isolation.

Therefore the effective retrieval query should be:

```text
current user query
+
relevant conversation state
```

But the system should avoid embedding the entire chat history.

Instead construct:

```text
retrieval_query =
    current question
    +
    compact conversation summary
    +
    resolved references
```

For example:

```text
"that algorithm"
```

should become:

```text
"Dijkstra's algorithm"
```

before retrieval.

---

# 58. Follow-up retrieval

This is another reason the LLM tool is useful.

If the user says:

> "What if the graph is a DAG?"

the model can query:

```text
DAG shortest paths negative edge weights
```

rather than relying entirely on the first retrieval.

---

# 59. Context architecture

The final LLM context should conceptually look like:

```text
SYSTEM INSTRUCTIONS

CURRENT QUESTION

CONVERSATION CONTEXT

RETRIEVED KNOWLEDGE

[TOPIC]
...

[PARAGRAPH S1]
...

[PARAGRAPH S2]
...

[TOOL RESULT S3]
...

INSTRUCTIONS:
Reason rigorously.
Use evidence where applicable.
Do not fabricate sources.
```

---

# 60. Retrieval state object

Internally define something like:

```text
RetrievalResult
```

containing:

```text
query

books[]
chapters[]
topics[]

paragraphs[]
expanded_topics[]

knee:
    candidate_k
    selected_k
    knee_index
    threshold

tool_calls[]

context_tokens
```

This object should be serializable.

That means:

* UI can display it.
* evaluator can log it.
* debugging can inspect it.
* benchmark analysis can process it.

---

# 61. Logging

Every benchmark run should produce a complete JSON record.

Example:

```json
{
  "question_id": "Q027",
  "model": "...",
  "rag_enabled": true,

  "retrieval": {
    "books": [...],
    "chapters": [...],
    "topics": [...],
    "paragraphs": [...],
    "selected_k": 7,
    "knee": 6
  },

  "tool_calls": [
    {
      "query": "...",
      "results": [...]
    }
  ],

  "answer": "...",

  "scores": {
    "correctness": 4,
    "reasoning": 3,
    "complexity": 4,
    "groundedness": 4
  },

  "latency": 2.31
}
```

This makes post-experiment analysis easy.

---

# 62. Reproducibility

Every run should record:

```text
model
embedding model
retrieval settings
prompt version
dataset version
knowledge-base manifest hash
timestamp
```

Especially:

```text
knowledge_base_hash
```

because changing the corpus changes the experiment.

---

# 63. Embedding model

Don't spend too much time optimizing this initially.

A lightweight embedding model is sufficient.

The initial goal is:

```text
fast
stable
local
reproducible
```

Then you can later compare embedding models.

That is an entirely separate experiment.

---

# 64. Vector store

FAISS remains a good choice.

Reasons:

* simple
* local
* fast
* transparent
* easy to inspect
* no external service
* excellent for a few thousand or tens of thousands of vectors

You don't need a production database.

---

# 65. Do not overengineer the retrieval algorithm

The elegance of the system should come from:

```text
document hierarchy
+
dynamic retrieval
+
tool-driven retrieval
```

not 17 different rerankers.

You previously questioned whether reranking is still useful with frontier models. This project is actually a nice way to test that empirically rather than argue about it.

Start with:

```text
dense similarity
+
hierarchical restriction
+
dynamic cutoff
```

Then optionally add reranking later as an ablation.

---

# 66. Why dynamic retrieval is interesting in modern RAG

The conceptual hypothesis is:

> Modern LLMs can handle heterogeneous context surprisingly well, therefore retrieval should focus less on imposing an arbitrary fixed context length and more on determining **which context is actually worth passing.**

That is exactly what your knee-based mechanism tests.

It is not:

> "The model needs only five chunks."

It is:

> "The number of useful chunks depends on the information structure of the query."

---

# 67. Broad-query behavior

Query:

> "Teach me dynamic programming."

Expected retrieval:

```text
Book:
    Algorithms text

Chapter:
    Dynamic Programming

Topics:
    optimal substructure
    overlapping subproblems
    memoization
    tabulation
    knapsack
    LIS
```

Possibly entire topic groups.

---

# 68. Narrow-query behavior

Query:

> "Why does 0/1 knapsack use dp[i-1] in the include transition?"

Expected:

```text
Topic:
    0/1 Knapsack

Paragraphs:
    recurrence definition
    state transition
    distinction from unbounded knapsack
```

No need to retrieve 30 paragraphs on dynamic programming.

---

# 69. Multi-hop behavior

Query:

> "Why does Kruskal need DSU, and could I replace it with a BFS after every edge?"

Expected:

```text
MST
+
Kruskal
+
DSU
+
cycle detection
+
connectivity
```

The model may use its tool:

```text
search:
"dynamic connectivity after edge additions cycle detection"
```

---

# 70. Out-of-KB behavior

This is essential.

Ask:

> "Explain van Emde Boas trees."

when the corpus does not contain them.

The system should answer:

> "I don't have sufficient information about van Emde Boas trees in the retrieved knowledge base."

It can optionally still provide a caveat:

> "I may know this from general model knowledge, but it is outside the retrieved curriculum."

That distinction is important.

---

# 71. RAG OFF behavior

When disabled, the UI should visibly say:

```text
RAG OFF
```

and the model should have:

```text
zero knowledge-base context
zero retrieval tool
```

The only difference from RAG ON is the availability of retrieval.

This is crucial for experimental validity.

---

# 72. Agentic RAG OFF

There is a subtle but important point.

If:

```text
RAG OFF
```

then the retrieval tool itself should not exist.

Otherwise the model can bypass the experimental switch.

So:

```text
RAG OFF
    ↓
tool unavailable

RAG ON
    ↓
tool available
```

---

# 73. Evaluation question design process

The gold dataset should be generated in layers.

### Stage 1

Compile a topic inventory:

```text
binary search
trees
heaps
graphs
...
```

### Stage 2

Generate candidate question types:

```text
definition
why
prove
counterexample
implementation
complexity
compare
design
debug
multi-hop
```

### Stage 3

Manually audit.

### Stage 4

Attempt to break the answers.

For each question ask:

> "How could a smart but slightly careless model answer this incorrectly?"

Then turn that into an adversarial version.

---

# 74. Frontier-model calibration

You specifically want questions difficult enough that a frontier model is not just trivially perfect.

The right difficulty is **not obscure trivia**.

Don't ask:

> "What theorem did some obscure 1986 paper prove?"

Ask:

> "What exactly breaks in this apparently reasonable algorithm under this altered assumption?"

Frontier models know DSA.

The challenge is:

```text
precision
```

rather than:

```text
memorization
```

---

# 75. Difficult question examples

### Example 1

> A graph contains one negative edge that does not belong to any shortest path from the source. Is Dijkstra necessarily correct?

Tests:

```text
greedy invariant
negative weights
counterexample reasoning
```

---

### Example 2

> Solve \(T(n)=2T(n/2)+n/\log n\) tightly.

Tests:

```text
recurrence reasoning
Master theorem limitations
```

---

### Example 3

> Explain why bottom-up heap construction is O(n), despite individual heapify operations being O(log n).

Tests:

```text
aggregate analysis
node height distribution
```

---

### Example 4

> Give a graph where DFS finds a target but not the shortest path first.

Tests:

```text
graph traversal guarantees
counterexample construction
```

---

# 76. Code evaluation

For implementation questions, natural-language judging isn't enough.

Where possible, create:

```text
question
candidate code
hidden tests
```

Then run the code.

For example:

> "Implement lower_bound."

Evaluation:

```text
correctness on 100 random arrays
+
adversarial edge cases
+
complexity review
```

This gives much stronger ground truth.

---

# 77. Mathematical question evaluation

For mathematical questions:

```text
gold derivation
+
symbolic/numeric checker
```

For recurrence problems:

```text
expected asymptotic form
```

For graph questions:

```text
small graph test harness
```

For algorithms:

```text
reference implementation
+
randomized differential testing
```

This is how you make the benchmark genuinely difficult.

---

# 78. Benchmark size

Initial:

```text
100 questions
```

Good final target:

```text
200–300 questions
```

Possible distribution:

```text
30 algorithm selection
30 correctness/proof
30 complexity
30 adversarial
30 implementation/debugging
20 multi-hop
20 competitive-programming
20 DP/recurrence
10 out-of-KB
```

---

# 79. Statistical reporting

Don't overinterpret a 2-question difference.

For each system calculate:

```text
mean
median
standard deviation
```

And ideally confidence intervals via bootstrap resampling.

For model comparison:

$$
\Delta = Score_{RAG}-Score_{NoRAG}
$$

Then bootstrap the question set to estimate uncertainty in \(\Delta\).

---

# 80. Per-category analysis

The overall score can hide everything.

Suppose:

```text
Small model

RAG improvement:

conceptual            +2%
complexity            +18%
proof                 +21%
multi-hop             +28%
implementation        +12%
out-of-KB              -1%
```

That tells a real story.

---

# 81. What success would look like

The strongest plausible result would look something like:

```text
                  No RAG    Hierarchical RAG

Small model          58          81
Medium model         72          84
Frontier model       91          94
```

while retrieval recall might be:

```text
87%
```

and average context size:

```text
dynamic:
    5.7 paragraphs/query
```

instead of:

```text
fixed:
    15 paragraphs/query
```

That gives you **both model-level and retrieval-level evidence**.

---

# 82. What failure would teach you

Suppose:

```text
Flat RAG              75
Hierarchical RAG      74
Dynamic RAG            73
Agentic RAG            72
```

That would be incredibly useful.

It would suggest:

* additional context hurts
* frontier-like models don't benefit from hierarchical filtering
* tool retrieval is unnecessary
* the baseline retrieval is already sufficient

That's a legitimate conclusion.

---

# 83. Recommended ablation sequence

Do not build everything simultaneously and then have no idea what mattered.

Build conceptually:

```text
A = LLM only

B = flat paragraph RAG

C = hierarchical RAG

D = hierarchical + dynamic cutoff

E = hierarchical + dynamic + topic expansion

F = E + agentic retrieval
```

Then test:

```text
A vs B
B vs C
C vs D
D vs E
E vs F
```

This isolates the contribution of every major idea.

---

# 84. Final system architecture

The full system should look like this:

```text
                         ┌──────────────────────┐
                         │   Knowledge Corpus   │
                         │ PDFs / MD / TXT      │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │ Structural Parser    │
                         │ book/chapter/topic   │
                         │ paragraph hierarchy  │
                         └──────────┬───────────┘
                                    │
             ┌──────────────────────┼────────────────────────┐
             │                      │                        │
             ▼                      ▼                        ▼
        Book Index            Chapter Index             Topic Index
                                                             │
                                                             │
                                                             ▼
                                                      Paragraph Index


                           USER QUESTION
                                │
                                ▼
                       Conversation resolver
                                │
                                ▼
                       Query understanding
                                │
                                ▼
                         Book retrieval
                                │
                                ▼
                       Chapter retrieval
                                │
                                ▼
                        Topic retrieval
                                │
                   ┌────────────┴─────────────┐
                   │                          │
                   ▼                          ▼
             Topic expansion            Global paragraph
                   │                     vector search
                   │                          │
                   │                     similarity curve
                   │                          │
                   │                       knee point
                   │                          │
                   │                     dynamic chunks
                   │                          │
                   └────────────┬─────────────┘
                                ▼
                         context builder
                                │
                                ▼
                               LLM
                                │
                   ┌────────────┴─────────────┐
                   │                          │
                answer                   retrieval tool
                                              │
                                              ▼
                                         vector DB
                                              │
                                              ▼
                                         more evidence
                                              │
                                              └──────► LLM
```

---

# 85. Evaluation architecture

Keep it completely separate from the UI:

```text
                    HARD DSA BENCHMARK
                           │
                           ▼
                       Question
                           │
                           ▼
                    config.rag_enabled
                       /          \
                     OFF          ON
                     │             │
                     ▼             ▼
                    LLM       Retrieval
                                  │
                                  ▼
                             LLM + tools
                                  │
                     ┌────────────┴─────────────┐
                     │                          │
                     ▼                          ▼
                   Answer                 Retrieval logs
                     │
                     └────────────┬─────────────┘
                                  ▼
                             Evaluation
                                  │
                 ┌────────────────┼──────────────────┐
                 ▼                ▼                  ▼
             Retrieval        Correctness       Groundedness
                metrics          metrics            metrics
```

---

# 86. The README's central narrative

The eventual README should not sell this as:

> "I built a chatbot using LangChain."

It should say something closer to:

> **DSA Mentor is a hierarchical, adaptive RAG system designed to study when retrieval improves algorithmic reasoning.**
>
> The system preserves the natural hierarchy of DSA textbooks, retrieves from coarse to fine granularity, dynamically determines the amount of paragraph-level evidence using similarity-score knee detection, and allows the language model to issue additional retrieval queries when its initial evidence is insufficient.
>
> The benchmark evaluates the system against a large set of adversarial DSA questions under both RAG and no-RAG conditions, across models of different capabilities.

That sounds like an AI project rather than an API integration.

---

# 87. What I would actually implement first

The blueprint should be implemented in this order:

```text
PHASE 1
Document ingestion
↓
Hierarchy extraction
↓
Paragraph chunks
↓
Four vector indices


PHASE 2
Flat retrieval baseline


PHASE 3
Hierarchical retrieval


PHASE 4
Similarity curve + knee cutoff


PHASE 5
Topic expansion + paragraph neighborhood


PHASE 6
LLM generation


PHASE 7
Tool-based retrieval


PHASE 8
Streamlit chat


PHASE 9
Hard benchmark


PHASE 10
Ablation experiments
```

Don't build agentic retrieval before you've established that the ordinary hierarchical retriever works.

---

# 88. The core conceptual contribution

The final project should effectively have **three hypotheses**:

### H1 — Structure

> Preserving textbook hierarchy improves retrieval over flat paragraph retrieval.

### H2 — Adaptivity

> Dynamic evidence selection based on the similarity distribution provides a better relevance/context tradeoff than fixed top-k retrieval.

### H3 — Model dependence

> Retrieval provides greater marginal benefit to weaker models than to frontier models on difficult DSA reasoning tasks.

Then:

### H4 — Agentic retrieval

> Allowing the model to perform additional targeted searches improves multi-hop and evidence-deficient queries more than straightforward conceptual questions.

That is already a very respectable experimental framing.

---

# 89. The thing I would be most careful about

There is one conceptual trap in your proposed project:

> **Do not assume that more retrieved information is better.**

Your whole dynamic-k approach should be built around the opposite principle:

$$
\text{Useful Context}
\neq
\text{Maximum Context}
$$

The objective is:

$$
\boxed{
\text{maximize relevant information}
\over
\text{minimize irrelevant information}
}
$$

subject to:

$$
\text{context length} \leq C
$$

and ideally:

$$
\text{retrieval cost} \leq B
$$

where \(C\) is the context budget and \(B\) is the latency/cost budget.

That gives a clean mathematical motivation for your retrieval system.

---

# 90. Final mental model

The whole project can be reduced to four layers:

```text
LAYER 1 — KNOWLEDGE
Books
→ Chapters
→ Topics
→ Paragraphs


LAYER 2 — RETRIEVAL
Broad retrieval
→ hierarchical narrowing
→ dynamic paragraph selection
→ local expansion


LAYER 3 — REASONING
LLM
→ grounded answer
→ optional retrieval tool
→ iterative evidence gathering


LAYER 4 — SCIENCE
RAG OFF vs RAG ON
→ small vs frontier models
→ fixed vs dynamic retrieval
→ flat vs hierarchical
→ passive vs agentic
```

And the core question is:

> **Can we give a model exactly the DSA knowledge it needs, rather than simply giving it as much context as possible?**

That is the idea I'd build the entire project around.

The next implementation phase should therefore start from **the data model and knowledge hierarchy**, because every later component—hierarchical retrieval, knee detection, tool calls, citations, UI inspection, and evaluation—depends on getting that representation right.
