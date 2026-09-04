# DSA Mentor

> A hierarchical, adaptive RAG system for algorithmic reasoning — built to study when and how retrieval improves DSA tutoring across models of different capabilities.

---

## What It Is

DSA Mentor is a retrieval-augmented generation (RAG) system built specifically for teaching Data Structures & Algorithms. Unlike generic chatbots that rely on parametric knowledge alone, DSA Mentor:

- **Preserves the natural hierarchy** of DSA textbooks (Book → Chapter → Topic → Subtopic → Paragraph)
- **Retrieves from coarse to fine granularity**, narrowing the search space at each level
- **Dynamically determines evidence volume** using similarity-score knee detection — broad queries get broad context, narrow queries get precise citations
- **Allows the LLM to perform follow-up searches** when initial evidence is insufficient (agentic retrieval)
- **Is fully observable** — every retrieval decision, similarity score, and knee selection is inspectable

The system was built around a central research question: *When does retrieval materially improve algorithmic reasoning, and how does the effect vary with model capability?*

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        USER QUERY                                   │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│  QUERY UNDERSTANDING                                                │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │ Regex-based breadth classifier: BROAD / MODERATE / NARROW   │    │
│  │ 690+ narrow technical terms · 30 DSA concept names          │    │
│  └─────────────────────────────────────────────────────────────┘    │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌────────────────────────────────────────────────────────────────────────────────┐
│  HIERARCHICAL RETRIEVAL (5 levels × 5 FAISS indices)                           │
│                                                                                │
│  Book Index     → top-k books     → knee detection                             │
│       ↓                                                                        │
│  Chapter Index  → top-k chapters  → knee detection (dedup across books)        │
│       ↓                                                                        │
│  Topic Index    → top-k topics    → knee detection (dedup across chapters)     │
│       ↓                                                                        │
│  Subtopic Index → top-k subtopics → knee detection                             │
│       ↓                                                                        │
│  Paragraph Index → top-k paragraphs → knee detection (flat fallback if <3)     │
│                                                                                │
│  Each level: FAISS cosine search → similarity curve → elbow detection          │
└────────────────────────────┬───────────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│  CONTEXT EXPANSION                                                  │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │ Topic Expansion: populate full_text for selected topics     │    │
│  │   BROAD → all · MODERATE → top 3 · NARROW → top 1           │    │
│  └─────────────────────────────────────────────────────────────┘    │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │ Neighbor Expansion: ±2 paragraphs within same topic         │    │
│  │ Provides surrounding context (definitions, examples, proofs)│    │
│  └─────────────────────────────────────────────────────────────┘    │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │ Context Budget: ~20,000 token ceiling, truncate lowest-sim  │    │
│  └─────────────────────────────────────────────────────────────┘    │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌───────────────────────────────────────────────────────────────────────┐
│  AGENTIC LLM LOOP                                                     │
│  ┌───────────────────────────────────────────────────────────────┐    │
│  │ 1. Build context: system prompt + retrieved knowledge + query │    │
│  │ 2. Offer search_knowledge(query, scope, max_results) tool     │    │
│  │ 3. Model calls tool → results appended → repeat (max 3)       │    │
│  │ 4. Budget exhausted → force final answer without tools        │    │
│  └───────────────────────────────────────────────────────────────┘    │
└────────────────────────────┬──────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│  RESPONSE + RETRIEVAL INSPECTOR                                     │
│  Grounded answer with source citations · Full retrieval trace       │
└─────────────────────────────────────────────────────────────────────┘
```

### Component Breakdown

#### 1. Knowledge Ingestion Pipeline

Documents in `docs/` are parsed into a **5-level hierarchy** with unique IDs, parent-child links, and provenance metadata:

| Level | Embedding Strategy | FAISS Index |
|---|---|---|
| **Book** | Aggregated from child chapters | `IndexFlatIP` (cosine) |
| **Chapter** | Title + section headings | `IndexFlatIP` (cosine) |
| **Topic** | Title + full_text body | `IndexFlatIP` (cosine) |
| **Subtopic** | Title + content | `IndexFlatIP` (cosine) |
| **Paragraph** | Content (1,500 char chunks, 200 char overlap) | `IndexFlatIP` (cosine) |

Three parsers handle different source formats:
- **PDF parser** — page-by-page extraction via pypdf, heading detection via regex + uppercase-ratio heuristic
- **Markdown parser** — heading-based subtopic boundaries, fenced code block preservation
- **Text parser** — GfG article format with Source/Title/Extractor headers

The output is `knowledge_base.json` containing all hierarchy nodes plus a `KnowledgeManifest` with SHA-256 file hashes for integrity verification.

#### 2. Embedding Engine

Three swappable backends with automatic fallback:

```
OpenAI-compatible HTTP endpoint  →  sentence-transformers (all-MiniLM-L6-v2)  →  TF-IDF (scikit-learn)
```

All backends return **L2-normalized vectors** so that dot product in FAISS equals cosine similarity. Batch processing (64 docs per batch) with lazy model loading.

#### 3. FAISS Index Layer

- **`FAISSIndex`** — thin wrapper with GPU auto-detection (`GpuIndexFlatIP` when CUDA is available, `IndexFlatIP` otherwise), save/load with CPU serialization
- **`MultiIndexManager`** — manages all 5 indices, maintains metadata mappings (id → node, node_id → position), scoped search with parent-id filtering

#### 4. Knee Detection (Core Innovation)

At each hierarchy level, instead of using a fixed `top_k`, the system computes the **similarity-score curve** and finds the elbow point:

1. Take top `candidate_k` sorted scores
2. Compute first differences: `Δᵢ = sᵢ - sᵢ₊₁`
3. Normalize by `max(scores)`
4. Find index with maximum normalized drop → **knee position**
5. If max drop ≥ threshold (0.02): select all items through the knee
6. Else: select all scores ≥ `similarity_threshold` (0.15)
7. Enforce `[minimum, maximum]` bounds from config

This means a broad query like "explain sorting" might select 5 books, 8 chapters, 10 topics, 25 paragraphs — while a narrow query like "settled vertex invariant in Dijkstra" might select 1 book, 1 chapter, 1 topic, 3 paragraphs.

#### 5. Query Breadth Classification

A regex-based classifier (no LLM dependency) categorizes queries:

| Category | Trigger | Behavior |
|---|---|---|
| **NARROW** | 1+ technical term from 690-term lexicon | Top 1 topic, minimal expansion |
| **MODERATE** | 1-2 concept names, short query | Top 3 topics, ±2 neighbors |
| **BROAD** | General starters ("explain", "teach"), 3+ concepts | All topics, full expansion |

#### 6. Agentic Retrieval Tool Loop

The LLM can call `search_knowledge(query, scope, max_results)` where scope is `all`, `book`, `chapter`, or `topic`. The loop:

1. Sends initial context with tool schema
2. Model responds with tool calls or text answer
3. Each tool call executes a scoped FAISS search, appends results
4. After `max_tool_calls` (3) executions, tool is removed — model is forced to answer from gathered evidence
5. Tool failures are caught and returned as error strings for model recovery

#### 7. Context Builder

Assembles the final prompt with:
- System prompt (varies by RAG mode and tool availability)
- Conversation context (last 5 turns)
- Retrieved knowledge formatted as `[TOPIC]` / `[SUBTOPIC]` sections
- Instructions for the model

Deduplication, diversity constraints (max 6 paragraphs per source), and token budget enforcement are applied.

#### 8. Streamlit UI

Three interaction modes:
- **Learn** — standard Q&A with hierarchical retrieval
- **Hint** — progressive hints for LeetCode-style problems (max 4 levels)
- **Explain** — code analysis: correctness, bugs, invariants, complexity, alternatives

Features:
- Streaming responses (character-by-character yield)
- Retrieval inspector (expandable panel with all scores, knee selections, tool calls)
- Source citations appended below responses
- RAG toggle, model selector, index build controls
- `<thinking>` tag extraction → collapsible grey block

#### 9. Benchmark & Ablation Suite

**Experimental matrix:** 3 model tiers × 2 RAG states × 5 retrieval methods = 30 runs

| Metric | Method |
|---|---|
| **Correctness** (0–4) | Required claims coverage + forbidden claims penalty |
| **Groundedness** (0–1) | Answer claim overlap with retrieved text |
| **Recall@K / Precision@K** | Set overlap of retrieved vs relevant paragraphs |
| **Rescue Matrix** | 5-way classification: baseline correct/relevant, baseline correct/irrelevant, RAG rescue, unavoidable failure, generation failure |

**6 ablation levels** trace the contribution of each architectural choice:

| Level | Configuration | Tests |
|---|---|---|
| A | LLM only (RAG OFF) | Parametric knowledge baseline |
| B | Flat paragraph RAG | Dense similarity alone |
| C | Hierarchical + fixed top-k | Document structure value |
| D | Hierarchical + knee detection | Dynamic evidence selection |
| E | D + topic/neighbor expansion | Topic-aware context |
| F | E + agentic tool calling | Active retrieval |

Additional evaluation modules:
- **RAGAS retrieval eval** — 20 questions with gold context, context precision/recall/faithfulness
- **System profiler** — per-stage timing (embedding, FAISS, knee, expansion), query complexity analysis, system health reporting

---

## Corpus

| Source | Type | Files | Role |
|---|---|---:|---|
| Open Data Structures (Morin et al.) | PDF | 1 | Primary textbook |
| Introduction to Computer Science (OpenStax) | PDF | 1 | Primary textbook |
| CP-Algorithms | Markdown | 164 | Algorithm reference |
| JavaScript Algorithms | Markdown | 236 | Data structure reference |
| The Algorithms (Python) | Markdown | 22 | Category overviews |
| GeeksforGeeks DSA Tutorial | Text | 1,505 | Tutorial corroboration |

**~85 MB total** — deliberately small but deep. Quality over quantity.

---

## Quick Start

### Prerequisites

- Python 3.11+
- GPU (optional) — CUDA with `faiss-gpu` for faster search
- ~4 GB disk for knowledge base, indices, and model cache

```bash
cd "DSA Instructor"

# Install dependencies
pip install -r requirements.txt

# Build knowledge base + vector index
python -m dsa_mentor.ingestion.build --index

# Launch the chat UI
streamlit run app/streamlit_app.py
```

Open the Streamlit URL (usually `http://localhost:8501`).

### GPU Setup

```bash
pip uninstall -y faiss-cpu && pip install faiss-gpu
python -m dsa_mentor.ingestion.build --index --force
```

Auto-detects CUDA at runtime — no config changes needed.

---

## Configuration

All tunables live in `config.json` — nothing is hardcoded:

```jsonc
{
  "llm": {
    "base_url": "http://localhost:1234/v1",
    "models": { "large": "qwen3.6-35b-a3b-mtp", "medium": "phi-4-reasoning-plus", "small": "gemma-4-e4b-it" }
  },
  "retrieval": {
    "similarity_threshold": 0.15,
    "max_context_tokens": 20000,
    "paragraph_max_chars": 1500,
    "paragraph_overlap_chars": 200,
    "book_knee":     { "candidate_k": 10, "minimum": 1, "maximum": 5 },
    "chapter_knee":  { "candidate_k": 20, "minimum": 1, "maximum": 8 },
    "topic_knee":    { "candidate_k": 30, "minimum": 1, "maximum": 10 },
    "subtopic_knee": { "candidate_k": 40, "minimum": 1, "maximum": 15 },
    "paragraph_knee":{ "candidate_k": 50, "minimum": 1, "maximum": 25 }
  },
  "agentic_retrieval": { "enabled": true, "max_tool_calls": 3 }
}
```

---

## Project Structure

```
DSA Instructor/
├── config.json                  # All tunables (models, retrieval, budgets)
├── spec.md                      # Technical blueprint
├── knowledge_base_strategy.md   # Corpus sources & curation
├── knowledge_base.json          # Parsed corpus (5-level hierarchy)
│
├── docs/                        # Source documents (85 MB)
│   ├── *.pdf                    # 2 primary books
│   ├── cp-algorithms/           # 164 md articles
│   ├── javascript-algorithms/   # 236 md docs
│   ├── thealgorithms/           # 22 category READMEs
│   └── geeksforgeeks/           # 1,505 txt articles
│
├── dsa_mentor/                  # Core Python package
│   ├── config.py                # Typed config loader with validation
│   ├── models.py                # Dataclasses: Book/Chapter/Topic/Subtopic/Paragraph/RetrievalResult
│   ├── llm.py                   # OpenAI-compatible client + agentic tool loop
│   ├── embeddings.py            # 3-backend embedding client (HTTP → ST → TF-IDF)
│   ├── prompts.py               # System prompts (RAG ON/OFF, tool enabled/disabled)
│   ├── context.py               # Context builder (dedup, budget, diversity)
│   ├── ingestion/               # PDF/MD/TXT parsers → hierarchy nodes
│   ├── index/                   # FAISS indices (base, flat, multi, hierarchy)
│   └── retrieval/               # Knee detection, breadth expansion, tools
│
├── app/
│   └── streamlit_app.py         # Chat UI with retrieval inspector
│
├── benchmark/
│   ├── dataset.py               # Dataset loader + sample questions
│   ├── scoring.py               # Correctness, grounding, recall/precision@k, rescue matrix
│   ├── runner.py                # Experiment runner (3×2×5 matrix)
│   ├── ablations.py             # 6 ablation levels (A–F), pairwise deltas, report
│   ├── ragas_retrieval.py       # RAGAS retrieval evaluation
│   └── system_logging.py        # Pipeline profiler + system health
│
├── index/                       # Built FAISS indices (5 subdirectories)
└── benchmark/results/           # Saved experiment outputs
```

---

## Key Design Decisions

| Decision | Rationale |
|---|---|
| **Paragraph-level evidence** | Preserves document structure — definitions with qualifications, theorems with proofs, algorithms with complexity |
| **Dynamic evidence selection** | Knee detection adapts to query breadth; no manual `top_k` tuning per query type |
| **Hierarchical filtering** | Each level narrows the search space before the next, reducing noise and improving precision |
| **Agentic tool loop** | Model can request targeted follow-ups when initial evidence is insufficient (hard budget: 3 calls) |
| **Config-driven everything** | Every tunable in `config.json` — models, knees, budgets, flags. Zero hardcoded values in code |
| **Swappable embeddings** | HTTP endpoint → sentence-transformers → TF-IDF fallback chain |
| **GPU auto-detection** | `GpuIndexFlatIP` when CUDA available, `IndexFlatIP` otherwise — no config flag needed |
| **RAG OFF = no tool** | Clean experimental separation between parametric and retrieval-augmented reasoning |

---

## CLI Reference

### Build

```bash
python -m dsa_mentor.ingestion.build              # Parse docs → knowledge_base.json
python -m dsa_mentor.ingestion.build --index      # Parse + embed + build FAISS indices
python -m dsa_mentor.ingestion.build --index --force  # Force rebuild
```

### Benchmark

```bash
python -m benchmark                     # Single run: large model, RAG ON, knee
python -m benchmark --full              # Full matrix: 3 models × 2 RAG × 5 methods
python -m benchmark --model medium --rag off --method fixed_top_10
```

---

## Tech Stack

| Component | Technology |
|---|---|
| **Language** | Python 3.11 |
| **Vector Search** | FAISS (CPU or GPU) |
| **Embeddings** | sentence-transformers / OpenAI-compatible HTTP / TF-IDF |
| **LLM Interface** | OpenAI-compatible HTTP API |
| **UI** | Streamlit |
| **PDF Parsing** | pypdf |
| **Evaluation** | Custom scoring + RAGAS |
