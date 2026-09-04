# DSA Mentor

A hierarchical, adaptive RAG system for DSA tutoring — designed to study when retrieval improves algorithmic reasoning across models of different capabilities.

The system preserves the natural hierarchy of DSA textbooks, retrieves from coarse to fine granularity, dynamically determines the amount of paragraph-level evidence using similarity-score knee detection, and allows the language model to issue additional retrieval queries when its initial evidence is insufficient.

---

## Quick Start

### Prerequisites

- **Python 3.11+**
- **GPU (optional)** — CUDA with `faiss-gpu` for faster embedding and search. CPU-only works with `faiss-cpu`.
- **~4 GB disk** for the knowledge base, vector indices, and model cache

### Step-by-step setup

```bash
# 1. Navigate to the project
cd "DSA Instructor"

# 2. Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate              # Windows
# source venv/bin/activate         # Linux / macOS

# 3. Install dependencies
pip install -r requirements.txt

# 4. Build the knowledge base AND vector index (one command)
python -m dsa_mentor.ingestion.build --index

# 5. Launch the chat UI
streamlit run app/streamlit_app.py
```

Open the Streamlit URL shown in the terminal (usually `http://localhost:8501`).

### GPU setup (optional)

If you have a CUDA-capable GPU, replace `faiss-cpu` with `faiss-gpu` after step 3:

```bash
pip uninstall -y faiss-cpu
pip install faiss-gpu
# Rebuild indices to use GPU
python -m dsa_mentor.ingestion.build --index --force
```

The project auto-detects CUDA at runtime — no config changes needed.

---

## Configuration

All tunables live in `config.json`:

```jsonc
{
  "llm": {
    "base_url": "http://26.50.165.63:1234/v1",
    "models": { "large": "...", "medium": "...", "small": "..." },
    "sampling": { "temperature": 0.7, "top_p": 0.95, "top_k": 40, "max_tokens": 8192 }
  },
  "retrieval": {
    "similarity_threshold": 0.35,
    "max_context_tokens": 16000,
    "paragraph_max_chars": 1800,
    "paragraph_overlap_chars": 250,
    "book_knee":     { "candidate_k": 10, "minimum": 1, "maximum": 4 },
    "chapter_knee":  { "candidate_k": 15, "minimum": 1, "maximum": 6 },
    "topic_knee":    { "candidate_k": 20, "minimum": 1, "maximum": 8 },
    "paragraph_knee":{ "candidate_k": 40, "minimum": 3, "maximum": 20 }
  },
  "agentic_retrieval": { "enabled": true, "max_tool_calls": 3 },
  "diversity": { "max_paragraphs_per_source": 6 }
}
```

Key settings:
- **`llm.base_url`** — OpenAI-compatible API endpoint for your LLM server
- **`llm.models`** — model names for large / medium / small tiers
- **`retrieval.max_context_tokens`** — total token budget for retrieved context
- **`agentic_retrieval.enabled`** — set `false` to disable tool-based follow-up queries

---

## How It Works

```
USER QUERY
    ↓
Query Understanding (breadth classification: BROAD / MODERATE / NARROW)
    ↓
Book Retrieval ──→ knee detection
    ↓
Chapter Retrieval ──→ knee detection
    ↓
Topic Retrieval ──→ knee detection
    ↓
Full Topic Text + Global Paragraph Vector Search
    ↓
Topic Expansion + Neighbor Selection
    ↓
Context Builder (dedup, budget, diversity)
    ↓
LLM (with optional tool-based retrieval loop)
```

1. **Documents in `docs/`** are parsed into a 5-level hierarchy: Book → Chapter → Topic → Subtopic → Paragraph.
2. **`--index`** builds 4 FAISS vector indices (book, chapter, topic, paragraph) in the `index/` directory using pre-normalized vectors for cosine similarity.
3. **Streamlit app** auto-loads existing indices from `index/` and provides the chat interface.
4. **RAG retrieval** uses knee detection on the similarity-score curve at each hierarchy level to dynamically determine how many items to keep — broad queries get more evidence, narrow queries get precise evidence.
5. **Agentic mode** lets the LLM call `search_knowledge(query, scope, max_results)` up to 3 times when initial evidence is insufficient.

---

## CLI Reference

### Build command

```bash
python -m dsa_mentor.ingestion.build              # Parse docs/ → knowledge_base.json only
python -m dsa_mentor.ingestion.build --index      # Parse + vectorize + build FAISS indices
python -m dsa_mentor.ingestion.build --index --force  # Force rebuild indices (skip existing)
python -m dsa_mentor.ingestion.build --verbose    # Enable debug logging
```

| Flag | Description |
|---|---|
| *(none)* | Parse documents and save `knowledge_base.json` (no vector indexing) |
| `--index` | Parse documents, embed, and build all 4 FAISS indices |
| `--force` | Rebuild indices even if they already exist |
| `--verbose` | Print detailed parsing and embedding progress |

### Benchmark command

```bash
python -m benchmark                     # Single run: large model, RAG ON, knee detection
python -m benchmark --model medium --rag off --method fixed_top_10
python -m benchmark --full              # Full matrix: 3 models × 2 RAG × 5 methods = 30 runs
```

---

## Running Benchmarks

The benchmark suite evaluates retrieval quality and answer correctness across models and retrieval methods.

### Single Run

```bash
# RAG ON, large model, knee detection (default)
python -m benchmark

# Specify options
python -m benchmark --model medium --rag off --method fixed_top_10

# Custom questions and output
python -m benchmark --questions benchmark/sample_questions.jsonl --output my_results.jsonl
```

### Full Experiment Matrix

```bash
python -m benchmark --full
```

Output files:
- `benchmark/results.jsonl` — per-question results for the last run
- `benchmark/results_{model}_{rag}_{method}.jsonl` — individual result files from `--full`
- `benchmark/results.json` — full experiment results (only from `--full`)

### Ablation Studies

```python
from benchmark.ablations import AblationStudy

# Run all ablation levels
AblationStudy.run_all(config_path="config.json")

# Run a specific level
AblationStudy.run_level_A(config_path="config.json")  # LLM only (RAG OFF)
AblationStudy.run_level_D(config_path="config.json")  # Hierarchical + knee detection
```

---

## Corpus

The knowledge base (~85 MB) consists of:

| Source | Type | Files | Size | Role |
|---|---|---:|---:|---|
| *Open Data Structures* (Morin et al.) | PDF | 1 | 1.7 MB | Primary book |
| *Introduction to Computer Science* (OpenStax) | PDF | 1 | 56 MB | Primary book |
| CP-Algorithms | Markdown | 164 | 1.6 MB | Supplementary reference |
| JavaScript Algorithms | Markdown | 236 | 0.7 MB | Supplementary reference |
| The Algorithms (Python) | Markdown | 22 | 41 KB | Category overviews |
| GeeksforGeeks DSA Tutorial | Text | 1505 | 24 MB | Tutorial corroboration |

See `knowledge_base_strategy.md` for full source details, curation decisions, and licensing notes.

---

## UI Features

The Streamlit app provides:

- **Learn mode** — conceptual explanations with hierarchical retrieval
- **Hint mode** — progressive hints for LeetCode-style problems
- **Explain mode** — code analysis (correctness, bugs, invariants, complexity, alternatives)
- **Retrieval Inspector** — expandable panel showing books, chapters, topics, paragraphs with similarity scores and knee detection results
- **RAG toggle** — switch between RAG ON (retrieval + tool calling) and RAG OFF (model parametric knowledge only)
- **Model selector** — Large / Medium / Small

---

## Architecture

```
LAYER 1 — KNOWLEDGE
  Books → Chapters → Topics → Subtopics → Paragraphs

LAYER 2 — RETRIEVAL
  Broad retrieval → hierarchical narrowing → dynamic paragraph selection → local expansion

LAYER 3 — REASONING
  LLM → grounded answer → optional retrieval tool → iterative evidence gathering

LAYER 4 — SCIENCE
  RAG OFF vs RAG ON → small vs frontier models → fixed vs dynamic retrieval → flat vs hierarchical → passive vs agentic
```

### Key Design Decisions

- **Paragraph-level retrieval** — the atomic evidence unit is the paragraph, not arbitrary token chunks. This preserves document structure (definition with qualification, theorem with proof, algorithm with complexity).
- **Dynamic evidence selection** — knee detection on the similarity-score curve determines how many candidates to keep at each level, rather than using a fixed `top_k`. This adapts to query breadth: broad queries get more evidence, narrow queries get precise evidence.
- **Hierarchical filtering** — four separate FAISS indices (book, chapter, topic, paragraph) enable coarse-to-fine retrieval. Each level narrows the search space before passing to the next.
- **Agentic retrieval** — the model can call `search_knowledge(query, scope, max_results)` to request additional evidence when initial retrieval is insufficient (max 3 tool calls).
- **RAG OFF is a first-class mode** — when disabled, there is no retrieval tool either. This enables clean experimental comparison between parametric knowledge and retrieval-augmented reasoning.

---

## Project Structure

```
DSA Instructor/
├── config.json                  # All tunables (models, retrieval, budgets)
├── spec.md                      # Technical blueprint (source of truth)
├── knowledge_base_strategy.md   # Corpus sources & curation strategy
├── EXECUTION_NOTES.md           # Session-by-session build log
│
├── docs/                        # Corpus sources (see knowledge_base_strategy.md)
│   ├── *.pdf                    # 2 primary books
│   ├── cp-algorithms/           # 164 md articles
│   ├── javascript-algorithms/   # 236 md docs
│   ├── thealgorithms/           # 22 category READMEs
│   └── geeksforgeeks/           # 1505 txt articles
│
├── dsa_mentor/                  # Python package
│   ├── config.py                # Config loader (typed, validated)
│   ├── llm.py                   # OpenAI-compatible client + tool loop
│   ├── embeddings.py            # Embedding client (3 backends)
│   ├── models.py                # Dataclasses: Book/Chapter/Topic/Paragraph/RetrievalResult
│   ├── prompts.py               # System prompts (RAG ON/OFF, tool enabled/disabled)
│   ├── context.py               # Context builder (dedup, budget, diversity)
│   ├── ingestion/               # Parsers: pdf, md, txt → hierarchy nodes
│   ├── index/                   # FAISS indices: base, flat, multi, hierarchy
│   └── retrieval/               # Knee detection, expand, tools
│
├── app/                         # Streamlit chat UI
│   └── streamlit_app.py         # Main application
│
└── benchmark/                   # Evaluation
    ├── dataset.py               # Dataset loader + sample questions
    ├── scoring.py               # Correctness, grounding, recall/precision@k, rescue matrix
    ├── runner.py                # RAG on/off runner + full experiment matrix
    └── ablations.py             # 6 ablation levels (A–F), pairwise deltas, report
```

---

## Research Questions

The system is designed to answer:

> **When does retrieval materially improve DSA reasoning, and how does the effect vary with model capability?**

### Hypotheses

- **H1 (Structure):** Preserving textbook hierarchy improves retrieval over flat paragraph retrieval.
- **H2 (Adaptivity):** Dynamic evidence selection based on the similarity distribution provides a better relevance/context tradeoff than fixed `top_k` retrieval.
- **H3 (Model Dependence):** Retrieval provides greater marginal benefit to weaker models than to frontier models on difficult DSA reasoning tasks.
- **H4 (Agentic Retrieval):** Allowing the model to perform additional targeted searches improves multi-hop and evidence-deficient queries more than straightforward conceptual questions.

### Ablation Sequence

| Level | System | What it tests |
|---|---|---|
| A | LLM only (RAG OFF) | Baseline parametric knowledge |
| B | Flat paragraph RAG | Dense similarity alone |
| C | Hierarchical RAG (fixed top-k) | Document structure |
| D | Hierarchical + knee detection | Dynamic evidence selection |
| E | D + topic expansion + neighbors | Topic-aware context |
| F | E + agentic tool calling | Active retrieval |

The full experimental matrix covers 3 model tiers × 2 RAG states × 5 retrieval methods (spec §53).

---

## Metrics

| Metric | Description |
|---|---|
| **Correctness** (0–4) | Based on required_claims coverage and forbidden_claims penalty |
| **Groundedness** (0.0–1.0) | Overlap between answer claims and retrieved evidence |
| **Recall@K** | Relevant retrieved / all relevant evidence |
| **Precision@K** | Relevant retrieved / retrieved evidence |
| **Rescue Matrix** | 5-way classification: baseline correct/relevant, baseline correct/irrelevant, RAG rescue, unavoidable failure, generation failure |

---

## Troubleshooting

| Issue | Solution |
|---|---|
| **No GPU / CUDA not found** | Uses `faiss-cpu` automatically. Ensure `faiss-cpu` is installed (not `faiss-gpu`). |
| **Embedding model fails** | Falls back: HTTP OpenAI API → `sentence-transformers` (`all-MiniLM-L6-v2`) → TF-IDF. Check which backends are installed. |
| **Index already exists** | Use `--force` to rebuild: `python -m dsa_mentor.ingestion.build --index --force` |
| **Streamlit won't start** | Ensure virtual environment is activated and `streamlit` is installed in it. |
| **LLM connection refused** | Verify `config.json` `llm.base_url` matches your running LLM server. |
| **OneDrive PermissionError during ingestion** | The scraper retries on lock. Move `docs/` out of OneDrive sync if issues persist. |
| **GeeksforGeeks scraper artifacts** | Files matching `_*.py`, `_*.txt`, `_*.json`, `_*.log`, `_*.html` are excluded automatically. |

---

## Dependencies

All dependencies are listed in `requirements.txt`. Install with `pip install -r requirements.txt`.

| Package | Purpose |
|---|---|
| `torch` | GPU detection for FAISS and embeddings (required, even for CPU-only setups) |
| `faiss-cpu` | Vector similarity search (CPU; install `faiss-gpu` instead if CUDA is available) |
| `faiss-gpu` | GPU-accelerated vector search (alternative to `faiss-cpu` when CUDA is available) |
| `sentence-transformers` | Local embedding model (`all-MiniLM-L6-v2`) |
| `scikit-learn` | TF-IDF fallback embedding backend |
| `pypdf` | PDF text extraction |
| `streamlit` | Chat UI |
| `requests` | HTTP client for LLM API |
| `numpy` | Numerical operations |

---

## License Notes

| Source | License |
|---|---|
| Open Data Structures | Open/CC (verify exact terms in PDF) |
| Introduction to Computer Science (OpenStax) | CC BY-NC-SA 4.0 |
| The Algorithms / JavaScript Algorithms | MIT |
| CP-Algorithms | Verify repo license |
| GeeksforGeeks | Proprietary — personal/research use only |

---

## References

- Technical specification: `spec.md`
- Corpus strategy: `knowledge_base_strategy.md`
- Build log: `EXECUTION_NOTES.md`
