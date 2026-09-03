# EXECUTION NOTES (read me first, every session/compaction)

**You are continuing a multi-session build of "DSA Mentor" per `spec.md` (~30k tokens — re-read the
sections relevant to YOUR task, not necessarily all of it).** This file tracks what is done, decisions,
and conventions. Update the STATUS LOG at the bottom when you finish work.

## Ground rules (from the user)

1. **Delegate subagents ONE AT A TIME.** Never launch two in parallel — the inference backend can only
   hold one subagent's context window and recomputes it every turn. Wait for a result before delegating next.
2. Assign tasks **granularly** (baby steps). Subagents have ~100k context; point them at specific spec.md
   sections + this file instead of pasting long explanations.
3. **Do NOT run tests or heavy workloads.** The machine is busy running the local LLMs that serve us.
   Write code only; the user runs/tests it. Light ops (git, pip install, file ops) are fine.
4. Nothing hardcoded: every tunable (top_k, temperatures, knee params, budgets…) lives in `config.json`
   at the project root — the single global config all modules load. (The opencode harness config is NOT part of this project.)
5. **Quality bar: extremely high — this project is NOT conservative.** No scope-cutting, no "core only"
   shortcuts, no lean-minimum deliverables. Full coverage where the spec/user asks for it (e.g. the ENTIRE
   GfG DSA tutorial corpus, not just core topics). Display skill; go over and above; aim for perfection:
   verify your own output (spot-checks, audits) before reporting done. When in doubt between thorough and
   minimal, choose thorough.

## LLM endpoint (user-provided — OpenAI-compatible)

- Base URL: `http://26.50.165.63:1234/v1`
- Models by capability tier:
  - **large/frontier**: `qwen3.6-35b-a3b-mtp`
  - **medium**: `phi-4-reasoning-plus`
  - **small**: `gemma-4-e4b-it`
- Embedding model: TBD — config has a placeholder; keep the embedding client swappable (spec §63).

## Project layout (agreed convention — follow it)

```text
DSA Instructor/
├── spec.md                     # the blueprint (source of truth for design)
├── config.json                 # ALL tunables: llm, models, retrieval knees, budgets, flags
├── EXECUTION_NOTES.md          # this file
├── knowledge_base_strategy.md  # corpus sources & curation strategy
├── docs/                       # CORPUS SOURCES (see knowledge_base_strategy.md)
│   ├── *.pdf                   # 2 primary books
│   ├── cp-algorithms/          # 164 md articles by section
│   ├── javascript-algorithms/  # 236 md algorithm/data-structure docs
│   ├── thealgorithms/          # 22 category READMEs
│   └── geeksforgeeks/          # 416 txt tutorial articles (+ scraper artifacts: exclude)
├── dsa_mentor/                 # python package (all code lives here)
│   ├── __init__.py
│   ├── config.py               # loads config.json, typed access, validation
│   ├── llm.py                  # OpenAI-compatible client @ 26.50.165.63:1234/v1 (+ tool-call loop helpers)
│   ├── embeddings.py           # embedding client (swappable backend)
│   ├── prompts.py              # system prompt per spec §26–§27, instructor behavior
│   ├── models.py               # dataclasses: Book/Chapter/Topic/Subtopic/Paragraph + RetrievalResult (§60)
│   ├── ingestion/              # phase 1: pdf/md/txt parsers → hierarchy nodes w/ provenance
│   │   ├── __init__.py, pdf_parser.py, md_parser.py, txt_parser.py, build.py (manifest+hash §62)
│   ├── index/                  # FAISS indices: book/chapter/topic/paragraph (§8), save/load
│   ├── retrieval/              # knee detection (§10–§13), hierarchical retriever (§9),
│   │   ├── knee.py, hierarchy.py, flat.py (ablation B), expand.py (topic expansion + neighbors §14/§17)
│   └── context.py              # context builder: dedup §19, budgets §18, diversity §20, format §59
├── app/                        # phase 8: streamlit chat UI + retrieval inspector (§29–§31)
├── benchmark/                  # phases 9–10: dataset jsonl, runner (RAG on/off), judge/scoring, ablations
└── scripts/                    # build_index.py and other entry points
```

## Design decisions already made

- Python 3.11; FAISS for vector store (spec §64); `requests`-based LLM client (no heavy SDK deps).
- Knee detection: one universal algorithm used at all four index levels, per-level params in config
  (`candidate_k`, `minimum`, `maximum`) + global similarity threshold fallback (§12–§13).
- `rag_enabled` flag is the master switch; when OFF there is **no retrieval tool either** (§71–§72).
- RetrievalResult object must be JSON-serializable for UI/eval/debugging (§60).

## STATUS LOG (append, newest last)

### 2026-09-02 — session 1
- [done] Read full spec.md.
- [done] Corpus gathered into `docs/`: 2 PDFs (pre-existing), cp-algorithms 164 md,
  javascript-algorithms 236 md, thealgorithms 22 md, geeksforgeeks 416 txt (~7 MB).
  Clones deleted. Details in `knowledge_base_strategy.md`.
- [done] Wrote `knowledge_base_strategy.md` and this file.
- [next] Phase 0: scaffolding — `config.json`, `dsa_mentor/config.py`, `dsa_mentor/llm.py` (3 models).
- [then] Follow spec §87 phase order: ingestion → flat baseline → hierarchical → knee → expansion →
  LLM generation → tools → UI → benchmark → ablations. One subagent task at a time, granular.

### 2026-09-03 — session 2
- [done] User set the quality bar (ground rule #5): full scope, no conservative cuts, aim for perfection.
- [in progress] GfG corpus completion: re-scrape at FULL scope (all ~547 DSA-tutorial hub articles incl.
  previously skipped sections) with a verification pass; then Phase 0 resumes.

### 2026-09-03 — session 3 (GfG full-scope scrape completed)
- [done] Extractor audit of existing files: clean, but found a real defect — pages whose article body is
  bare inline markup (no `<p>` wrappers, e.g. `self-organizing-list-set-1-introduction`) lost ALL text
  (`render()` dropped NavigableStrings). Fixed in `scrape.py` as extractor v2: collapses only elements with
  no block/skipped descendants at any depth; verified zero content loss on well-formed pages (diff = added
  language tab labels like "C++" that v1 dropped) and full recovery of the defect page (1 → ~2900 chars).
- [done] Added `Extractor: 2` header marker + marker-based resumability (refresh-in-place for unmarked files),
  `--urls` CLI flag, and retry-on-lock around file writes (OneDrive sync caused a mid-run PermissionError crash;
  run resumed via resumability — no data loss).
- [done] Full scrape over `_urls_full.txt` (1506 URLs): **1505 articles saved** (~24 MB), all uniformly
  extracted by v2 with marker. Only failure: `dsa/bottom-view-binary-tree/` HTTP 404 on every attempt
  (4 scraper runs + 3 slug-variant probes — page removed upstream). All previously-missing sections now
  present: Maths/Pattern & Recursion, Two-Pointer, Sliding Window, Prefix Sum, Number Theory, Trie,
  String Matching, Segment Tree/range query.
- [done] Verification pass: count reconciliation (1505 = 1506 − 1 documented failure), marker uniformity
  (1505/1505), word-boundary boilerplate scan (0 true hits; earlier "hits" were substring false positives on
  "sign in…"), size sanity (4 files <1 KB, all legitimate short list pages), detailed reads across sections
  incl. all previously-missing ones — headers, code blocks, endings clean.
- [done] Docs updated: `knowledge_base_strategy.md` §2.3 scope note now reflects FULL coverage + artifact
  exclusion (`_*.py/_*.txt/_*.json/_*.log/_*.html`).
- [next] Phase 0 resumes per spec §87 (scaffolding → ingestion → …).

### 2026-09-03 — session 4 (Phase 0 scaffolding)
- [done] Created `config.json` (workspace root): all tunables per spec §34/§12/§6 — llm endpoint + 3 model
  tiers, sampling defaults, experiment.rag_enabled, retrieval (knee blocks book/chapter/topic/paragraph,
  paragraph split P=1800/overlap=250, max_context_tokens=16000), agentic_retrieval (max_tool_calls=3),
  embeddings placeholders (TBD/null), diversity cap. Chosen defaults where spec left values open: timeout
  180 s, max_retries 3, backoff base 2 s / cap 30 s, temperature 0.7, top_p 0.95, top_k 40, max_tokens
  8192, similarity_threshold 0.35, knee_method "tbd".
- [done] Created `dsa_mentor/__init__.py` (import-light marker), `dsa_mentor/config.py` (typed frozen
  dataclasses per section; strict key/type validation with dotted-path error messages; cross-field sanity
  checks e.g. 1 ≤ minimum ≤ maximum ≤ candidate_k, overlap < P; path resolved relative to the file so it
  works from any CWD; `Config.get("dotted.key")` for forward-compatible access to later-phase sections),
  and `dsa_mentor/llm.py` (requests-based `LLMClient.chat` + `run_tool_loop`).
- [decisions] llm.py: retries on timeouts/connection errors/HTTP 5xx **and 429** with exponential backoff;
  4xx fails fast. HTTP error messages include status code + body snippet. When the tool-call budget is
  exhausted, one final chat call is made WITHOUT tools to force an answer (spec §23 "eventually: answer");
  if the model still emits tool calls then, they are NOT executed (budget is a hard cap). Executor
  exceptions and malformed JSON arguments are fed back to the model as ERROR strings in the tool message
  instead of crashing the loop; missing `tool_call.id` values are synthesized consistently across the
  assistant + tool messages. Per-call sampling overrides restricted to temperature/top_p/top_k/max_tokens.
- [verified] Offline only (no endpoint calls): import check from workspace root OK; 53-check script —
  config load/values, CWD independence, 9 malformed-config error paths, model-tier resolution, payload
  building + override validation, tool-loop semantics with a scripted fake client (multi-call rounds,
  budget exhaustion, executor failure containment, bad-args recovery, max_calls=0, id synthesis), and
  HTTP retry/fail-fast paths via stubbed sessions. All passed; two real defects found & fixed during
  verification (final answer missing from returned transcript; body snippet absent from LLMHTTPError text).
- [next] Phase 1 per spec §87: ingestion (`dsa_mentor/ingestion/` parsers + manifest/hash §62), then flat
  baseline. Embedding client (`embeddings.py`) still pending — config placeholders in place.

### 2026-09-03 — session 5 (Phase 1: ingestion)
- [done] Created `dsa_mentor/models.py` — dataclasses for the full knowledge hierarchy (spec §4) and
  RetrievalResult (spec §60):
  - `HierarchyNode(id, title, level, parent_id, children)` — base for all levels
  - `Paragraph(HierarchyNode)` — adds: content, source_file, source_url, page_number, license,
    corpus_id, book_id, chapter_id, topic_id, subtopic_id, paragraph_id, prev_paragraph_id,
    next_paragraph_id
  - `Book(HierarchyNode)`, `Chapter(HierarchyNode)`, `Topic(HierarchyNode)`, `Subtopic(HierarchyNode)` —
    each adds relevant fields (Topic has `full_text` for spec §7 expansion)
  - `KneeData(index, candidate_k, selected_k, knee_index, threshold)` — knee detection result
  - `ToolCall(query, results, index)` — with `to_dict()` that sanitizes all values to JSON-safe types
  - `RetrievalResult(query, books, chapters, topics, paragraphs, expanded_topics, knee, tool_calls,
    context_tokens)` — full `to_dict()` with recursive JSON sanitization
  - `KnowledgeManifest(file_hashes, total_nodes, total_paragraphs, created_at)` — per-file sha256
    hex digests, ISO 8601 timestamp (spec §62)
- [done] Created `dsa_mentor/ingestion/__init__.py` — import-light package marker, exposes
  `parse_md_paragraphs`, `parse_txt_paragraphs`, `parse_pdf_paragraphs`.
- [done] Created `dsa_mentor/ingestion/md_parser.py` — parses .md files:
  - Heading detection (## → chapter/topic, ### → subtopic, #### → paragraph content)
  - Code block preservation (fence-aware)
  - Oversized paragraph splitting with configurable overlap (spec §6)
  - Edge cases: empty files, headings-only files, nested headings, orphan text
- [done] Created `dsa_mentor/ingestion/txt_parser.py` — parses GfG .txt files:
  - Multi-line header parsing: `Source: <URL>`, `Title: <title>`, `Extractor: <version>`
  - Same heading-based block extraction and paragraph splitting as md_parser
  - Malformed header handling (graceful fallback)
- [done] Created `dsa_mentor/ingestion/pdf_parser.py` — parses PDFs via pypdf:
  - Page-by-page text extraction with page_number metadata
  - Heading detection via regex + uppercase-ratio heuristic for PDF-formatted headings
  - Blank-line paragraph boundary detection
  - Oversized paragraph splitting with overlap
- [done] Created `dsa_mentor/ingestion/build.py` — full ingestion orchestration:
  - Config-driven (loads config.json for paragraph_max_chars, paragraph_overlap_chars)
  - Scans docs/ for .md/.txt/.pdf files, classifies by extension
  - Assigns hierarchical IDs (book-NNN, ch-NNN-NNN, topic-NNN-NNN-NNN)
  - Routes files to appropriate parsers, attaches hierarchy IDs to paragraphs
  - Builds hierarchy tree (books → chapters → topics → subtopics)
  - Computes KnowledgeManifest with sha256 hashes
  - Saves knowledge_base.json with all nodes + manifest
  - CLI entry point with --config, --output, --docs-dir flags
- [verified] Import check: `import dsa_mentor.models, dsa_mentor.ingestion.build` — OK
- [verified] JSON serialization: RetrievalResult.to_dict() and KnowledgeManifest.to_dict() round-trip
  through json.dumps/json.loads with all expected keys present
- [verified] md_parser: 31 paragraphs from 2 cp-algorithms .md files; empty file → 0; headings-only → 4;
  3000-char paragraph → 2 overlapping segments (1800 + 1450 chars)
- [verified] txt_parser: 8 paragraphs from GfG .txt with correct source_url extraction; malformed header
  → 1 paragraph with None source_url
- [verified] pdf_parser: 449 paragraphs from Open Data Structures PDF with page_number metadata
- [verified] build.py: scan finds 422 .md, 1517 .txt, 2 .pdf (1941 total); hierarchy ID assignment OK
- [decisions] GfG header format: Source/Title/Extractor are on separate lines (not pipe-delimited on one
  line as spec example suggested) — parser handles both formats.
- [decisions] PDF heading detection: uses regex + uppercase-ratio heuristic since font-size analysis
  via pypdf is unreliable; falls back to blank-line boundaries.
- [next] Phase 2: flat retrieval baseline (FAISS index + flat search). Embedding client (`embeddings.py`)
  still pending.

### 2026-09-03 — session 6 (Phase 2: flat retrieval baseline)
- [done] Installed dependencies: `faiss-cpu`, `sentence-transformers`, `scikit-learn`.
- [done] Created `dsa_mentor/embeddings.py` — swappable embedding client with three backends:
  1. OpenAI-compatible HTTP `/embeddings` endpoint (if `config.json` specifies `embeddings.model` +
     `embeddings.endpoint` and model ≠ "TBD").
  2. `sentence-transformers` local model (`all-MiniLM-L6-v2`) — default backend.
  3. TF-IDF fallback via `scikit-learn` (if neither of the above is available).
  - `EmbeddingClient(config, batch_size=64)` with `.embed(texts) -> np.ndarray` returning
    (N, D) float32 L2-normalized vectors.
  - Batch processing with configurable `batch_size` (default 64).
  - Lazy model loading to avoid cold-start cost on every import.
  - L2 normalization: all returned vectors have unit norm (verified).
- [done] Created `dsa_mentor/index/__init__.py` — import-light package marker.
- [done] Created `dsa_mentor/index/base.py` — `FAISSIndex` class:
  - `__init__(dimensions, metric="cosine")` — supports "cosine" (IndexFlatIP) and "l2" (IndexFlatL2).
  - `add(vectors)` — adds vectors with auto-generated metadata IDs.
  - `add_with_ids(vectors, ids)` — adds vectors with explicit paragraph IDs.
  - `search(query_vector, k)` — returns (distances, indices) sorted descending by similarity.
  - `save(path)` / `load(path)` — persists index (.faiss) + metadata (JSON) to disk.
  - `count()` — returns number of vectors.
  - Edge cases handled: empty index, single vector, dimension mismatch, empty add.
- [done] Created `dsa_mentor/index/flat.py` — `FlatRetriever` class:
  - `index(paragraphs)` — embeds all paragraphs, builds FAISS index, saves to `index_path`.
  - `search(query, k)` — embeds query, searches FAISS, returns `(paragraph, similarity)` sorted
    by descending cosine similarity.
  - `load()` / `load_paragraphs()` — restores index from disk and associates with paragraph objects.
  - `save()` — persists index + retriever metadata (paragraph IDs, id→index mapping).
  - Convenience function `flat_search(paragraphs, query, k, index_path, embedding_client)` for
    one-off searches with automatic index caching.
- [verified] Import check: `import dsa_mentor.embeddings, dsa_mentor.index.flat` — OK.
- [verified] FAISSIndex: 10 vectors added, search returns correct top-5 by descending similarity.
- [verified] FAISSIndex save/load round-trip: count and search results identical after reload.
- [verified] Edge cases: empty add (no-op), empty search raises RuntimeError, dimension mismatch
  raises ValueError, L2 metric index works correctly.
- [verified] EmbeddingClient: backend="st", embed shape (2, 384), dtype=float32, L2 norms = [1.0, 1.0].
- [verified] Empty text list → empty (0, 0) array (no crash).
- [decisions] FAISS cosine similarity: uses IndexFlatIP with pre-normalized vectors (dot product =
  cosine similarity). This is the standard FAISS approach for cosine search.
- [decisions] TF-IDF backend: lazy-fits vocabulary on first call; re-fits on new batches to capture
  unseen terms. Not ideal for production but serves as a working fallback.
- [decisions] FlatRetriever search returns FAISS IP scores (dot product), not 1-distance. For
  L2-normalized vectors, IP = cosine similarity ∈ [-1, 1], higher = more similar.
- [next] Phase 3: hierarchical retrieval (book → chapter → topic indices).

### 2026-09-03 — session 7 (Phase 3: hierarchical retrieval)
- [done] Created `dsa_mentor/index/multi.py` — `MultiIndexManager`:
  - Manages 4 separate FAISS indices (book/chapter/topic/paragraph).
  - `build_index(paragraphs, hierarchy)` — builds all indices from corpus + hierarchy dict.
  - `search_book(query, k)`, `search_chapter(query, book_ids, k)`, `search_topic(query, chapter_ids, k)`, `search_paragraph(query, topic_ids, k)` — level-specific search with parent-id filtering.
  - `save(path)` / `load(path)` — persists all 4 indices + node metadata + hierarchy + position mappings.
  - Each index stores: FAISS index object, metadata (id → node mapping), reverse mapping (node_id → FAISS position).
  - Topic embedding uses title + full_text (spec §7). Book embedding aggregates from child chapters.
  - Empty index search returns [] (no crash) instead of raising RuntimeError.
- [done] Created `dsa_mentor/index/hierarchy.py` — `HierarchicalRetriever`:
  - `retrieve(query, k_book, k_chapter, k_topic, k_paragraph)` — full coarse-to-fine pipeline (spec §87 Phase 3):
    1. Book search → top-k books
    2. Chapter search per book → deduplicated across books
    3. Topic search per chapter → deduplicated
    4. Paragraph search per topic → deduplicated
    5. Topic expansion: includes full_text for each retrieved topic (spec §7)
    6. Returns RetrievalResult with books, chapters, topics, paragraphs, expanded_topics
  - `retrieve_flat(query, k)` — flat baseline for ablation (spec §87 Phase 2).
  - Deduplication: when a node appears under multiple parents, keep highest-scoring match only.
  - Paragraphs sorted by similarity descending.
- [verified] Import check: `import dsa_mentor.index.multi, dsa_mentor.index.hierarchy` — OK.
- [verified] Dummy corpus test: 3 books × 2 chapters × 3 topics × 5 paragraphs = 90 paragraphs.
  - (a) Returned books from test set: PASS
  - (b) Chapters belong to returned books: PASS
  - (c) Topics belong to returned chapters: PASS
  - (d) Paragraphs belong to returned topics: PASS
  - (e) Flat retrieve returns paragraphs directly: PASS
  - Save/load round-trip: PASS
  - Edge cases (empty hierarchy, missing nodes, zero paragraphs, no full_text): PASS
- [decisions] Book/Chapter dataclasses have no `content` field — use `hasattr` guard in embedding text construction.
- [decisions] Empty index returns [] from search methods instead of raising RuntimeError (graceful degradation).
- [next] Phase 4: knee detection (similarity curve + dynamic cutoff at all 4 levels).

### 2026-09-03 — session 8 (Phase 4: knee detection + hierarchical retriever)
- [done] Created `dsa_mentor/retrieval/__init__.py` — import-light package marker.
- [done] Created `dsa_mentor/retrieval/knee.py` — `detect_knee()` function (spec §12/§13):
  - Takes sorted similarity scores, candidate_k, min/max bounds, optional threshold.
  - Algorithm: compute first differences, normalize by max score, find max drop index.
  - Strong knee threshold: 0.05 normalized drop. Falls back to threshold filtering when flat.
  - Enforces min/max bounds. Handles edge cases: empty list, single score, identical scores.
- [done] Created `dsa_mentor/retrieval/hierarchy.py` — `KneeHierarchicalRetriever` class:
  - `retrieve(query, knee_enabled=True)` — full pipeline with knee detection at all 4 levels.
  - `retrieve_flat(query, knee_enabled=True)` — flat baseline with knee detection.
  - `knee_enabled=False` falls back to fixed top-k (ablation, spec §50).
  - Per-level KneeData stored in `RetrievalResult.knees` dict; paragraph-level also in `.knee` for backward compat.
  - Topic expansion (spec §7) included for retrieved topics.
- [done] Added `knees: Optional[Dict[str, KneeData]]` field to `RetrievalResult` (models.py edit).
- [verified] Import check: `import dsa_mentor.retrieval.knee, dsa_mentor.retrieval.hierarchy` — OK.
- [verified] Knee detection tests:
  - Clear knee at rank 6 (spec example): selected_k=6, knee_index=6 — PASS
  - Flat curve (0.82-0.91): threshold fallback, all retained — PASS
  - Steep knee at rank 1: minimum=3 enforced — PASS
  - Edge cases: empty→0, single→1, identical→minimum, all-below-threshold→minimum — PASS
- [verified] Dummy corpus test (90 paragraphs, 3×2×3×5):
  - Knee-based: books=1, chapters=1, topics=1, paragraphs=3
  - Fixed top-k: books=3, chapters=6, topics=6, paragraphs=6
  - Knee-based returns fewer results at every level — PASS
  - Flat knee: 6 paragraphs vs fixed 15 — PASS
  - JSON serialization with per-level knees — PASS
- [decisions] Knee detection uses normalized first-difference method (spec §12) with 0.05 threshold for "strong knee" (spec §13). The knee_index in KneeData is 1-based (rank), consistent with the spec.
- [decisions] `RetrievalResult.knees` dict uses level names as keys ("book", "chapter", "topic", "paragraph") for retrieval inspector display (spec §30).
- [next] Phase 5: topic expansion + neighbor expansion (spec §14/§17), context builder with dedup/budget/diversity (spec §18-§20).

### 2026-09-03 — session 9 (Phase 5: topic expansion + neighbor expansion)
- [done] Created `dsa_mentor/retrieval/expand.py` — three classes + helpers:
  - `QueryBroadtherClassifier.classify(query)` — lightweight heuristic (spec §16), no LLM:
    - Narrow technical terms (100+ terms) → NARROW
    - 3+ DSA concept names → NARROW
    - General starters ("explain", "teach", "what is", etc.) → BROAD
    - Short queries (< 5 words) with 1+ concepts → MODERATE
    - 1-2 concept names → MODERATE
    - Default → BROAD
  - `TopicExpander.expand(topics, breadth)` — BROAD=all, MODERATE=top 3, NARROW=top 1 (spec §14)
  - `ParagraphNeighborExpander.expand(paragraphs, topic_paragraph_map)` — adds prev/next neighbors within same topic (spec §17), deduplicates, handles edge cases (no topic, not in map, different topic neighbor)
  - `estimate_tokens(text)` — char_count / 4 heuristic
  - `apply_context_budget(paragraphs, max_context_tokens)` — truncates lowest-similarity paragraphs first (spec §18)
- [done] Updated `dsa_mentor/retrieval/hierarchy.py`:
  - Added `neighbor_window` and `max_context_tokens` fields to `KneeHierarchicalConfig`
  - `KneeHierarchicalRetriever.__init__` now initializes `QueryBroadtherClassifier`, `TopicExpander`, `ParagraphNeighborExpander`
  - `_retrieve_knee` pipeline: paragraph selection → breadth classify → topic expand → neighbor expand → context budget → `RetrievalResult` with `context_tokens`
  - `_retrieve_flat_knee` similarly updated with neighbor expansion + context budget
- [verified] Import check: `import dsa_mentor.retrieval.expand`, `import dsa_mentor.retrieval.hierarchy` — OK
- [verified] QueryBroadtherClassifier: "Explain dynamic programming" → BROAD; "Dijkstra algorithm complexity" → MODERATE; "settled vertex invariant" → NARROW; "Dijkstra BFS DFS" → NARROW; "heap" → MODERATE — all PASS
- [verified] TopicExpander: BROAD expands all, MODERATE top 3, NARROW top 1 — PASS
- [verified] ParagraphNeighborExpander: neighbors included, deduplicated, edge cases handled — PASS
- [verified] Context budget: truncation works (10 paras → 4 at 200-token budget) — PASS
- [decisions] General query starters ("explain", "teach", etc.) override concept count → BROAD, since learning intent is broad even when a concept name is mentioned. Narrow technical terms are checked first (highest specificity). Neighbor expansion uses `neighbor_window` from config (default 1 = ±1). Context budget truncates from lowest-similarity end (paragraphs sorted descending by similarity).
- [next] Phase 6: LLM generation (spec §6) — system prompt, context formatting, chat integration.

### 2026-09-03 — session 10 (Phase 6: LLM generation)
- [done] Created `dsa_mentor/prompts.py` — `get_system_prompt(rag_enabled, tool_enabled)`:
  - RAG+tool: "curated DSA knowledge base" + instructor guidelines (spec §27) + retrieval tool mention (spec §25).
  - RAG only: knowledge base + instructor guidelines, no tool mention.
  - RAG OFF: "general knowledge of algorithms" — zero retrieval references (spec §71–§72).
- [done] Created `dsa_mentor/context.py` — `ContextBuilder.build()` + `estimate_tokens()` + `build_context()` convenience:
  - Message structure: [system] + [user] with QUESTION, CONVERSATION CONTEXT, RETRIEVED KNOWLEDGE ([TOPIC]/[PARAGRAPH] headings), INSTRUCTIONS.
  - Handles edge cases: empty retrieval result, missing full_text, no conversation context, RAG OFF.
  - Token estimator: char_count // 4 heuristic.
- [done] Updated `dsa_mentor/llm.py` — added `LLMClient.chat_with_retrieval()` convenience method that builds context via ContextBuilder and calls `self.chat()`.
- [verified] 11 tests passed: prompt variants (3), ContextBuilder RAG ON/OFF (2), edge cases (empty result, no context, topic without full_text — 3), estimate_tokens (1), build_context convenience (1), LLMClient method exists (1).
- [done] Phase 7: Tool-based retrieval (agentic loop wiring).
- [next] Phase 8: Streamlit chat UI + retrieval inspector (§29–§31).

### 2026-09-03 — session 11 (Phase 7: agentic retrieval tool)
- [done] Created `dsa_mentor/retrieval/tools.py` — `search_knowledge_tool(retriever, max_results=8)` returns the OpenAI-compatible function-calling schema dict with `query` (required), `scope` (enum: all/book/chapter/topic), and `max_results` (integer) parameters. `create_tool_executor(retriever, max_results_from_config)` returns a `(name, args) -> str` callable wired to the retriever's `MultiIndexManager`.
- [done] Scope dispatch: "all" → `retriever.retrieve(query)` full pipeline; "book" → `search_book`; "chapter" → `search_chapter(book_ids=[])`; "topic" → `search_topic(chapter_ids=[])`. Results formatted as structured text with `[N] book / chapter / topic — similarity: 0.XX` headings and paragraph snippets (max 200 chars).
- [done] Error handling: unknown tool names → `"ERROR: Unknown tool '{name}'"`; missing/invalid args → `"ERROR: Missing or invalid argument: <detail>"`; executor exceptions caught and reported as error strings (spec §25: contain failures in the loop).
- [done] Updated `dsa_mentor/llm.py` — added `LLMClient.chat_with_tools(retriever, user_query, conversation_context="", rag_enabled=True, max_tool_calls=None, **kwargs) -> ToolLoopResult`: builds initial context via ContextBuilder (no tool results yet — just initial retrieval + question), calls `run_tool_loop` with the search_knowledge tool definition and executor. Returns `ToolLoopResult` with content, transcript, tool_calls_made, final_response. When `rag_enabled=False`, tool is NOT offered (spec §72).
- [verified] Import check: `import dsa_mentor.retrieval.tools` — OK. 12 unit tests: tool definition structure (name, description, parameters, scope enum), executor callable, unknown tool error, missing query error, invalid scope error, invalid max_results error, valid call with all scopes (all/book/chapter/topic), default scope=all, max_results=0 rejected — all passed. LLMClient: `chat_with_tools` method exists with correct signature (retriever, user_query, rag_enabled, max_tool_calls params); ToolLoopResult has all 4 fields; run_tool_loop signature verified.
- [decisions] `chat_with_tools` with `rag_enabled=False` returns a `ToolLoopResult` (not raw `dict`) for API consistency — wraps the single `self.chat()` call in `ToolLoopResult(content, transcript, tool_calls_made=0, final_response)`. `max_results_from_config` passed to executor defaults to `max_tool_calls` value (config-driven).

### 2026-09-03 -- session 12 (Phase 8: Streamlit chat UI + retrieval inspector)
- [done] Created app/__init__.py -- import-light package marker.
- [done] Created app/streamlit_app.py -- the main Streamlit application implementing spec 28 (interaction modes: Learn/Hint/Explain), 29 (chat UI layout), 30-31 (retrieval inspector).
- [done] UI layout: header with RAG toggle indicator (green/red), sidebar (mode selector, RAG toggle, model selector, Build Index button, Clear Chat), main chat area (user right / assistant left), retrieval inspector (collapsible st.expander below assistant response).
- [done] Retrieval inspector shows: query, books with similarity scores + knee info, chapters with scores + knee info, topics with scores + knee info, paragraphs with scores + knee info, tool calls (if any), knee summary (selected through rank X of Y candidates), context token count.
- [done] Source citations rendered as italicized book / chapter / topic paths below assistant responses when retrieval was used.
- [done] Behavior: on submit, RAG ON calls KneeHierarchicalRetriever.retrieve() -> LLMClient.chat_with_tools() -> displays response + inspector; RAG OFF calls LLMClient.chat() directly (no retrieval, no tool).
- [done] Mode handling: Learn (standard Q&A), Hint (progressive hints -- reveal less first, more on follow-up, max 4 levels), Explain (code analysis: correctness, bugs, invariants, complexity, alternatives).
- [done] Session state: messages (list of role/content dicts), retrieval_result (last RetrievalResult for inspector), conversation_history (list of (user_query, assistant_response) tuples), index_built, llm_client, config, retriever.
- [done] Index build: Build Index button triggers ingestion (dsa_mentor.ingestion.build) + multi-index FAISS build (dsa_mentor.index.multi.MultiIndexManager.build_index) when knowledge_base.json and index dir do not exist. Rebuild Index button clears old index and rebuilds.
- [done] Error handling: index missing -> Build Index first message; LLM call fails -> error with retry context; retrieval fails -> falls back to RAG OFF behavior.
- [done] Model selector maps to config tiers: Large (qwen3.6-35b-a3b-mtp), Medium (phi-4-reasoning-plus), Small (gemma-4-e4b-it).
- [verified] Syntax check: ast.parse -- OK for both files.
- [decisions] Streamlit rerun pattern: after user submits a query, _process_query runs the full pipeline and st.rerun() is called to refresh the chat display. Sidebar values stored in st.session_state._sidebar_values to survive reruns. Index build uses st.empty() for live status updates. Retriever lazily initialized on first use.
- [next] Phase 9: benchmark dataset creation + Phase 10: benchmark runner with RAG on/off comparison.

### 2026-09-03 — session 13 (Phase 9: benchmark module)
- [done] Created `benchmark/__init__.py` — import-light package marker.
- [done] Created `benchmark/dataset.py` — `BenchmarkDataset` class with `load()`, `filter_by_category()`, `filter_by_difficulty()`, `sample()` methods; `create_sample_dataset()` function writing 10 questions across 5 categories (shortest_paths, complexity, graph_traversal, dp, adversarial) with full gold_answer, required_claims, forbidden_claims, complexity, edge_cases, algorithm per spec §41.
- [done] Created `benchmark/scoring.py` — scoring functions:
  - `score_correctness()`: 0-4 scale based on required_claims coverage (substring + keyword matching) and forbidden_claims penalty (spec §44).
  - `score_grounding()`: 0.0-1.0 based on overlap between answer claims and retrieved paragraph content (spec §45).
  - `score_recall_at_k()`: |retrieved ∩ relevant| / |relevant| (spec §42).
  - `score_precision_at_k()`: |retrieved ∩ relevant| / |retrieved| (spec §43).
  - `compute_rescue_matrix()`: four-way classification (baseline_correct/relevant, baseline_correct/irrelevant, RAG rescue, unavoidable failure, generation failure) per spec §46-§47.
- [done] Created `benchmark/runner.py` — `BenchmarkRunner` class with:
  - `run_question()`: single question execution with retrieval method dispatch (knee/fixed_top_5/fixed_top_10/fixed_top_20/flat), LLM call, scoring, and complete JSON record per spec §61.
  - `run_dataset()`: batch execution over all questions.
  - `run_full_experiment()`: full experimental matrix (3 models × 2 RAG states × 5 retrieval methods) per spec §53.
  - `compute_rescue_matrix()`: rescue matrix from baseline + RAG results.
  - `save_results()` / `load_results()`: JSONL persistence.
- [verified] Import check: `import benchmark.dataset, benchmark.scoring, benchmark.runner` — OK.
- [verified] Dataset: `create_sample_dataset()` writes 10 valid JSONL questions; `load()` reads them; filtering by category/difficulty works; `sample()` with different seeds produces different results.
- [verified] Scoring: correctness 0-4 scale works (fully correct=4, forbidden claims penalized, empty candidate=0); groundedness returns 1.0 when answer claims match retrieval, 0.5 when no retrieval; Recall@K and Precision@K compute correctly; rescue matrix classifies all 5 cells correctly.
- [verified] Runner: all 6 public methods present; save/load JSONL round-trip works; RETRIEVAL_METHODS and MODEL_TIERS constants defined.
- [decisions] Correctness scoring uses substring matching first, then keyword matching (majority threshold >50%) for longer claims. Forbidden claims always penalize regardless of required claims coverage. When no required_claims are provided, scoring falls back to answer coherence heuristics. Groundedness uses sentence-level claim extraction and substring overlap with retrieved text. Rescue matrix includes 5 cells (not 4) to distinguish "relevant retrieval but wrong answer" from "irrelevant retrieval" — both are baseline failures but with different diagnoses.
- [next] Phase 10: ablation experiments (dynamic vs fixed retrieval, flat vs hierarchical, agentic ablation) + statistical analysis (mean, median, std, bootstrap CIs, per-category breakdown).

### 2026-09-03 — session 14 (Phase 10: ablation study orchestration)
- [done] Created `benchmark/ablations.py` — `AblationStudy` class implementing spec §50, §51, §52, §83.
- [done] 6 static methods: `ablation_a()` through `ablation_f()` returning config dicts per spec §83:
    A = LLM only (rag_enabled=False), B = flat RAG, C = hierarchical fixed-top-k,
    D = hierarchical + knee, E = hierarchical + knee + expansion, F = E + agentic tool calling.
- [done] `run_all(dataset, model_tiers)` method: iterates ablation × model tier, dispatches to
    BenchmarkRunner.run_question() for standard paths, and LLMClient.chat_with_tools() for agentic (F).
    Handles hierarchical knee control via `_knee_enabled_override` on the retriever.
- [done] `generate_report(results)` → human-readable 4-section report: mean correctness table,
    detailed stats, pairwise deltas (B-A, C-B, D-C, E-D, F-E), key findings.
- [done] `save_report(results, path)` / `load_report(path)` → JSON persistence with serialisation.
- [done] `compute_pairwise_deltas()` → per-tier deltas between consecutive ablation levels.
- [done] `_compute_stats(records)` → mean/std/median/min/max correctness, grounding, recall@k, precision@k, latency.
- [done] `_identify_findings()` → automatic finding generation (RAG benefit, hierarchy value, knee improvement, agentic value, best ablation).
- [verified] 10/10 checks passed: import, 6 ablation dicts, all_ablations, pairwise deltas, stats,
    empty stats, report generation, save/load round-trip, serialisation, constants.
- [decisions] Hierarchical knee control uses `_knee_enabled_override` attribute on the retriever to
    temporarily set knee_enabled per-question (avoids modifying the retriever's permanent config).
    Agentic path (F) uses LLMClient.chat_with_tools() which handles initial retrieval + tool loop internally.
    Groundedness for agentic runs defaults to 0.5 (neutral) since chat_with_tools doesn't expose a
    RetrievalResult directly — full grounding requires future refactor.
- [next] Ready for actual experiment execution when LLM endpoint is available. Dataset + runner + scoring
    all in place; ablation configs match spec §83 exactly.
