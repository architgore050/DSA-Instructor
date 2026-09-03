# Knowledge-Base Curation Strategy & Sources

This document explains what is in `docs/`, where each source came from, and how it will be
used to build the DSA Mentor knowledge base described in `spec.md` (esp. §3–§8).

## 1. Corpus philosophy (from spec §3)

- **Small but deep**, not internet-scale: a few primary books + carefully selected supplementary articles.
- Quality of knowledge + strength of document structure matter more than quantity.
- Target shape (spec §54): ~2–4 books, ~30–60 chapters, ~150–300 topics, ~2k–10k paragraphs.
- The hierarchy `BOOK → CHAPTER → TOPIC → SUBTOPIC → PARAGRAPH` must be preserved in metadata;
  the paragraph is the atomic evidence unit (spec §4–§5).

## 2. Source inventory

### 2.1 Primary books (PDF) — deep, structured sources

| File | Work | Notes |
|---|---|---|
| `docs/120226_99Z_Morin_2013-Open_Data_Structures.pdf` | *Open Data Structures* (Morin, Munro et al., 2013) | Primary book #1. Strong chapter/topic structure; good for graphs, shortest paths, trees, DP. Open license (verify exact CC terms before redistribution). |
| `docs/Introduction_To_Computer_Science_-_WEB.pdf` | *Introduction to Computer Science* — Dr. Jean-Claude Franchitti (NYU Courant), OpenStax / Rice University, 939 pages | Primary book #2. Broad CS coverage incl. DSA chapters; CC BY-NC-SA 4.0 (non-commercial). |

Role: these two PDFs are the **primary books** of the corpus. Their chapter/section headings map
directly to CHAPTER/TOPIC levels. Paragraphs extracted from them carry full provenance
(`source_file`, `page_number`, hierarchy titles) per spec §4/§55.

### 2.2 Supplementary articles (Markdown)

| Folder | Source repo | Files | Size | What it is |
|---|---|---:|---:|---|
| `docs/cp-algorithms/` | github.com/cp-algorithms/cp-algorithms (`src/`) | 164 `.md` | ~1.6 MB | Competitive-programming reference articles, organized by section (`graph/`, `data_structures/`, `dynamic_programming/`, `string/`, …). The repo has migrated from HTML to markdown; YAML frontmatter kept as-is. |
| `docs/javascript-algorithms/` | github.com/trekhleb/javascript-algorithms (`src/algorithms/**`, `src/data-structures/**`) | 236 `.md` | ~0.7 MB | One doc per algorithm/data structure (the content source of its website). Preserved folder structure. |
| `docs/thealgorithms/` | github.com/TheAlgorithms/Python (per-directory category READMEs) | 22 `.md` | ~41 KB | Category-level overviews (e.g. what belongs in each algorithm family). Thin but useful for topic grouping. |

Curation decisions made during collection:
- Only **algorithm-methodology** docs were copied; repo meta files (root READMEs, CONTRIBUTING,
  CODE_OF_CONDUCT, localized READMEs, CI/tooling docs) were excluded.
- For cp-algorithms, top-level nav/meta pages (`index.md`, `navigation.md`, `tags.md`, …) were skipped.
- cp-algorithms sections that are off-core for DSA tutoring (e.g. `game_theory/`, `geometry/`,
  `num_methods/`) may be **pruned or deprioritized** during ingestion — decide per section when building indices; they can still serve multi-hop queries.

Role: these are the **supplementary articles** of spec §3 (~10–25 carefully selected is the ideal,
we have more and will prune). They map to TOPIC/SUBTOPIC-level units: one article ≈ one topic or
subtopic, with its paragraphs as evidence units. They corroborate and extend the books (esp. for
competitive-programming-style topics like segment trees, DSU, FFT).

### 2.3 Tutorial articles (plain text)

| Folder | Source | Files | Size | What it is |
|---|---|---:|---:|---|
| `docs/geeksforgeeks/` | geeksforgeeks.org DSA tutorial (scraped) | 1505 `.txt` | ~24 MB | Core DSA tutorial articles as plain text. Each file starts with a header block (`Source:` URL, `Title:`, `Extractor: 2`) then headings/lists/code preserved as text. FULL coverage of the GfG DSA tutorial hub (1506 unique article URLs): fundamentals & complexity, arrays/strings, searching, sorting, bit manipulation, hashing, backtracking, linked lists, stacks, queues/deques, binary trees, BSTs, heaps, graphs, greedy, DP — plus Maths/Pattern & Recursion, Two-Pointer, Sliding Window, Prefix Sum, Number Theory, Trie, String Matching, Range Query/Segment Tree, A2Z reference, and topic-wise index harvests (Geometric, Branch & Bound, Randomized, Divide & Conquer, Matrix, etc.). |

Collection artifacts in the same folder (NOT corpus content — exclude from ingestion):
`scrape.py` and every file matching `_*.py`, `_*.txt`, `_*.json`, `_*.log`, `_*.html`
(URL lists, section maps, exploration/inspection scripts, hub HTML snapshots, run logs).

Scope note: FULL coverage achieved (2026-09-03) — all 1506 unique article URLs from the DSA tutorial
hub were scraped; 1505 saved. The only failure is `dsa/bottom-view-binary-tree/` (HTTP 404 on every
attempt, incl. slug variants — page removed upstream); see `_failures.txt`. Two hub pages are
non-articles (`excluded_non_articles`) and 28 section index/hub pages were intentionally not saved as
articles (`index_pages_not_saved`, both lists in `_sections.json`). Extractor v2 (see `scrape.py`)
adds an `Extractor: 2` header marker, language labels on code blocks, and fixes a defect where
pages with bare inline markup lost all text; every file carries the marker.

Role: **tutorial-style corroboration** — multiple independent phrasings of the same concepts help
retrieval recall and source diversity (spec §20), but GfG is proprietary content: keep usage to
personal/research, do not redistribute.

## 3. How sources map onto the hierarchy

```text
BOOK level        ← each PDF book; optionally one "book" per supplementary collection
                    (e.g. "CP-Algorithms", "GeeksforGeeks DSA Tutorial") if that helps retrieval
CHAPTER level     ← PDF chapters/sections; md/txt collections: top-level section folders
TOPIC level       ← PDF subsections; each cp-algorithms / javascript-algorithms article
SUBTOPIC level    ← finer headings inside articles (## / ###)
PARAGRAPH level   ← every paragraph in any source (atomic evidence unit, spec §5)
```

- Oversized paragraphs (> `paragraph_max_chars`, see `config.json`) are split into overlapping
  subparagraph segments only when necessary (spec §6).
- Every node carries provenance metadata: corpus/book/chapter/topic/subtopic ids + titles,
  `source_file`, `source_url` (for GfG), `page_number` (for PDFs), license (spec §4/§55).

## 4. Curation & dedup strategy

1. **Books first.** Ingest the two PDFs as the backbone; their structure defines most CHAPTER/TOPIC nodes.
2. **Supplements attach, don't duplicate.** A supplementary article about "Dijkstra" should link to /
   corroborate the book's Dijkstra topic rather than create a parallel hierarchy. The context builder
   dedups by text overlap (spec §19) so overlapping coverage is safe at query time.
3. **Prune aggressively at ingestion.** Drop: non-DSA content, stub/empty articles, near-duplicate
   GfG pages on the same topic. Keep a manifest of what was dropped and why.
4. **Source diversity is a soft preference** (spec §20): `max_paragraphs_per_source` in config caps
   single-source flooding without forcing artificial diversity.
5. **Manifest + hash.** After ingestion, record a knowledge-base manifest with per-file hashes
   (`knowledge_base_hash`, spec §62) so experiments are reproducible and corpus changes are visible.

## 5. Licensing summary (verify before any redistribution)

| Source | License |
|---|---|
| Open Data Structures (Morin et al.) | Open/CC — verify exact terms in the PDF front matter |
| Introduction to Computer Science (OpenStax) | CC BY-NC-SA 4.0 (non-commercial share-alike) |
| TheAlgorithms/Python | MIT |
| javascript-algorithms | MIT |
| cp-algorithms | Verify repo license for article text |
| GeeksforGeeks articles | Proprietary — personal/research use only, no redistribution |
