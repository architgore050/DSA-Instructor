"""CLI entry point for running DSA Mentor benchmarks.

Usage:
    python -m benchmark [--config config.json] [--output results.jsonl]
                        [--questions sample_questions.jsonl]
                        [--model large|medium|small]
                        [--rag on|off]
                        [--method knee|fixed_top_5|fixed_top_10|fixed_top_20|flat]
                        [--full]

Options:
    --full          Run the full experiment matrix (3 models x 2 RAG x 5 methods)
    --model         Model tier: large, medium, or small (default: large)
    --rag           RAG mode: on or off (default: on)
    --method        Retrieval method (default: knee)
    --questions     Path to questions JSONL file (default: benchmark/sample_questions.jsonl)
    --output        Output file path (default: benchmark/results.jsonl)
    --config        Path to config.json (default: config.json)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict

try:
    from tqdm import tqdm
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False

    class _NoOpTqdm:
        def __init__(self, *args, **kwargs):
            self._total = kwargs.get("total", 0)
        def __iter__(self):
            return iter(self._items)
        def update(self, n=1):
            pass
        def set_description(self, desc=""):
            pass
        def set_postfix(self, **kwargs):
            pass
    tqdm = _NoOpTqdm  # type: ignore

# Ensure project root is on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from dsa_mentor.config import load_config
from dsa_mentor.embeddings import EmbeddingClient
from dsa_mentor.index.multi import MultiIndexManager
from dsa_mentor.retrieval.hierarchy import KneeHierarchicalRetriever

from benchmark.dataset import BenchmarkDataset
from benchmark.runner import BenchmarkRunner, MODEL_TIERS, RETRIEVAL_METHODS


def build_retriever(config_path: str = "config.json"):
    """Build the retriever from the saved index."""
    cfg = load_config(config_path)
    index_dir = _PROJECT_ROOT / "index"
    kb_path = _PROJECT_ROOT / "knowledge_base.json"

    if not kb_path.exists():
        print("ERROR: knowledge_base.json not found. Run ingestion first.")
        sys.exit(1)

    if not (index_dir / "paragraphs" / "index.faiss").exists():
        print("ERROR: FAISS index not found. Run ingestion + index build first.")
        sys.exit(1)

    emb_client = EmbeddingClient(cfg)
    mgr = MultiIndexManager.load(str(index_dir), embedding_client=emb_client)
    retriever = KneeHierarchicalRetriever(
        multi_index_manager=mgr,
        embedding_client=emb_client,
        config=cfg,
    )
    return retriever, emb_client, cfg


def run_single(config_path: str, output_path: str, question_file: str,
               model_tier: str, rag_enabled: bool, retrieval_method: str) -> None:
    """Run a single configuration."""
    print(f"Loading retriever...")
    retriever, emb_client, cfg = build_retriever(config_path)

    print(f"Loading questions from {question_file}...")
    dataset = BenchmarkDataset(question_file)
    questions = list(dataset)
    print(f"  {len(questions)} questions loaded")

    runner = BenchmarkRunner(cfg, retriever, emb_client)

    print(f"\nRunning: model={model_tier}, rag={'on' if rag_enabled else 'off'}, method={retrieval_method}")
    print("-" * 60)

    results = []
    for i, question in tqdm(enumerate(questions), total=len(questions),
                            desc="Running questions", unit="q",
                            disable=not _HAS_TQDM):
        result = runner.run_question(question, rag_enabled=rag_enabled,
                                      model_tier=model_tier, retrieval_method=retrieval_method)
        results.append(result)
        if _HAS_TQDM:
            tqdm.write(f"  [{i+1}/{len(questions)}] {question.get('id', '?')}: "
                      f"correctness={result['scores']['correctness']}, "
                      f"groundedness={result['scores']['groundedness']:.2f}")

    runner.save_results(results, output_path)

    # Print summary
    correct = sum(1 for r in results if r["scores"]["correctness"] >= 3)
    avg_grounding = sum(r["scores"]["groundedness"] for r in results) / len(results) if results else 0
    avg_latency = sum(r["latency"] for r in results) / len(results) if results else 0

    print(f"\n{'=' * 60}")
    print(f"Results saved to: {output_path}")
    print(f"Questions: {len(results)}")
    print(f"Correct (>=3): {correct}/{len(results)}")
    print(f"Avg groundedness: {avg_grounding:.2f}")
    print(f"Avg latency: {avg_latency:.1f}s")


def run_full_experiment(config_path: str, output_path: str, question_file: str) -> None:
    """Run the full experiment matrix."""
    print(f"Loading retriever...")
    retriever, emb_client, cfg = build_retriever(config_path)

    print(f"Loading questions from {question_file}...")
    dataset = BenchmarkDataset(question_file)
    questions = list(dataset)
    print(f"  {len(questions)} questions loaded")

    runner = BenchmarkRunner(cfg, retriever, emb_client)

    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nRunning full experiment matrix:")
    print(f"  Models: {', '.join(MODEL_TIERS)}")
    print(f"  RAG: ON, OFF")
    print(f"  Methods: {', '.join(RETRIEVAL_METHODS)}")
    total_runs = len(MODEL_TIERS) * (1 + len(RETRIEVAL_METHODS))  # off=1 method, on=len(methods)
    print(f"  Total runs: {total_runs}")
    print("-" * 60)

    start = time.time()
    all_results: Dict[str, Dict[str, Dict[str, list[dict]]]] = {}

    for model_tier in tqdm(MODEL_TIERS, desc="Models", disable=not _HAS_TQDM):
        all_results[model_tier] = {}

        for rag_enabled in (True, False):
            rag_key = "on" if rag_enabled else "off"
            all_results[model_tier][rag_key] = {}

            # For RAG OFF, only run with a neutral retrieval method
            if not rag_enabled:
                methods = ["knee"]
            else:
                methods = RETRIEVAL_METHODS

            for method in tqdm(methods, desc=f"{model_tier}/{rag_key}",
                               leave=False, disable=not _HAS_TQDM):
                print(f"\n  [{model_tier} | {rag_key} | {method}] Starting...")
                results = runner.run_dataset(
                    questions,
                    rag_enabled=rag_enabled,
                    model_tier=model_tier,
                    retrieval_method=method,
                )
                all_results[model_tier][rag_key][method] = results

                # Print per-method summary
                correct = sum(1 for r in results if r["scores"]["correctness"] >= 3)
                avg_grounding = sum(r["scores"]["groundedness"] for r in results) / len(results) if results else 0
                print(f"    ✓ Done: {len(results)} questions, "
                      f"correct={correct}/{len(results)}, "
                      f"groundedness={avg_grounding:.2f}")

    elapsed = time.time() - start

    # Save full results
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, default=str)

    # Save per-run JSONL files
    for model_tier in MODEL_TIERS:
        for rag_key in ("on", "off"):
            for method in RETRIEVAL_METHODS:
                if rag_key == "off" and method != "knee":
                    continue
                results = all_results[model_tier][rag_key].get(method, [])
                if results:
                    per_run_path = str(output_dir / f"results_{model_tier}_{rag_key}_{method}.jsonl")
                    runner.save_results(results, per_run_path)

    print(f"\n{'=' * 60}")
    print(f"Full experiment results saved to: {output_path}")
    print(f"Per-run results saved to: {output_dir}/")
    print(f"Total time: {elapsed:.1f}s")


def main() -> None:
    parser = argparse.ArgumentParser(description="DSA Mentor — Benchmark Runner")
    parser.add_argument("--config", default=str(_PROJECT_ROOT / "config.json"),
                        help="Path to config.json")
    parser.add_argument("--output", default=str(_PROJECT_ROOT / "benchmark" / "results.jsonl"),
                        help="Output file path")
    parser.add_argument("--questions", default=str(_PROJECT_ROOT / "benchmark" / "sample_questions.jsonl"),
                        help="Path to questions JSONL file")
    parser.add_argument("--model", choices=MODEL_TIERS, default="large",
                        help="Model tier")
    parser.add_argument("--rag", choices=("on", "off"), default="on",
                        help="RAG mode")
    parser.add_argument("--method", choices=RETRIEVAL_METHODS, default="knee",
                        help="Retrieval method")
    parser.add_argument("--full", action="store_true",
                        help="Run full experiment matrix")

    args = parser.parse_args()

    rag_enabled = args.rag == "on"

    if args.full:
        run_full_experiment(args.config, args.output, args.questions)
    else:
        run_single(args.config, args.output, args.questions,
                   args.model, rag_enabled, args.method)


if __name__ == "__main__":
    main()
