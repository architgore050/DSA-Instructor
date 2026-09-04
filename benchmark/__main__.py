"""CLI entry point for running DSA Mentor benchmarks.

Usage:
    # LLM response benchmarking (existing)
    python -m benchmark [--config config.json] [--output results.jsonl]
                        [--questions sample_questions.jsonl]
                        [--model large|medium|small]
                        [--rag on|off]
                        [--method knee|fixed_top_5|fixed_top_10|fixed_top_20|flat]
                        [--full]

    # RAGAS retrieval benchmarking (new)
    python -m benchmark --ragas [--config config.json] [--output ragas_results.json]
                         [--questions ragas_questions.jsonl]

    # System logging / latency benchmark (new)
    python -m benchmark --system [--config config.json] [--output system_report.json]
                          [--questions system_questions.jsonl]

    # System health report (new)
    python -m benchmark --health [--config config.json] [--output health_report.json]

    # Run all three benchmarks
    python -m benchmark --full --ragas --system --health

Options:
    --full          Run LLM response benchmark (3 models x 2 RAG x 5 methods)
    --ragas         Run RAGAS retrieval quality benchmark
    --system        Run system logging / latency benchmark
    --health        Generate system health report
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
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

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

from benchmark.ragas_retrieval import RAGASRetrievalBenchmark, get_dsa_retrieval_dataset
from benchmark.system_logging import PipelineProfiler, SystemHealthReport


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


def _create_run_directory(base_output_path: str, run_type: str) -> str:
    """Create a timestamped results directory and save run metadata.

    Directory structure:
        benchmark/results/
            YYYYMMDD_HHMMSS_<run_type>/
                metadata.json           # config, dataset hash, CLI args, timestamp
                full_results.json       # aggregated results (full experiments)
                results_<config>.jsonl  # per-configuration JSONL files
                questions.jsonl         # copy of input questions used
                config.json             # copy of config.json used
                summary.txt             # human-readable summary

    Parameters
    ----------
    base_output_path : str
        The base output path from CLI.
    run_type : str
        One of: "single", "full", "ragas", "system", "health"

    Returns
    -------
    str
        Path to the created results directory.
    """
    results_base = os.path.join(os.path.dirname(base_output_path), "results")
    os.makedirs(results_base, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(results_base, f"{timestamp}_{run_type}")
    os.makedirs(run_dir, exist_ok=True)

    return run_dir


def _save_run_metadata(run_dir: str, run_type: str,
                       config_path: str, questions_path: str,
                       extra_info: Optional[Dict] = None) -> None:
    """Save comprehensive run metadata to the results directory.

    Parameters
    ----------
    run_dir : str
        Results directory path.
    run_type : str
        One of: "single", "full", "ragas", "system", "health"
    config_path : str
        Path to config.json.
    questions_path : str
        Path to questions JSONL.
    extra_info : dict or None
        Additional metadata (model tier, rag, method, etc.).
    """
    # Read config
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config_data = json.load(f)
    except Exception:
        config_data = {"error": "could not read config"}

    # Copy config file
    import shutil
    try:
        shutil.copy2(config_path, os.path.join(run_dir, "config.json"))
    except Exception:
        pass

    # Copy questions file
    try:
        shutil.copy2(questions_path, os.path.join(run_dir, "questions.jsonl"))
    except Exception:
        pass

    # Compute dataset hash
    try:
        import hashlib
        with open(questions_path, "rb") as f:
            dataset_hash = hashlib.sha256(f.read()).hexdigest()[:16]
    except Exception:
        dataset_hash = "unknown"

    # Build metadata
    metadata = {
        "run_type": run_type,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": config_path,
        "questions_path": questions_path,
        "dataset_hash": dataset_hash,
        "config": config_data,
        "extra": extra_info or {},
    }

    with open(os.path.join(run_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False, default=str)

    # Write human-readable summary header
    with open(os.path.join(run_dir, "summary.txt"), "w", encoding="utf-8") as f:
        f.write(f"DSA Mentor Benchmark Results\n")
        f.write(f"{'=' * 60}\n")
        f.write(f"Run type:      {run_type}\n")
        f.write(f"Timestamp:     {metadata['timestamp_utc']}\n")
        f.write(f"Config:        {config_path}\n")
        f.write(f"Questions:     {questions_path}\n")
        f.write(f"Dataset hash:  {dataset_hash}\n")
        if extra_info:
            f.write(f"\nRun configuration:\n")
            for k, v in extra_info.items():
                f.write(f"  {k}: {v}\n")
        f.write(f"\nFiles in this directory:\n")
        for fname in sorted(os.listdir(run_dir)):
            fpath = os.path.join(run_dir, fname)
            size = os.path.getsize(fpath)
            f.write(f"  {fname:>30}  ({size:>10,} bytes)\n")
        f.write(f"\n{'=' * 60}\n")


def run_single(config_path: str, output_path: str, question_file: str,
               model_tier: str, rag_enabled: bool, retrieval_method: str) -> None:
    """Run a single configuration."""
    print(f"Loading retriever...")
    retriever, emb_client, cfg = build_retriever(config_path)

    print(f"Loading questions from {question_file}...")
    dataset = BenchmarkDataset()
    questions = dataset.load(question_file)
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

    # Save to results directory with metadata
    results_dir = _create_run_directory(output_path, "single")
    save_path = os.path.join(results_dir, "results.jsonl")
    runner.save_results(results, save_path)

    # Save metadata
    _save_run_metadata(results_dir, "single", config_path, question_file, {
        "model": model_tier,
        "rag": "on" if rag_enabled else "off",
        "method": retrieval_method,
    })

    # Print summary
    correct = sum(1 for r in results if r["scores"]["correctness"] >= 3)
    avg_grounding = sum(r["scores"]["groundedness"] for r in results) / len(results) if results else 0
    avg_latency = sum(r["latency"] for r in results) / len(results) if results else 0

    print(f"\n{'=' * 60}")
    print(f"Results saved to: {results_dir}/")
    print(f"Questions: {len(results)}")
    print(f"Correct (>=3): {correct}/{len(results)}")
    print(f"Avg groundedness: {avg_grounding:.2f}")
    print(f"Avg latency: {avg_latency:.1f}s")


def run_full_experiment(config_path: str, output_path: str, question_file: str) -> None:
    """Run the full experiment matrix."""
    print(f"Loading retriever...")
    retriever, emb_client, cfg = build_retriever(config_path)

    print(f"Loading questions from {question_file}...")
    dataset = BenchmarkDataset()
    questions = dataset.load(question_file)
    print(f"  {len(questions)} questions loaded")

    runner = BenchmarkRunner(cfg, retriever, emb_client)

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
                print(f"    Done: {len(results)} questions, "
                      f"correct={correct}/{len(results)}, "
                      f"groundedness={avg_grounding:.2f}")

    elapsed = time.time() - start

    # Save full results to results directory
    results_dir = _create_run_directory(output_path, "full")

    # Save full results JSON
    with open(os.path.join(results_dir, "full_results.json"), "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, default=str)

    # Save per-run JSONL files
    for model_tier in MODEL_TIERS:
        for rag_key in ("on", "off"):
            for method in RETRIEVAL_METHODS:
                if rag_key == "off" and method != "knee":
                    continue
                results = all_results[model_tier][rag_key].get(method, [])
                if results:
                    per_run_path = os.path.join(results_dir, f"results_{model_tier}_{rag_key}_{method}.jsonl")
                    runner.save_results(results, per_run_path)

    print(f"\n{'=' * 60}")
    print(f"Full experiment results saved to: {results_dir}/")
    print(f"Total time: {elapsed:.1f}s")


def run_ragas_benchmark(
    config_path: str,
    output_path: str,
    questions_path: Optional[str] = None,
) -> None:
    """Run the RAGAS retrieval quality benchmark.

    Parameters
    ----------
    config_path : str
        Path to config.json.
    output_path : str
        Output file path for results.
    questions_path : str or None
        Path to custom questions JSONL. Uses built-in DSA dataset if None.
    """
    print("Loading retriever for RAGAS benchmark...")
    retriever, emb_client, cfg = build_retriever(config_path)

    print(f"Running RAGAS retrieval benchmark...")
    benchmark = RAGASRetrievalBenchmark(cfg, retriever, emb_client)

    # Use built-in DSA dataset
    dataset = get_dsa_retrieval_dataset()
    print(f"  Dataset size: {len(dataset)} questions")
    print(f"  Categories: {', '.join(sorted(set(q['category'] for q in dataset)))}")

    # Write dataset to temp file for metadata tracking
    tmp_dataset_path = os.path.join(os.path.dirname(output_path), "_ragas_builtin.jsonl")
    with open(tmp_dataset_path, "w", encoding="utf-8") as f:
        for q in dataset:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")

    results = benchmark.run(dataset=dataset, save_path=None)

    # Save to results directory
    results_dir = _create_run_directory(output_path, "ragas")
    results_file = os.path.join(results_dir, "ragas_results.json")
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)

    # Save metadata
    _save_run_metadata(results_dir, "ragas", config_path, tmp_dataset_path, {
        "dataset": "built-in DSA retrieval dataset",
        "dataset_size": len(dataset),
        "categories": ", ".join(sorted(set(q['category'] for q in dataset))),
    })

    # Clean up temp file
    os.remove(tmp_dataset_path)

    mode = results.get("mode", "unknown")
    aggregate = results.get("aggregate_metrics", {})

    print(f"\n{'=' * 60}")
    print(f"RAGAS Retrieval Benchmark Results")
    print(f"{'=' * 60}")
    print(f"Mode: {mode}")
    print(f"Questions: {results.get('dataset_size', 0)}")
    print(f"\nAggregate Metrics:")

    for metric_name, metric_data in aggregate.items():
        if isinstance(metric_data, dict):
            mean_val = metric_data.get("mean", "N/A")
            std_val = metric_data.get("std", "N/A")
            print(f"  {metric_name}: mean={mean_val}, std={std_val}")
        else:
            print(f"  {metric_name}: {metric_data}")

    print(f"\nResults saved to: {results_dir}/")


def run_system_benchmark(
    config_path: str,
    output_path: str,
    questions_path: Optional[str] = None,
) -> None:
    """Run the system logging / latency benchmark.

    Parameters
    ----------
    config_path : str
        Path to config.json.
    output_path : str
        Output file path for results.
    questions_path : str or None
        Path to custom questions JSONL. Uses sample_questions.jsonl if None.
    """
    print("Loading retriever for system benchmark...")
    retriever, emb_client, cfg = build_retriever(config_path)

    # Load questions
    if questions_path:
        dataset = BenchmarkDataset().load(questions_path)
        q_path = questions_path
    else:
        default_q = str(_PROJECT_ROOT / "benchmark" / "sample_questions.jsonl")
        dataset = BenchmarkDataset().load(default_q)
        q_path = default_q

    print(f"  Questions: {len(dataset)}")

    print(f"Running system logging benchmark...")
    profiler = PipelineProfiler(cfg, retriever, emb_client)
    report = profiler.profile_dataset(dataset, save_path=None)

    # Save to results directory
    results_dir = _create_run_directory(output_path, "system")
    report_file = os.path.join(results_dir, "system_report.json")
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)

    # Save metadata
    _save_run_metadata(results_dir, "system", config_path, q_path, {
        "questions_count": len(dataset),
    })

    aggregate = report.get("aggregate", {})

    print(f"\n{'=' * 60}")
    print(f"System Logging Benchmark Results")
    print(f"{'=' * 60}")

    # Latency stats
    latency = aggregate.get("latency", {})
    if latency:
        print(f"\nLatency Statistics:")
        print(f"  Mean:      {latency.get('mean_seconds', 'N/A')}s")
        print(f"  Median:    {latency.get('median_seconds', 'N/A')}s")
        print(f"  P90:       {latency.get('p90_seconds', 'N/A')}s")
        print(f"  P99:       {latency.get('p99_seconds', 'N/A')}s")
        print(f"  Min:       {latency.get('min_seconds', 'N/A')}s")
        print(f"  Max:       {latency.get('max_seconds', 'N/A')}s")

    # Query complexity
    complexity = aggregate.get("query_complexity", {})
    if complexity:
        print(f"\nQuery Complexity:")
        print(f"  Avg word count:      {complexity.get('avg_word_count', 'N/A')}")
        print(f"  Avg tech term density: {complexity.get('avg_tech_term_density', 'N/A')}")
        print(f"  Multi-concept queries: {complexity.get('multi_concept_queries', 'N/A')} "
              f"({complexity.get('multi_concept_pct', 'N/A')}%)")

    # Context building
    context = aggregate.get("context_building", {})
    if context:
        print(f"\nContext Building:")
        print(f"  Avg paragraphs:      {context.get('avg_paragraphs', 'N/A')}")
        print(f"  Avg tokens:          {context.get('avg_tokens', 'N/A')}")
        print(f"  Budget utilization:  {context.get('avg_budget_utilization', 'N/A')}")
        print(f"  Unique sources:      {context.get('avg_unique_sources', 'N/A')}")

    # Retrieval quality
    quality = aggregate.get("retrieval_quality", {})
    if quality:
        print(f"\nRetrieval Quality:")
        print(f"  Avg similarity:      {quality.get('avg_similarity', 'N/A')}")
        print(f"  Avg paragraphs:      {quality.get('avg_paragraphs', 'N/A')}")

    print(f"\nResults saved to: {results_dir}/")


def run_health_report(
    config_path: str,
    output_path: str,
) -> None:
    """Generate a system health report.

    Parameters
    ----------
    config_path : str
        Path to config.json.
    output_path : str
        Output file path for the report.
    """
    print("Loading retriever for health report...")
    retriever, emb_client, cfg = build_retriever(config_path)

    print("Generating system health report...")
    health = SystemHealthReport(cfg, retriever)
    report = health.generate()

    # Save to results directory
    results_dir = _create_run_directory(output_path, "health")
    report_file = os.path.join(results_dir, "health_report.json")
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)

    # Save metadata
    _save_run_metadata(results_dir, "health", config_path, "", {
        "report_type": "system_health",
    })

    # Print summary
    print(f"\n{'=' * 60}")
    print(f"System Health Report")
    print(f"{'=' * 60}")

    # Index structure
    index_structure = report.get("index_structure", {})
    if index_structure and "error" not in index_structure:
        print(f"\nIndex Structure:")
        for level, info in index_structure.items():
            print(f"  {level:>12}: {info.get('num_vectors', 0):>8,} vectors, "
                  f"dim={info.get('dimension', 0)}")

    # KB stats
    kb_stats = report.get("knowledge_base_stats", {})
    if kb_stats and "error" not in kb_stats:
        print(f"\nKnowledge Base:")
        for key, value in kb_stats.items():
            print(f"  {key:>20}: {value:>8,}")

    # Recommendations
    recommendations = report.get("recommendations", [])
    if recommendations:
        print(f"\nRecommendations:")
        for rec in recommendations:
            print(f"  - {rec}")

    print(f"\nHealth report saved to: {results_dir}/")


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
                        help="Run full experiment matrix (LLM response benchmarking)")
    parser.add_argument("--ragas", action="store_true",
                        help="Run RAGAS retrieval quality benchmark")
    parser.add_argument("--system", action="store_true",
                        help="Run system logging / latency benchmark")
    parser.add_argument("--health", action="store_true",
                        help="Generate system health report")

    args = parser.parse_args()

    # RAGAS retrieval benchmark
    if args.ragas:
        ragas_output = str(_PROJECT_ROOT / "benchmark" / "ragas_results.json")
        run_ragas_benchmark(args.config, ragas_output)

    # System logging benchmark
    if args.system:
        system_output = str(_PROJECT_ROOT / "benchmark" / "system_report.json")
        run_system_benchmark(args.config, system_output, args.questions)

    # System health report
    if args.health:
        health_output = str(_PROJECT_ROOT / "benchmark" / "health_report.json")
        run_health_report(args.config, health_output)

    # LLM response benchmarking (existing behavior)
    if args.full:
        run_full_experiment(args.config, args.output, args.questions)
    elif not (args.ragas or args.system or args.health):
        rag_enabled = args.rag == "on"
        run_single(args.config, args.output, args.questions,
                   args.model, rag_enabled, args.method)


if __name__ == "__main__":
    main()
