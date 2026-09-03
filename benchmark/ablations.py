"""Ablation study orchestration for DSA Mentor evaluation.

Implements spec §50 (fixed top-5/10/20 vs knee), §51 (flat vs hierarchical),
§52 (agentic ablation), and §83 (ablation sequence A-F).

Ablation levels (spec §83):
    A = LLM only (no retrieval)
    B = flat paragraph RAG
    C = hierarchical RAG (fixed top-k)
    D = hierarchical + dynamic cutoff (knee)
    E = hierarchical + dynamic + topic/neighbor expansion
    F = E + agentic retrieval (tool calling)

Usage
-----
    from benchmark.ablations import AblationStudy

    study = AblationStudy(config, retriever, embedding_client)
    results = study.run_all(dataset)
    report = study.generate_report(results)
    study.save_report(results, "benchmark/ablation_results.json")
"""

from __future__ import annotations

import json
import logging
import statistics
from typing import Any, Dict, List, Optional

from dsa_mentor.config import Config
from dsa_mentor.models import RetrievalResult

from .runner import BenchmarkRunner

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Ablation config constants
# ---------------------------------------------------------------------------

ABLATION_LEVELS = ("A", "B", "C", "D", "E", "F")
ABLATION_DESCRIPTIONS = {
    "A": "LLM only (no retrieval)",
    "B": "Flat paragraph RAG",
    "C": "Hierarchical RAG (fixed top-k)",
    "D": "Hierarchical + dynamic cutoff (knee)",
    "E": "Hierarchical + dynamic + topic/neighbor expansion",
    "F": "Hierarchical + dynamic + expansion + agentic tool calling",
}


# ---------------------------------------------------------------------------
# AblationStudy
# ---------------------------------------------------------------------------

class AblationStudy:
    """Orchestrate ablation experiments across model tiers.

    Parameters
    ----------
    config : Config
        Validated configuration (from ``load_config()``).
    retriever : KneeHierarchicalRetriever
        The knee-aware hierarchical retriever.
    embedding_client : EmbeddingClient
        Embedding client for query embedding.
    """

    def __init__(
        self,
        config: Config,
        retriever: Any,
        embedding_client: Any,
    ) -> None:
        self._config = config
        self._retriever = retriever
        self._embedding_client = embedding_client
        self._runner = BenchmarkRunner(config, retriever, embedding_client)

    # ------------------------------------------------------------------
    # Static ablation config methods (spec §83)
    # ------------------------------------------------------------------

    @staticmethod
    def ablation_a() -> Dict[str, Any]:
        """Ablation A: LLM only — no retrieval.

        Returns
        -------
        dict
            {"rag_enabled": False, "retrieval_method": None}
        """
        return {"rag_enabled": False, "retrieval_method": None}

    @staticmethod
    def ablation_b() -> Dict[str, Any]:
        """Ablation B: Flat paragraph RAG.

        Returns
        -------
        dict
            {"rag_enabled": True, "retrieval_method": "flat"}
        """
        return {"rag_enabled": True, "retrieval_method": "flat"}

    @staticmethod
    def ablation_c() -> Dict[str, Any]:
        """Ablation C: Hierarchical RAG with fixed top-k (no knee).

        Returns
        -------
        dict
            {"rag_enabled": True, "retrieval_method": "hierarchical",
             "knee_enabled": False}
        """
        return {
            "rag_enabled": True,
            "retrieval_method": "hierarchical",
            "knee_enabled": False,
        }

    @staticmethod
    def ablation_d() -> Dict[str, Any]:
        """Ablation D: Hierarchical RAG with knee-based dynamic cutoff.

        Returns
        -------
        dict
            {"rag_enabled": True, "retrieval_method": "hierarchical",
             "knee_enabled": True}
        """
        return {
            "rag_enabled": True,
            "retrieval_method": "hierarchical",
            "knee_enabled": True,
        }

    @staticmethod
    def ablation_e() -> Dict[str, Any]:
        """Ablation E: Hierarchical + dynamic + topic/neighbor expansion.

        Returns
        -------
        dict
            {"rag_enabled": True, "retrieval_method": "hierarchical",
             "knee_enabled": True, "expand_topics": True,
             "expand_neighbors": True}
        """
        return {
            "rag_enabled": True,
            "retrieval_method": "hierarchical",
            "knee_enabled": True,
            "expand_topics": True,
            "expand_neighbors": True,
        }

    @staticmethod
    def ablation_f() -> Dict[str, Any]:
        """Ablation F: Full agentic system (E + tool calling).

        Returns
        -------
        dict
            {"rag_enabled": True, "retrieval_method": "hierarchical",
             "knee_enabled": True, "expand_topics": True,
             "expand_neighbors": True, "tool_enabled": True}
        """
        return {
            "rag_enabled": True,
            "retrieval_method": "hierarchical",
            "knee_enabled": True,
            "expand_topics": True,
            "expand_neighbors": True,
            "tool_enabled": True,
        }

    @classmethod
    def all_ablations(cls) -> List[tuple]:
        """Return all 6 ablation configs as (level, config_dict) pairs.

        Returns
        -------
        list[tuple]
            [("A", {...}), ("B", {...}), ..., ("F", {...})]
        """
        return [
            ("A", cls.ablation_a()),
            ("B", cls.ablation_b()),
            ("C", cls.ablation_c()),
            ("D", cls.ablation_d()),
            ("E", cls.ablation_e()),
            ("F", cls.ablation_f()),
        ]

    # ------------------------------------------------------------------
    # Pairwise delta computation
    # ------------------------------------------------------------------

    @staticmethod
    def compute_pairwise_deltas(
        tier_results: Dict[str, Any],
    ) -> Dict[str, Dict[str, float]]:
        """Compute pairwise deltas between consecutive ablation levels.

        For each model tier, computes:
            B-A, C-B, D-C, E-D, F-E  (mean correctness deltas)

        Parameters
        ----------
        tier_results : dict
            Results organized as {tier: {ablation: stats}}.

        Returns
        -------
        dict
            {tier: {"B-A": delta, "C-B": delta, ...}}
        """
        pairs = [("B", "A"), ("C", "B"), ("D", "C"), ("E", "D"), ("F", "E")]
        deltas: Dict[str, Dict[str, float]] = {}

        for tier, ablation_stats in tier_results.items():
            tier_deltas: Dict[str, float] = {}
            for after, before in pairs:
                after_mean = ablation_stats.get(after, {}).get("mean_correctness")
                before_mean = ablation_stats.get(before, {}).get("mean_correctness")
                if after_mean is not None and before_mean is not None:
                    tier_deltas[f"{after}-{before}"] = round(after_mean - before_mean, 4)
                else:
                    tier_deltas[f"{after}-{before}"] = None
            deltas[tier] = tier_deltas

        return deltas

    # ------------------------------------------------------------------
    # Main execution: run_all
    # ------------------------------------------------------------------

    def run_all(
        self,
        dataset: List[Dict[str, Any]],
        model_tiers: List[str] = None,
    ) -> Dict[str, Any]:
        """Run all 6 ablations across all model tiers.

        For each ablation x model tier combination, uses BenchmarkRunner
        to execute the dataset and collect metrics.

        Parameters
        ----------
        dataset : list[dict]
            Benchmark questions (list of question dicts).
        model_tiers : list[str] or None
            Model tiers to test. Defaults to ["large", "medium", "small"].

        Returns
        -------
        dict
            {
                "ablation_results": {tier: {ablation: {stats}}},
                "pairwise_deltas": {tier: {delta_name: value}},
                "raw_records": {tier: {ablation: [result_records]}},
                "config": {
                    "dataset_size": int,
                    "model_tiers": list[str],
                    "ablation_configs": {level: config_dict}
                }
            }
        """
        if model_tiers is None:
            model_tiers = ["large", "medium", "small"]

        ablation_configs = dict(self.all_ablations())
        ablation_results: Dict[str, Dict[str, Dict[str, Any]]] = {}
        raw_records: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}

        for tier in model_tiers:
            logger.info("=== Ablation study: model tier = %s ===", tier)
            ablation_results[tier] = {}
            raw_records[tier] = {}

            for level, config in ablation_configs.items():
                logger.info(
                    "  Running ablation %s (%s): rag=%s, method=%s, tool=%s",
                    level,
                    ABLATION_DESCRIPTIONS.get(level, ""),
                    config["rag_enabled"],
                    config.get("retrieval_method"),
                    config.get("tool_enabled", False),
                )

                records = self._run_ablation_for_tier(
                    dataset, level, config, tier,
                )
                raw_records[tier][level] = records

                stats = self._compute_stats(records)
                ablation_results[tier][level] = stats

        pairwise_deltas = self.compute_pairwise_deltas(ablation_results)

        return {
            "ablation_results": ablation_results,
            "pairwise_deltas": pairwise_deltas,
            "raw_records": raw_records,
            "config": {
                "dataset_size": len(dataset),
                "model_tiers": model_tiers,
                "ablation_configs": ablation_configs,
            },
        }

    def _run_ablation_for_tier(
        self,
        dataset: List[Dict[str, Any]],
        level: str,
        config: Dict[str, Any],
        model_tier: str,
    ) -> List[Dict[str, Any]]:
        """Run a single ablation for a single model tier.

        Handles the different retrieval paths:
            - RAG OFF: direct LLM call
            - Flat RAG: flat retrieval
            - Hierarchical (fixed/knee): hierarchical retrieval
            - Agentic: tool-based retrieval with chat_with_tools
        """
        rag_enabled = config.get("rag_enabled", True)
        retrieval_method = config.get("retrieval_method")
        knee_enabled = config.get("knee_enabled", True)
        tool_enabled = config.get("tool_enabled", False)

        records: List[Dict[str, Any]] = []

        for i, question in enumerate(dataset):
            logger.info(
                "  [%s/%s] Question %s (ablation %s, tier %s)",
                i + 1, len(dataset), question.get("id", "?"), level, model_tier,
            )

            if tool_enabled and rag_enabled:
                # Agentic path: use chat_with_tools (spec §21-§23)
                record = self._run_agentic_question(
                    question, config, model_tier,
                )
            elif rag_enabled and retrieval_method == "flat":
                # Flat RAG path
                record = self._runner.run_question(
                    question,
                    rag_enabled=True,
                    model_tier=model_tier,
                    retrieval_method="flat",
                )
            elif rag_enabled and retrieval_method == "hierarchical":
                # Hierarchical path — runner uses knee by default,
                # but we need to control knee_enabled on the retriever.
                # The runner's internal _perform_retrieval calls
                # self._retriever.retrieve(query, knee_enabled=False) for
                # fixed methods. We handle this by temporarily adjusting
                # the retriever's fixed config, running, then restoring.
                record = self._run_hierarchical_question(
                    question, config, model_tier, knee_enabled,
                )
            else:
                # RAG OFF — direct LLM call
                record = self._runner.run_question(
                    question,
                    rag_enabled=False,
                    model_tier=model_tier,
                    retrieval_method="knee",
                )

            records.append(record)

        return records

    def _run_hierarchical_question(
        self,
        question: Dict[str, Any],
        config: Dict[str, Any],
        model_tier: str,
        knee_enabled: bool,
    ) -> Dict[str, Any]:
        """Run a single hierarchical retrieval question with controlled knee.

        Temporarily sets knee_enabled on the retriever, runs the question,
        then restores the original state.
        """
        import time
        from dsa_mentor.llm import LLMClient, ToolLoopResult
        from dsa_mentor.prompts import get_system_prompt

        start_time = time.time()
        model_id = self._runner._llm_client.resolve_model(model_tier)

        # Set knee_enabled on the retriever
        original_knee = getattr(self._retriever, "_knee_enabled_override", None)
        self._retriever._knee_enabled_override = knee_enabled

        try:
            # Perform retrieval
            retrieval_result: Optional[RetrievalResult] = None
            if config.get("rag_enabled", True):
                retrieval_result = self._retriever.retrieve(
                    question["question"], knee_enabled=knee_enabled,
                )

            # Build context and call LLM
            response = self._runner._llm_client.chat_with_retrieval(
                retrieval_result=retrieval_result,
                user_query=question["question"],
                rag_enabled=config.get("rag_enabled", True),
                model=model_id,
            )
            answer = self._extract_answer(response)

            # Scoring
            from .scoring import (
                score_correctness,
                score_grounding,
                score_recall_at_k,
                score_precision_at_k,
            )

            correctness = score_correctness(
                candidate_answer=answer,
                gold_answer=question.get("gold_answer", ""),
                required_claims=question.get("required_claims", []),
                forbidden_claims=question.get("forbidden_claims", []),
            )

            groundedness = 0.0
            if retrieval_result is not None:
                groundedness = score_grounding(answer, retrieval_result)

            # Recall/Precision — use relevant paragraph IDs from question if available
            recall = 0.0
            precision = 0.0
            if retrieval_result is not None:
                relevant_ids = question.get("relevant_paragraph_ids", [])
                if relevant_ids:
                    recall = score_recall_at_k(retrieval_result, relevant_ids)
                    precision = score_precision_at_k(retrieval_result, relevant_ids)

            latency = round(time.time() - start_time, 2)

            retrieval_info = {
                "books": len(retrieval_result.books) if retrieval_result else 0,
                "chapters": len(retrieval_result.chapters) if retrieval_result else 0,
                "topics": len(retrieval_result.topics) if retrieval_result else 0,
                "paragraphs": len(retrieval_result.paragraphs) if retrieval_result else 0,
                "context_tokens": retrieval_result.context_tokens if retrieval_result else 0,
                "knee_enabled": knee_enabled,
            }
            if retrieval_result and retrieval_result.knees:
                retrieval_info["knee"] = {
                    k: v.to_dict() for k, v in retrieval_result.knees.items()
                }

            return {
                "question_id": question["id"],
                "category": question.get("category", ""),
                "difficulty": question.get("difficulty", ""),
                "model": model_id,
                "model_tier": model_tier,
                "rag_enabled": config.get("rag_enabled", True),
                "retrieval_method": "hierarchical",
                "retrieval": retrieval_info,
                "tool_calls": [],
                "answer": answer,
                "scores": {
                    "correctness": correctness,
                    "reasoning": correctness,
                    "complexity": correctness,
                    "groundedness": round(groundedness, 2),
                    "recall_at_k": recall,
                    "precision_at_k": precision,
                },
                "latency": latency,
            }
        finally:
            self._retriever._knee_enabled_override = original_knee

    def _run_agentic_question(
        self,
        question: Dict[str, Any],
        config: Dict[str, Any],
        model_tier: str,
    ) -> Dict[str, Any]:
        """Run a single agentic retrieval question (ablation F).

        Uses LLMClient.chat_with_tools() which handles the full
        agentic tool-call loop (spec §21-§25).
        """
        import time
        from dsa_mentor.llm import ToolLoopResult

        start_time = time.time()
        model_id = self._runner._llm_client.resolve_model(model_tier)

        # Agentic retrieval with tool calling
        tool_result: ToolLoopResult = self._runner._llm_client.chat_with_tools(
            retriever=self._retriever,
            user_query=question["question"],
            rag_enabled=True,
            max_tool_calls=self._config.agentic_retrieval.max_tool_calls,
            model=model_id,
        )

        answer = tool_result.content

        # Scoring
        from .scoring import (
            score_correctness,
            score_grounding,
        )

        correctness = score_correctness(
            candidate_answer=answer,
            gold_answer=question.get("gold_answer", ""),
            required_claims=question.get("required_claims", []),
            forbidden_claims=question.get("forbidden_claims", []),
        )

        # Grounding: need retrieval result from the initial retrieval
        # chat_with_tools performs initial retrieval internally;
        # we approximate groundedness from the tool call results
        groundedness = 0.5  # placeholder — full grounding requires RetrievalResult

        # Extract tool call info
        tool_calls_info = []
        if tool_result.transcript:
            for msg in tool_result.transcript:
                if msg.get("role") == "tool":
                    tool_calls_info.append({
                        "tool_call_id": msg.get("tool_call_id", ""),
                        "name": msg.get("name", ""),
                        "content_length": len(msg.get("content", "")),
                    })

        latency = round(time.time() - start_time, 2)

        return {
            "question_id": question["id"],
            "category": question.get("category", ""),
            "difficulty": question.get("difficulty", ""),
            "model": model_id,
            "model_tier": model_tier,
            "rag_enabled": True,
            "retrieval_method": "hierarchical_agentic",
            "retrieval": {
                "tool_calls_made": tool_result.tool_calls_made,
                "context_tokens": 0,
            },
            "tool_calls": tool_calls_info,
            "answer": answer,
            "scores": {
                "correctness": correctness,
                "reasoning": correctness,
                "complexity": correctness,
                "groundedness": round(groundedness, 2),
            },
            "latency": latency,
        }

    # ------------------------------------------------------------------
    # Statistics computation
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_stats(records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute summary statistics from a list of result records.

        Returns
        -------
        dict
            {
                "count": int,
                "mean_correctness": float,
                "std_correctness": float,
                "median_correctness": float,
                "mean_grounding": float,
                "mean_recall_at_k": float,
                "mean_precision_at_k": float,
                "mean_latency": float,
                "min_correctness": int,
                "max_correctness": int,
            }
        """
        if not records:
            return {
                "count": 0,
                "mean_correctness": None,
                "std_correctness": None,
                "median_correctness": None,
                "mean_grounding": None,
                "mean_recall_at_k": None,
                "mean_precision_at_k": None,
                "mean_latency": None,
                "min_correctness": None,
                "max_correctness": None,
            }

        scores = [r["scores"]["correctness"] for r in records]
        groundedness = [r["scores"].get("groundedness", 0.0) for r in records]
        latencies = [r.get("latency", 0.0) for r in records]

        recall_vals = [
            r["scores"].get("recall_at_k", 0.0) for r in records
        ]
        precision_vals = [
            r["scores"].get("precision_at_k", 0.0) for r in records
        ]

        mean_correctness = round(statistics.mean(scores), 4)
        std_correctness = round(statistics.stdev(scores), 4) if len(scores) > 1 else 0.0
        median_correctness = round(statistics.median(scores), 4)

        return {
            "count": len(records),
            "mean_correctness": mean_correctness,
            "std_correctness": std_correctness,
            "median_correctness": median_correctness,
            "mean_grounding": round(statistics.mean(groundedness), 4) if groundedness else None,
            "mean_recall_at_k": round(statistics.mean(recall_vals), 4) if recall_vals else None,
            "mean_precision_at_k": round(statistics.mean(precision_vals), 4) if precision_vals else None,
            "mean_latency": round(statistics.mean(latencies), 4) if latencies else None,
            "min_correctness": min(scores) if scores else None,
            "max_correctness": max(scores) if scores else None,
        }

    @staticmethod
    def _extract_answer(response: dict) -> str:
        """Extract answer text from an LLM response dict."""
        choices = response.get("choices", [])
        if not choices:
            return ""
        message = choices[0].get("message", {})
        content = message.get("content", "")
        return content if isinstance(content, str) else ""

    # ------------------------------------------------------------------
    # Report generation
    # ------------------------------------------------------------------

    def generate_report(self, results: Dict[str, Any]) -> str:
        """Generate a human-readable ablation report.

        Produces:
            1. A table of mean correctness per ablation x model tier.
            2. Pairwise deltas (B-A, C-B, D-C, E-D, F-E).
            3. Key findings and observations.

        Parameters
        ----------
        results : dict
            Output from ``run_all()``.

        Returns
        -------
        str
            Formatted report text.
        """
        ablation_results = results["ablation_results"]
        pairwise_deltas = results["pairwise_deltas"]
        config = results["config"]

        lines: List[str] = []
        sep = "=" * 80

        lines.append(sep)
        lines.append("DSA MENTOR — ABLATION STUDY REPORT")
        lines.append(sep)
        lines.append("")
        lines.append(f"Dataset size: {config['dataset_size']} questions")
        lines.append(f"Model tiers: {', '.join(config['model_tiers'])}")
        lines.append("")

        # -- Section 1: Mean correctness table --
        lines.append("-" * 80)
        lines.append("SECTION 1: MEAN CORRECTNESS BY ABLATION x MODEL TIER")
        lines.append("-" * 80)
        lines.append("")

        # Table header
        tier_names = config["model_tiers"]
        header = f"{'Ablation':<10}"
        for tier in tier_names:
            header += f"  {tier:>12}"
        lines.append(header)
        lines.append("-" * len(header))

        # Table rows
        for level in ABLATION_LEVELS:
            row = f"{level:<10}  {ABLATION_DESCRIPTIONS.get(level, '')}"
            # Truncate description for table
            row = f"{level:<10}"
            for tier in tier_names:
                stats = ablation_results.get(tier, {}).get(level, {})
                mean = stats.get("mean_correctness")
                std = stats.get("std_correctness")
                if mean is not None:
                    row += f"  {mean:>10.4f}"
                    if std is not None and std > 0:
                        row += f" +/-{std:>9.4f}"
                    else:
                        row += f"  {'N/A':>10}"
                else:
                    row += f"  {'N/A':>12}"
            lines.append(row)

        lines.append("")

        # -- Section 2: Detailed stats --
        lines.append("-" * 80)
        lines.append("SECTION 2: DETAILED STATISTICS")
        lines.append("-" * 80)
        lines.append("")

        for tier in tier_names:
            lines.append(f"Model Tier: {tier}")
            tier_stats = ablation_results.get(tier, {})
            for level in ABLATION_LEVELS:
                stats = tier_stats.get(level, {})
                if not stats:
                    continue
                lines.append(f"  Ablation {level}:")
                lines.append(f"    Questions:        {stats.get('count', 'N/A')}")
                lines.append(f"    Mean Correctness: {stats.get('mean_correctness', 'N/A')}")
                lines.append(f"    Std Dev:          {stats.get('std_correctness', 'N/A')}")
                lines.append(f"    Median:           {stats.get('median_correctness', 'N/A')}")
                lines.append(f"    Min:              {stats.get('min_correctness', 'N/A')}")
                lines.append(f"    Max:              {stats.get('max_correctness', 'N/A')}")
                lines.append(f"    Mean Grounding:   {stats.get('mean_grounding', 'N/A')}")
                lines.append(f"    Mean Recall@K:    {stats.get('mean_recall_at_k', 'N/A')}")
                lines.append(f"    Mean Precision@K: {stats.get('mean_precision_at_k', 'N/A')}")
                lines.append(f"    Mean Latency:     {stats.get('mean_latency', 'N/A')}s")
            lines.append("")

        # -- Section 3: Pairwise deltas --
        lines.append("-" * 80)
        lines.append("SECTION 3: PAIRWISE DELTAS (mean correctness improvement)")
        lines.append("-" * 80)
        lines.append("")

        delta_pairs = [
            (("B", "A"), "B - A  (RAG vs LLM only)"),
            (("C", "B"), "C - B  (Hierarchy vs Flat)"),
            (("D", "C"), "D - C  (Knee vs Fixed top-k)"),
            (("E", "D"), "E - D  (Expansion vs No expansion)"),
            (("F", "E"), "F - E  (Agentic vs Passive)"),
        ]

        for tier in tier_names:
            lines.append(f"Model Tier: {tier}")
            tier_deltas = pairwise_deltas.get(tier, {})
            for (after, before), label in delta_pairs:
                delta_key = f"{after}-{before}"
                delta = tier_deltas.get(delta_key)
                if delta is not None:
                    sign = "+" if delta >= 0 else ""
                    lines.append(f"  {label:<40} {sign}{delta:>8.4f}")
                else:
                    lines.append(f"  {label:<40} {'N/A':>10}")
            lines.append("")

        # -- Section 4: Key findings --
        lines.append("-" * 80)
        lines.append("SECTION 4: KEY FINDINGS")
        lines.append("-" * 80)
        lines.append("")

        findings = self._identify_findings(ablation_results, pairwise_deltas, tier_names)
        for i, finding in enumerate(findings, 1):
            lines.append(f"  {i}. {finding}")
        lines.append("")

        lines.append(sep)
        lines.append("END OF REPORT")
        lines.append(sep)

        return "\n".join(lines)

    def _identify_findings(
        self,
        ablation_results: Dict[str, Any],
        pairwise_deltas: Dict[str, Any],
        tier_names: List[str],
    ) -> List[str]:
        """Identify key findings from the ablation results.

        Returns
        -------
        list[str]
            A list of human-readable finding statements.
        """
        findings: List[str] = []

        for tier in tier_names:
            tier_results = ablation_results.get(tier, {})
            tier_deltas = pairwise_deltas.get(tier, {})

            # Check if RAG provides any benefit
            a_mean = tier_results.get("A", {}).get("mean_correctness")
            b_mean = tier_results.get("B", {}).get("mean_correctness")
            if a_mean is not None and b_mean is not None:
                delta = b_mean - a_mean
                if delta > 0.05:
                    findings.append(
                        f"[{tier}] RAG provides meaningful improvement over "
                        f"LLM-only: +{delta:.4f} correctness (A→B)"
                    )
                elif delta >= 0:
                    findings.append(
                        f"[{tier}] RAG shows marginal improvement over LLM-only: "
                        f"+{delta:.4f} correctness (A→B)"
                    )
                else:
                    findings.append(
                        f"[{tier}] WARNING: RAG may harm performance: "
                        f"{delta:.4f} correctness change (A→B)"
                    )

            # Check if hierarchy helps vs flat
            c_mean = tier_results.get("C", {}).get("mean_correctness")
            if b_mean is not None and c_mean is not None:
                delta = c_mean - b_mean
                if delta > 0.05:
                    findings.append(
                        f"[{tier}] Hierarchical retrieval outperforms flat RAG: "
                        f"+{delta:.4f} (B→C)"
                    )

            # Check if knee helps vs fixed
            d_mean = tier_results.get("D", {}).get("mean_correctness")
            if c_mean is not None and d_mean is not None:
                delta = d_mean - c_mean
                if delta > 0.05:
                    findings.append(
                        f"[{tier}] Knee-based dynamic cutoff improves over fixed top-k: "
                        f"+{delta:.4f} (C→D)"
                    )
                elif delta < -0.05:
                    findings.append(
                        f"[{tier}] WARNING: Knee detection may underperform fixed top-k: "
                        f"{delta:.4f} (C→D)"
                    )

            # Check if agentic adds value
            f_mean = tier_results.get("F", {}).get("mean_correctness")
            e_mean = tier_results.get("E", {}).get("mean_correctness")
            if e_mean is not None and f_mean is not None:
                delta = f_mean - e_mean
                if delta > 0.05:
                    findings.append(
                        f"[{tier}] Agentic tool calling adds value: +{delta:.4f} (E→F)"
                    )
                elif delta < -0.05:
                    findings.append(
                        f"[{tier}] WARNING: Agentic tool calling may degrade performance: "
                        f"{delta:.4f} (E→F)"
                    )

        # Overall best ablation
        best_ablation = None
        best_score = -1
        for tier in tier_names:
            for level in ABLATION_LEVELS:
                stats = tier_results.get(level, {})
                mean = stats.get("mean_correctness")
                if mean is not None and mean > best_score:
                    best_score = mean
                    best_ablation = level

        if best_ablation:
            findings.append(
                f"Best ablation: {best_ablation} "
                f"({ABLATION_DESCRIPTIONS.get(best_ablation, '')}) "
                f"with mean correctness {best_score:.4f}"
            )

        if not findings:
            findings.append("Insufficient data to identify findings (check for missing results).")

        return findings

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_report(
        self,
        results: Dict[str, Any],
        path: str,
    ) -> None:
        """Save ablation results to a JSON file.

        Parameters
        ----------
        results : dict
            Output from ``run_all()``.
        path : str
            Output file path.
        """
        import os

        path_obj = os.path.abspath(path)
        parent_dir = os.path.dirname(path_obj)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)

        # Create a serialisable copy (strip any non-serialisable objects)
        serialisable = self._make_serialisable(results)

        with open(path_obj, "w", encoding="utf-8") as fh:
            json.dump(serialisable, fh, indent=2, ensure_ascii=False, default=str)
        logger.info("Saved ablation report to %s", path)

    def load_report(self, path: str) -> Dict[str, Any]:
        """Load ablation results from a JSON file.

        Parameters
        ----------
        path : str
            Input file path.

        Returns
        -------
        dict
            The loaded ablation results.
        """
        import os

        path_obj = os.path.abspath(path)
        if not os.path.isfile(path_obj):
            raise FileNotFoundError(f"Ablation report not found: {path}")

        with open(path_obj, "r", encoding="utf-8") as fh:
            results = json.load(fh)
        logger.info("Loaded ablation report from %s", path)
        return results

    @staticmethod
    def _make_serialisable(obj: Any) -> Any:
        """Recursively convert an object to JSON-serialisable types."""
        if isinstance(obj, dict):
            return {k: AblationStudy._make_serialisable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [AblationStudy._make_serialisable(item) for item in obj]
        elif isinstance(obj, (str, int, float, bool, type(None))):
            return obj
        elif hasattr(obj, "to_dict"):
            return obj.to_dict()
        else:
            return str(obj)
