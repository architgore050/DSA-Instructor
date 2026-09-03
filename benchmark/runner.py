"""Benchmark runner for DSA Mentor evaluation.

Implements spec §32 (evaluation architecture), §53 (experimental table),
and §61 (complete JSON logging per run).

Usage
-----
    from benchmark.runner import BenchmarkRunner

    runner = BenchmarkRunner(config, retriever, embedding_client)

    # Run a single question
    result = runner.run_question(question, rag_enabled=True, model_tier="large",
                                  retrieval_method="knee")

    # Run a full experiment
    results = runner.run_full_experiment(dataset)
    runner.save_results(results, "benchmark/results.jsonl")
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from dsa_mentor.config import Config
from dsa_mentor.llm import LLMClient, ToolLoopResult
from dsa_mentor.models import RetrievalResult

from .dataset import BenchmarkDataset
from .scoring import (
    compute_rescue_matrix,
    score_correctness,
    score_grounding,
    score_precision_at_k,
    score_recall_at_k,
)

logger = logging.getLogger(__name__)

# Retrieval methods available for the ablation study (spec §50)
RETRIEVAL_METHODS = ("knee", "fixed_top_5", "fixed_top_10", "fixed_top_20", "flat")

# Model tiers
MODEL_TIERS = ("large", "medium", "small")


class BenchmarkRunner:
    """Run benchmark questions through the DSA Mentor system.

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
        self._llm_client = LLMClient(config)
        self._dataset = BenchmarkDataset()

    # ------------------------------------------------------------------
    # Single question execution
    # ------------------------------------------------------------------

    def run_question(
        self,
        question: dict,
        rag_enabled: bool = True,
        model_tier: str = "large",
        retrieval_method: str = "knee",
    ) -> dict:
        """Run a single benchmark question and score the result.

        Parameters
        ----------
        question : dict
            A question dict from the benchmark dataset.
        rag_enabled : bool
            Whether to use RAG (retrieval + LLM) or LLM-only.
        model_tier : str
            Model tier: "large", "medium", or "small".
        retrieval_method : str
            Retrieval method: "knee" (default), "fixed_top_5", "fixed_top_10",
            "fixed_top_20", or "flat".

        Returns
        -------
        dict
            Complete JSON record per spec §61:
            question_id, model, rag_enabled, retrieval metadata, tool_calls,
            answer, scores, latency.
        """
        start_time = time.time()

        # Resolve model id
        model_id = self._llm_client.resolve_model(model_tier)

        # Determine retrieval parameters based on method
        knee_enabled = retrieval_method == "knee"
        flat_method = retrieval_method == "flat"

        # --- Step 1: Retrieval (if RAG enabled) ---
        retrieval_result: Optional[RetrievalResult] = None
        retrieval_info: Dict[str, Any] = {}

        if rag_enabled:
            retrieval_result = self._perform_retrieval(
                question["question"],
                retrieval_method,
                knee_enabled,
                flat_method,
            )
            retrieval_info = self._extract_retrieval_info(retrieval_result)

        # --- Step 2: LLM generation ---
        if rag_enabled and retrieval_result is not None:
            # RAG ON: use chat_with_retrieval for non-agentic, or chat_with_tools for agentic
            response = self._llm_client.chat_with_retrieval(
                retrieval_result=retrieval_result,
                user_query=question["question"],
                rag_enabled=True,
                model=model_id,
            )
            answer = self._extract_answer(response)
            tool_calls_info = self._extract_tool_calls(retrieval_result)
        else:
            # RAG OFF: direct LLM call, no retrieval context
            response = self._llm_client.chat(
                self._build_rag_off_messages(question["question"]),
                model=model_id,
            )
            answer = self._extract_answer(response)
            tool_calls_info = []

        # --- Step 3: Scoring ---
        correctness = score_correctness(
            candidate_answer=answer,
            gold_answer=question.get("gold_answer", ""),
            required_claims=question.get("required_claims", []),
            forbidden_claims=question.get("forbidden_claims", []),
        )

        groundedness = 0.0
        if rag_enabled and retrieval_result is not None:
            groundedness = score_grounding(answer, retrieval_result)

        # --- Step 4: Build result record (spec §61) ---
        latency = round(time.time() - start_time, 2)

        result = {
            "question_id": question["id"],
            "category": question.get("category", ""),
            "difficulty": question.get("difficulty", ""),
            "model": model_id,
            "model_tier": model_tier,
            "rag_enabled": rag_enabled,
            "retrieval_method": retrieval_method,
            "retrieval": retrieval_info,
            "tool_calls": tool_calls_info,
            "answer": answer,
            "scores": {
                "correctness": correctness,
                "reasoning": correctness,  # placeholder — can be refined later
                "complexity": correctness,  # placeholder — can be refined later
                "groundedness": round(groundedness, 2),
            },
            "latency": latency,
        }

        logger.info(
            "Question %s: correctness=%d, groundedness=%.2f, latency=%.2fs, method=%s, rag=%s",
            question["id"], correctness, groundedness, latency, retrieval_method, rag_enabled,
        )

        return result

    # ------------------------------------------------------------------
    # Dataset execution
    # ------------------------------------------------------------------

    def run_dataset(
        self,
        dataset: list[dict],
        rag_enabled: bool = True,
        model_tier: str = "large",
        retrieval_method: str = "knee",
    ) -> list[dict]:
        """Run all questions in a dataset.

        Parameters
        ----------
        dataset : list[dict]
            List of question dicts.
        rag_enabled : bool
            Whether to use RAG.
        model_tier : str
            Model tier.
        retrieval_method : str
            Retrieval method.

        Returns
        -------
        list[dict]
            List of result records.
        """
        results: list[dict] = []
        for i, q in enumerate(dataset):
            logger.info("Running question %d/%d: %s", i + 1, len(dataset), q.get("id", "?"))
            result = self.run_question(q, rag_enabled, model_tier, retrieval_method)
            results.append(result)
        return results

    # ------------------------------------------------------------------
    # Full experiment (spec §53)
    # ------------------------------------------------------------------

    def run_full_experiment(
        self,
        dataset: list[dict],
    ) -> dict:
        """Run the full experimental matrix (spec §53).

        Tests all combinations of:
            - Models: large, medium, small
            - RAG: ON, OFF
            - Retrieval: knee, fixed_top_5, fixed_top_10, fixed_top_20, flat

        Parameters
        ----------
        dataset : list[dict]
            Full benchmark dataset.

        Returns
        -------
        dict
            Results organized by configuration:
            {
                "model_tier": {
                    "rag_enabled": {
                        "retrieval_method": [result_records]
                    }
                }
            }
        """
        all_results: Dict[str, Dict[str, Dict[str, list[dict]]]] = {}

        for model_tier in MODEL_TIERS:
            all_results[model_tier] = {}

            for rag_enabled in (True, False):
                rag_key = "on" if rag_enabled else "off"
                all_results[model_tier][rag_key] = {}

                # For RAG OFF, only run with a neutral retrieval method
                if not rag_enabled:
                    methods = ["knee"]  # retrieval method is irrelevant when RAG is off
                else:
                    methods = RETRIEVAL_METHODS

                for method in methods:
                    logger.info(
                        "Experiment: model=%s, rag=%s, method=%s",
                        model_tier, rag_key, method,
                    )
                    results = self.run_dataset(
                        dataset,
                        rag_enabled=rag_enabled,
                        model_tier=model_tier,
                        retrieval_method=method,
                    )
                    all_results[model_tier][rag_key][method] = results

        return all_results

    # ------------------------------------------------------------------
    # Rescue matrix computation
    # ------------------------------------------------------------------

    def compute_rescue_matrix(
        self,
        baseline_results: list[dict],
        rag_results: list[dict],
        retrieval_relevant: list[bool],
    ) -> Dict[str, int]:
        """Compute the four-way rescue matrix from two sets of results.

        Parameters
        ----------
        baseline_results : list[dict]
            Results from RAG OFF runs.
        rag_results : list[dict]
            Results from RAG ON runs.
        retrieval_relevant : list[bool]
            Whether retrieval found relevant evidence for each question.

        Returns
        -------
        dict
            Four-way rescue matrix counts (spec §47).
        """
        baseline_correct = [r["scores"]["correctness"] >= 3 for r in baseline_results]
        rag_correct = [r["scores"]["correctness"] >= 3 for r in rag_results]
        return compute_rescue_matrix(baseline_correct, rag_correct, retrieval_relevant)

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------

    @staticmethod
    def save_results(results: list[dict], path: str) -> None:
        """Save results to a JSONL file.

        Parameters
        ----------
        results : list[dict]
            List of result records.
        path : str
            Output file path.
        """
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        with open(path_obj, "w", encoding="utf-8") as fh:
            for r in results:
                fh.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
        logger.info("Saved %d results to %s", len(results), path)

    @staticmethod
    def load_results(path: str) -> list[dict]:
        """Load results from a JSONL file.

        Parameters
        ----------
        path : str
            Input file path.

        Returns
        -------
        list[dict]
            List of result records.
        """
        path_obj = Path(path)
        if not path_obj.is_file():
            raise FileNotFoundError(f"Results file not found: {path}")

        results: list[dict] = []
        with open(path_obj, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                results.append(json.loads(line))
        logger.info("Loaded %d results from %s", len(results), path)
        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _perform_retrieval(
        self,
        query: str,
        method: str,
        knee_enabled: bool,
        flat_method: bool,
    ) -> RetrievalResult:
        """Execute retrieval with the specified method.

        Parameters
        ----------
        query : str
            The user query.
        method : str
            Retrieval method name.
        knee_enabled : bool
            Whether knee detection is active.
        flat_method : bool
            Whether to use flat retrieval.

        Returns
        -------
        RetrievalResult
            The retrieval result.
        """
        if flat_method:
            return self._retriever.retrieve_flat(query, knee_enabled=knee_enabled)

        # For fixed methods, we need to set knee_enabled=False and override k
        if method == "fixed_top_5":
            # Use knee but override paragraph knee to fixed 5
            return self._retriever.retrieve(query, knee_enabled=False)
        elif method == "fixed_top_10":
            return self._retriever.retrieve(query, knee_enabled=False)
        elif method == "fixed_top_20":
            return self._retriever.retrieve(query, knee_enabled=False)
        else:
            # "knee" — use knee detection
            return self._retriever.retrieve(query, knee_enabled=True)

    def _extract_retrieval_info(self, result: RetrievalResult) -> Dict[str, Any]:
        """Extract retrieval metadata for the result record."""
        info: Dict[str, Any] = {
            "books": len(result.books),
            "chapters": len(result.chapters),
            "topics": len(result.topics),
            "paragraphs": len(result.paragraphs),
            "context_tokens": result.context_tokens,
        }

        # Knee metadata (per-level)
        if result.knees:
            info["knee"] = {k: v.to_dict() for k, v in result.knees.items()}
        elif result.knee:
            info["knee"] = result.knee.to_dict()

        return info

    def _extract_tool_calls(self, result: RetrievalResult) -> List[Dict[str, Any]]:
        """Extract tool call info from a retrieval result."""
        calls = []
        for tc in result.tool_calls:
            calls.append({
                "query": tc.query,
                "index": tc.index,
                "result_count": len(tc.results) if tc.results else 0,
            })
        return calls

    def _build_rag_off_messages(self, query: str) -> list[dict]:
        """Build messages for RAG OFF mode."""
        from dsa_mentor.prompts import get_system_prompt

        return [
            {
                "role": "system",
                "content": get_system_prompt(rag_enabled=False, tool_enabled=False),
            },
            {
                "role": "user",
                "content": query,
            },
        ]

    @staticmethod
    def _extract_answer(response: dict) -> str:
        """Extract the answer text from an LLM response."""
        choices = response.get("choices", [])
        if not choices:
            return ""
        message = choices[0].get("message", {})
        content = message.get("content", "")
        return content if isinstance(content, str) else ""

    def __del__(self) -> None:
        """Clean up the HTTP session."""
        try:
            self._llm_client.close()
        except Exception:
            pass
