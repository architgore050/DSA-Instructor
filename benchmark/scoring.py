"""Scoring functions for benchmark evaluation.

Implements spec §42-§47:
    - Correctness scoring (0-4 scale, spec §44)
    - Groundedness scoring (0.0-1.0, spec §45)
    - Recall@K and Precision@K (spec §42-§43)
    - Four-way rescue matrix (spec §46-§47)
"""

from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Optional, Tuple

from dsa_mentor.models import RetrievalResult


# ---------------------------------------------------------------------------
# Correctness scoring (spec §44)
# ---------------------------------------------------------------------------

def score_correctness(
    candidate_answer: str,
    gold_answer: str,
    required_claims: List[str],
    forbidden_claims: List[str],
) -> int:
    """Score candidate answer against gold answer on a 0-4 scale.

    Scoring rubric (spec §44):
        0 = wrong (major contradictions, no correct claims)
        1 = major errors (some correct claims but fundamental mistakes)
        2 = partially correct (key claims present but incomplete or with errors)
        3 = correct (all key claims present, minor omissions)
        4 = fully correct (all claims present, no errors, well-reasoned)

    Parameters
    ----------
    candidate_answer : str
        The model's generated answer.
    gold_answer : str
        The reference gold answer.
    required_claims : list[str]
        Claims that should appear in the answer (spec §41).
    forbidden_claims : list[str]
        Claims that should NOT appear in the answer (spec §41).

    Returns
    -------
    int
        Score from 0 to 4.
    """
    candidate_lower = candidate_answer.lower()
    gold_lower = gold_answer.lower()

    # Check required claims coverage
    if required_claims:
        matched = 0
        for claim in required_claims:
            claim_lower = claim.lower()
            # Use substring matching for claim presence
            if claim_lower in candidate_lower:
                matched += 1
            else:
                # Try keyword-level matching for longer claims
                keywords = _extract_keywords(claim)
                if keywords and _check_keywords(keywords, candidate_lower):
                    matched += 1
        coverage = matched / len(required_claims)
    else:
        coverage = 1.0  # no required claims means we rely on other signals

    # Check forbidden claims
    forbidden_violations = 0
    for claim in forbidden_claims:
        claim_lower = claim.lower()
        if claim_lower in candidate_lower:
            forbidden_violations += 1

    # Forbidden claims always penalize — they indicate fundamental misconceptions
    if forbidden_violations > 0:
        if coverage >= 0.75:
            return 2  # correct claims but with a fundamental misconception
        elif coverage >= 0.5:
            return 1  # partially correct with a misconception
        else:
            return 0  # wrong and with a misconception

    # No forbidden claims — score based on required claims coverage
    if coverage >= 1.0:
        # All required claims present, no forbidden claims
        # Check if answer is coherent (not just keyword stuffing)
        if _is_coherent_answer(candidate_answer, gold_answer):
            return 4
        else:
            return 3
    elif coverage >= 0.75:
        return 3
    elif coverage >= 0.5:
        return 2
    elif coverage >= 0.25:
        return 1
    else:
        return 0


def _extract_keywords(claim: str) -> List[str]:
    """Extract meaningful keywords from a claim for matching."""
    # Remove common stop words and short tokens
    stop_words = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been",
        "being", "have", "has", "had", "do", "does", "did", "will",
        "would", "could", "should", "may", "might", "shall", "can",
        "to", "of", "in", "for", "on", "with", "at", "by", "from",
        "as", "into", "through", "during", "before", "after", "above",
        "below", "between", "out", "off", "over", "under", "again",
        "further", "then", "once", "here", "there", "when", "where",
        "why", "how", "all", "each", "every", "both", "few", "more",
        "most", "other", "some", "such", "no", "nor", "not", "only",
        "own", "same", "so", "than", "too", "very", "just", "because",
        "but", "and", "or", "if", "while", "that", "this", "these",
        "those", "it", "its", "what", "which", "who", "whom",
    }
    tokens = re.findall(r"[a-z_]+", claim.lower())
    return [t for t in tokens if len(t) > 2 and t not in stop_words]


def _check_keywords(keywords: List[str], text: str) -> bool:
    """Check if enough keywords are present in the text.

    Requires a majority of keywords (strictly > 50%) to reduce false positives.
    """
    if not keywords:
        return False
    matched = sum(1 for kw in keywords if kw in text)
    # Require strictly more than half the keywords to be present
    return matched > len(keywords) / 2


def _is_coherent_answer(candidate: str, gold: str) -> bool:
    """Heuristic check that the answer is coherent, not just keyword stuffing."""
    # Check that the answer has reasonable length relative to gold
    if len(candidate) < len(gold) * 0.3:
        return False
    # Check that the answer contains a reasonable number of sentences
    sentences = re.split(r"[.!?]+", candidate)
    sentences = [s.strip() for s in sentences if s.strip()]
    return len(sentences) >= 2


# ---------------------------------------------------------------------------
# Groundedness scoring (spec §45)
# ---------------------------------------------------------------------------

def score_grounding(
    candidate_answer: str,
    retrieval_result: RetrievalResult,
) -> float:
    """Score how well the answer is grounded in retrieved evidence (spec §45).

    Returns a score from 0.0 to 1.0 indicating the degree to which
    source-dependent claims in the answer are supported by the retrieved
    paragraphs.

    Parameters
    ----------
    candidate_answer : str
        The model's generated answer.
    retrieval_result : RetrievalResult
        The retrieval result that was used to generate the answer.

    Returns
    -------
    float
        Groundedness score in [0.0, 1.0].
    """
    # If no retrieval was performed, groundedness is N/A → return 0.5 (neutral)
    if not retrieval_result.paragraphs:
        return 0.5

    # Extract key claims from the candidate answer
    answer_text = candidate_answer.lower()

    # Get the set of all text content from retrieved paragraphs
    retrieved_texts = []
    for para in retrieval_result.paragraphs:
        content = getattr(para, "content", "") or ""
        retrieved_texts.append(content.lower())

    if not retrieved_texts:
        return 0.5

    # Combine all retrieved text for overlap analysis
    all_retrieved = " ".join(retrieved_texts)

    # Extract meaningful phrases from the answer (claims that reference specific facts)
    answer_claims = _extract_answer_claims(candidate_answer)

    if not answer_claims:
        # No extractable claims — assume partial grounding if retrieval existed
        return 0.6

    # Check how many claims are supported by retrieved text
    supported = 0
    for claim in answer_claims:
        claim_lower = claim.lower()
        # Direct substring match
        if claim_lower in all_retrieved:
            supported += 1
        else:
            # Check keyword overlap
            keywords = _extract_keywords(claim)
            if keywords and _check_keywords(keywords, all_retrieved):
                supported += 1

    # Grounding score = fraction of claims supported by retrieval
    score = supported / len(answer_claims)
    return round(min(1.0, max(0.0, score)), 2)


def _extract_answer_claims(answer: str) -> List[str]:
    """Extract factual claims from an answer for grounding analysis.

    Splits on sentence boundaries and filters out very short / very long segments.
    """
    # Split into sentences
    sentences = re.split(r"[.!?]+\s*", answer)
    claims = []
    for s in sentences:
        s = s.strip()
        # Filter: skip very short or very long sentences
        if 20 <= len(s) <= 500:
            claims.append(s)
    return claims[:10]  # limit to first 10 claims for efficiency


# ---------------------------------------------------------------------------
# Recall@K and Precision@K (spec §42-§43)
# ---------------------------------------------------------------------------

def score_recall_at_k(
    retrieval_result: RetrievalResult,
    relevant_paragraph_ids: List[str],
    k: Optional[int] = None,
) -> float:
    """Compute Recall@K for paragraph retrieval.

    Recall@K = |retrieved ∩ relevant| / |relevant|

    Parameters
    ----------
    retrieval_result : RetrievalResult
        The retrieval result to evaluate.
    relevant_paragraph_ids : list[str]
        IDs of paragraphs that should have been retrieved.
    k : int or None
        Number of top results to consider. If None, uses all retrieved paragraphs.

    Returns
    -------
    float
        Recall@K score in [0.0, 1.0].
    """
    relevant_set = set(relevant_paragraph_ids)

    # Get retrieved paragraph IDs
    retrieved_ids = []
    for para in retrieval_result.paragraphs:
        para_id = getattr(para, "id", None) or getattr(para, "paragraph_id", None)
        if para_id:
            retrieved_ids.append(para_id)

    if not retrieved_ids and not relevant_set:
        return 1.0  # both empty → perfect recall
    if not relevant_set:
        return 1.0  # no relevant paragraphs → vacuous truth
    if not retrieved_ids:
        return 0.0  # nothing retrieved → zero recall

    # Limit to top-k if specified
    if k is not None:
        retrieved_ids = retrieved_ids[:k]

    retrieved_set = set(retrieved_ids)
    true_positives = len(retrieved_set & relevant_set)
    recall = true_positives / len(relevant_set)
    return round(min(1.0, max(0.0, recall)), 4)


def score_precision_at_k(
    retrieval_result: RetrievalResult,
    relevant_paragraph_ids: List[str],
    k: Optional[int] = None,
) -> float:
    """Compute Precision@K for paragraph retrieval.

    Precision@K = |retrieved ∩ relevant| / |retrieved|

    Parameters
    ----------
    retrieval_result : RetrievalResult
        The retrieval result to evaluate.
    relevant_paragraph_ids : list[str]
        IDs of paragraphs that should have been retrieved.
    k : int or None
        Number of top results to consider. If None, uses all retrieved paragraphs.

    Returns
    -------
    float
        Precision@K score in [0.0, 1.0].
    """
    relevant_set = set(relevant_paragraph_ids)

    # Get retrieved paragraph IDs
    retrieved_ids = []
    for para in retrieval_result.paragraphs:
        para_id = getattr(para, "id", None) or getattr(para, "paragraph_id", None)
        if para_id:
            retrieved_ids.append(para_id)

    if not retrieved_ids:
        return 0.0  # nothing retrieved → zero precision

    # Limit to top-k if specified
    if k is not None:
        retrieved_ids = retrieved_ids[:k]

    retrieved_set = set(retrieved_ids)
    true_positives = len(retrieved_set & relevant_set)
    precision = true_positives / len(retrieved_set)
    return round(min(1.0, max(0.0, precision)), 4)


# ---------------------------------------------------------------------------
# Four-way rescue matrix (spec §46-§47)
# ---------------------------------------------------------------------------

def compute_rescue_matrix(
    baseline_correct: List[bool],
    rag_correct: List[bool],
    retrieval_relevant: List[bool],
) -> Dict[str, int]:
    """Compute the four-way rescue matrix (spec §46-§47).

    Classifies each question into one of four categories:
        - baseline_correct_retrieval_relevant: model knew it, retrieval was relevant
        - baseline_correct_retrieval_irrelevant: model knew it, retrieval was irrelevant
        - baseline_wrong_retrieval_relevant_answer_correct: RAG rescue (exciting cell!)
        - baseline_wrong_retrieval_irrelevant: unavoidable failure
        - baseline_wrong_retrieval_relevant_answer_wrong: generation/reasoning failure

    Parameters
    ----------
    baseline_correct : list[bool]
        Whether the model answered correctly without RAG.
    rag_correct : list[bool]
        Whether the model answered correctly with RAG.
    retrieval_relevant : list[bool]
        Whether retrieval found relevant evidence.

    Returns
    -------
    dict
        Counts for each cell in the rescue matrix.
    """
    matrix = {
        "baseline_correct_retrieval_relevant": 0,
        "baseline_correct_retrieval_irrelevant": 0,
        "baseline_wrong_retrieval_relevant_answer_correct": 0,
        "baseline_wrong_retrieval_irrelevant": 0,
        "baseline_wrong_retrieval_relevant_answer_wrong": 0,
    }

    n = min(len(baseline_correct), len(rag_correct), len(retrieval_relevant))

    for i in range(n):
        b_corr = baseline_correct[i]
        r_corr = rag_correct[i]
        r_rel = retrieval_relevant[i]

        if b_corr and r_rel:
            matrix["baseline_correct_retrieval_relevant"] += 1
        elif b_corr and not r_rel:
            matrix["baseline_correct_retrieval_irrelevant"] += 1
        elif not b_corr and r_rel and r_corr:
            matrix["baseline_wrong_retrieval_relevant_answer_correct"] += 1
        elif not b_corr and not r_rel:
            matrix["baseline_wrong_retrieval_irrelevant"] += 1
        elif not b_corr and r_rel and not r_corr:
            matrix["baseline_wrong_retrieval_relevant_answer_wrong"] += 1

    return matrix
