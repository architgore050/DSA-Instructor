"""Knee detection algorithm for dynamic evidence selection.

Implements spec §12 (universal knee detection at every index level) and
spec §13 (threshold fallback when no strong knee exists).

The same algorithm applies to Book, Chapter, Topic, and Paragraph indices —
only the candidate pool sizes and bounds differ.
"""

from __future__ import annotations

import math
from typing import List, Optional

from ..models import KneeData


def detect_knee(
    similarities: List[float],
    candidate_k: int,
    minimum: int,
    maximum: int,
    similarity_threshold: Optional[float] = None,
) -> KneeData:
    """Detect the knee (elbow) in a sorted similarity-score curve.

    Algorithm (spec §12):
        1. Take top ``candidate_k`` scores (or all if fewer).
        2. Compute first differences: ``Δ_i = s_i - s_{i+1}``.
        3. Normalize each ``Δ_i`` by ``max(similarities)``.
        4. Find the index with the maximum normalized drop — that's the knee.
        5. Select candidates through ``knee_index`` (inclusive).
        6. If no strong knee (max normalized drop < 0.05), fall back to
           retaining all candidates above ``similarity_threshold`` (default 0.35),
           capped at ``maximum`` (spec §13).
        7. Enforce bounds: at least ``minimum``, at most ``maximum``.

    Parameters
    ----------
    similarities : list[float]
        Sorted in descending order.
    candidate_k : int
        Maximum number of candidates to consider.
    minimum : int
        Minimum number of candidates to always return.
    maximum : int
        Maximum number of candidates to ever return.
    similarity_threshold : float or None
        Fallback relevance floor. Defaults to 0.35.

    Returns
    -------
    KneeData
        Describes the detection result for this level.
    """
    if similarity_threshold is None:
        similarity_threshold = 0.35

    # Edge case: empty list
    if not similarities:
        return KneeData(
            index="",
            candidate_k=0,
            selected_k=0,
            knee_index=0,
            threshold=similarity_threshold,
        )

    # Edge case: fewer than 2 scores — no differences to compute
    if len(similarities) < 2:
        count = min(len(similarities), maximum)
        count = max(count, minimum) if len(similarities) >= minimum else count
        return KneeData(
            index="",
            candidate_k=len(similarities),
            selected_k=count,
            knee_index=len(similarities),
            threshold=similarity_threshold,
        )

    # Take top candidate_k scores
    n = min(len(similarities), candidate_k)
    scores = similarities[:n]

    # Re-check after slicing
    if len(scores) < 2:
        count = max(len(scores), minimum)
        count = min(count, maximum)
        return KneeData(
            index="",
            candidate_k=n,
            selected_k=count,
            knee_index=len(scores),
            threshold=similarity_threshold,
        )

    # Compute first differences: Δ_i = s_i - s_{i+1}
    deltas = [scores[i] - scores[i + 1] for i in range(len(scores) - 1)]

    # Normalize by max(similarities)
    max_score = max(scores)
    if max_score == 0:
        # All scores are zero — treat as identical
        count = max(minimum, 1)
        count = min(count, maximum)
        return KneeData(
            index="",
            candidate_k=n,
            selected_k=count,
            knee_index=0,
            threshold=similarity_threshold,
        )

    normalized_drops = [d / max_score for d in deltas]

    # Check if all scores are identical (all deltas are zero)
    if all(d == 0 for d in deltas):
        count = max(minimum, 1)
        count = min(count, maximum)
        return KneeData(
            index="",
            candidate_k=n,
            selected_k=count,
            knee_index=0,
            threshold=similarity_threshold,
        )

    # Find the index with the maximum normalized drop
    max_drop_idx = max(range(len(normalized_drops)), key=lambda i: normalized_drops[i])
    max_normalized_drop = normalized_drops[max_drop_idx]

    # Knee index (0-based) = max_drop_idx, meaning we select through index max_drop_idx
    # The knee occurs AFTER index max_drop_idx, so we include indices 0..max_drop_idx
    knee_index_0based = max_drop_idx

    # Check if the knee is "strong" (spec §13: threshold of 0.05)
    if max_normalized_drop >= 0.05:
        # Strong knee detected — select through knee_index (inclusive)
        selected_count = knee_index_0based + 1  # inclusive
    else:
        # No strong knee — fall back to threshold filtering (spec §13)
        selected_count = sum(1 for s in scores if s >= similarity_threshold)
        selected_count = min(selected_count, maximum)

    # Enforce bounds
    selected_count = max(selected_count, minimum)
    selected_count = min(selected_count, maximum)

    # knee_index in KneeData is 1-based (the rank at which the knee was detected)
    knee_index_1based = knee_index_0based + 1

    return KneeData(
        index="",
        candidate_k=n,
        selected_k=selected_count,
        knee_index=knee_index_1based,
        threshold=similarity_threshold,
    )
