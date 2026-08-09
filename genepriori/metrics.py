"""Ranking metrics used in the manuscript (AUPRC, Recall@K, bootstrap)."""

from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np


def rank_genes(scores: Mapping[str, float]) -> List[str]:
    """Sort genes by score descending; break ties by gene symbol."""
    return sorted(scores.keys(), key=lambda g: (-float(scores[g]), g))


def recall_at_k(ranked: Sequence[str], positives: Set[str], k: int = 20) -> float:
    if not positives or k <= 0:
        return 0.0
    hit = sum(1 for g in ranked[:k] if g in positives)
    return hit / float(len(positives))


def auprc(ranked: Sequence[str], positives: Set[str], scores: Optional[Mapping[str, float]] = None) -> float:
    """Area under the precision-recall curve for a ranked list.

    Mirrors the metric definition used for the reported study results:
    precision and recall are computed at every rank, the curve starts at the
    anchor point (recall = 0, precision = 1), and integration uses the
    trapezoidal rule. This follows the trapezoidal approximation cited in the
    manuscript and differs slightly from step-wise
    ``sklearn.metrics.average_precision_score``.

    If ``scores`` is provided, ranking is re-derived for safety.
    """
    if scores is not None:
        ranked = rank_genes(scores)
    pos = set(positives)
    n_pos = len(pos)
    if not ranked or n_pos == 0:
        return 0.0

    tp = 0
    precisions: List[float] = [1.0]  # anchor point (recall=0, precision=1)
    recalls: List[float] = [0.0]
    for i, g in enumerate(ranked):
        if g in pos:
            tp += 1
        precisions.append(tp / float(i + 1))
        recalls.append(tp / float(n_pos))

    # Trapezoidal integration of precision w.r.t. recall.
    a = 0.0
    for i in range(1, len(recalls)):
        a += (recalls[i] - recalls[i - 1]) * (precisions[i] + precisions[i - 1]) / 2.0
    return float(a)


def bootstrap_mean_ci(
    values: Sequence[float],
    *,
    n_bootstrap: int = 2000,
    seed: int = 42,
) -> Dict[str, float]:
    """Percentile bootstrap CI for the mean; one-sided p = P(mean <= 0)."""
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return {
            "mean": 0.0,
            "ci_low": 0.0,
            "ci_high": 0.0,
            "p_one_sided": 1.0,
        }
    rng = np.random.default_rng(seed)
    means = np.empty(n_bootstrap, dtype=float)
    n = arr.size
    for i in range(n_bootstrap):
        sample = arr[rng.integers(0, n, size=n)]
        means[i] = sample.mean()
    return {
        "mean": float(arr.mean()),
        "ci_low": float(np.percentile(means, 2.5)),
        "ci_high": float(np.percentile(means, 97.5)),
        "p_one_sided": float(np.mean(means <= 0.0)),
    }


def evaluate_ranking(
    scores: Mapping[str, float],
    positives: Iterable[str],
    *,
    k: int = 20,
) -> Dict[str, float]:
    pos = set(positives)
    ranked = rank_genes(scores)
    return {
        "auprc": auprc(ranked, pos),
        f"recall_at_{k}": recall_at_k(ranked, pos, k=k),
        "n_positives": float(len(pos)),
        "n_ranked": float(len(ranked)),
    }
