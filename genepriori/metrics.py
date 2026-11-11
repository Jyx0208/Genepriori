"""Ranking metrics: AUPRC, Recall@K and bootstrap helpers."""

from __future__ import annotations

import warnings
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set

import numpy as np
from scipy import stats


def rank_genes(scores: Mapping[str, float]) -> List[str]:
    """Sort genes by score descending; break ties by gene symbol."""
    return sorted(scores.keys(), key=lambda g: (-float(scores[g]), g))


def recall_at_k(ranked: Sequence[str], positives: Set[str], k: int = 20) -> float:
    if not positives or k <= 0:
        return 0.0
    hit = sum(1 for g in ranked[:k] if g in positives)
    return hit / float(len(positives))


def auprc(
    ranked: Sequence[str],
    positives: Set[str],
    scores: Optional[Mapping[str, float]] = None,
) -> float:
    """Area under the precision-recall curve for a ranked list.

    Mirrors the metric definition used for the reported study results:
    precision and recall are computed at every rank, the curve starts at the
    anchor point (recall = 0, precision = 1), and integration uses the
    trapezoidal rule. This follows the trapezoidal approximation cited in the
    This differs slightly from step-wise
    ``sklearn.metrics.average_precision_score``.

    If ``scores`` is provided, the ranking is re-derived from it. Passing a
    ``ranked`` sequence that disagrees with ``scores`` emits a warning instead
    of silently ignoring it.
    """
    if scores is not None:
        derived = rank_genes(scores)
        if list(ranked) != derived:
            warnings.warn(
                "auprc(): `ranked` disagrees with the ordering implied by "
                "`scores`; the ranking derived from `scores` is used.",
                UserWarning,
                stacklevel=2,
            )
        ranked = derived
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
    """Percentile bootstrap CI for the mean; one-sided p = P(mean <= 0).

    Resamples are drawn in chunks so that large inputs do not allocate a
    ``n_bootstrap x n`` matrix at once. Given ``seed``, results are
    deterministic and independent of the chunk size.
    """
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return {
            "mean": 0.0,
            "ci_low": 0.0,
            "ci_high": 0.0,
            "p_one_sided": 1.0,
        }
    rng = np.random.default_rng(seed)
    n = arr.size
    means = np.empty(n_bootstrap, dtype=float)
    chunk = max(1, min(n_bootstrap, 2_000_000 // max(n, 1)))
    for start in range(0, n_bootstrap, chunk):
        end = min(start + chunk, n_bootstrap)
        idx = rng.integers(0, n, size=(end - start, n))
        means[start:end] = arr[idx].mean(axis=1)
    return {
        "mean": float(arr.mean()),
        "ci_low": float(np.percentile(means, 2.5)),
        "ci_high": float(np.percentile(means, 97.5)),
        "p_one_sided": float(np.mean(means <= 0.0)),
    }


def paired_bootstrap(
    candidate: Sequence[float],
    baseline: Sequence[float],
    *,
    n_bootstrap: int = 2000,
    seed: int = 42,
) -> Dict[str, float]:
    """Bootstrap CI of the per-item delta ``candidate - baseline``.

    Both sequences must be aligned (same item, e.g. the same disease) and of
    equal length. Returns the same keys as :func:`bootstrap_mean_ci` plus
    ``n_items``.
    """
    a = np.asarray(list(candidate), dtype=float)
    b = np.asarray(list(baseline), dtype=float)
    if a.size != b.size:
        raise ValueError(
            f"paired_bootstrap needs aligned sequences, got {a.size} and {b.size}."
        )
    out = bootstrap_mean_ci(a - b, n_bootstrap=n_bootstrap, seed=seed)
    out["n_items"] = float(a.size)
    return out


def sign_test(
    wins: int,
    losses: int,
    *,
    alternative: str = "two-sided",
) -> Dict[str, float]:
    """Exact binomial sign test on win/loss counts (ties are excluded).

    Args:
        wins: number of items where the candidate beat the baseline.
        losses: number of items where the baseline beat the candidate.
        alternative: ``"two-sided"``, ``"greater"`` or ``"less"``.
    """
    n = int(wins) + int(losses)
    if n == 0:
        return {"wins": float(wins), "losses": float(losses), "n": 0.0, "p_value": 1.0}
    result = stats.binomtest(int(wins), n, 0.5, alternative=alternative)
    return {
        "wins": float(wins),
        "losses": float(losses),
        "n": float(n),
        "p_value": float(result.pvalue),
    }


def benjamini_hochberg(
    pvalues: Sequence[float],
    *,
    alpha: float = 0.05,
) -> Dict[str, List[float]]:
    """Benjamini-Hochberg FDR correction, preserving input order.

    Returns ``adjusted`` p-values and ``rejected`` flags. Input order is
    maintained so callers can zip the result with labels.
    """
    p = np.asarray(list(pvalues), dtype=float)
    m = p.size
    if m == 0:
        return {"adjusted": [], "rejected": []}
    order = np.argsort(p)
    ranked = p[order]
    ranks = np.arange(1, m + 1, dtype=float)
    adjusted_ranked = ranked * m / ranks
    adjusted_ranked = np.minimum.accumulate(adjusted_ranked[::-1])[::-1]
    adjusted_ranked = np.clip(adjusted_ranked, 0.0, 1.0)
    adjusted = np.empty(m, dtype=float)
    adjusted[order] = adjusted_ranked
    return {
        "adjusted": [float(v) for v in adjusted],
        "rejected": [bool(v <= alpha) for v in adjusted],
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
