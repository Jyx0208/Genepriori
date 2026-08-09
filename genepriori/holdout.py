"""Dual-holdout positive-set construction and fold splitting (Methods)."""

from __future__ import annotations

import random
from typing import Dict, Iterable, List, Mapping, Sequence, Set, Tuple

from .evidence import normalize_holdout_type


def build_positives(
    gene_type_scores: Mapping[str, Mapping[str, float]],
    holdout_type: str,
    *,
    threshold: float = 0.30,
) -> Set[str]:
    """Positives: genes with holdout-type score > threshold (manuscript: 0.30).

    Holdout types: drug (known_drug) or genetics (genetic_association).
    """
    h = normalize_holdout_type(holdout_type)
    pos: Set[str] = set()
    for gene, tmap in gene_type_scores.items():
        if float(tmap.get(h, 0.0) or 0.0) > threshold:
            pos.add(gene)
    return pos


def kfold_splits(
    positives: Iterable[str],
    *,
    n_folds: int = 5,
    seed: int = 42,
) -> List[Set[str]]:
    """Alphabetical sort, deterministic shuffle (seed=42), then split into K folds.

    Each returned set is the *held-out test positives* for one fold
    (removed from the gold standard for that fold, per Methods).
    """
    items = sorted(set(positives))
    rng = random.Random(seed)
    rng.shuffle(items)
    if len(items) < n_folds:
        raise ValueError(
            f"Need at least {n_folds} positives for {n_folds}-fold CV, got {len(items)}."
        )
    folds: List[List[str]] = [[] for _ in range(n_folds)]
    for i, g in enumerate(items):
        folds[i % n_folds].append(g)
    return [set(f) for f in folds]


def train_positives(all_positives: Set[str], test_fold: Set[str]) -> Set[str]:
    return set(all_positives) - set(test_fold)
