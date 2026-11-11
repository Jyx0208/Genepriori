"""Tests for dual-holdout positives and fold splitting."""

from __future__ import annotations

import pytest

from genepriori.holdout import build_positives, kfold_splits, train_positives


def test_build_positives_uses_strict_threshold() -> None:
    scores = {
        "A": {"drug": 0.30},
        "B": {"drug": 0.31},
        "C": {"drug": 0.9},
        "D": {"drug": 0.0},
    }
    assert build_positives(scores, "drug") == {"B", "C"}


def test_build_positives_accepts_alias() -> None:
    scores = {"A": {"genetics": 0.5}}
    assert build_positives(scores, "genetic_association") == {"A"}
    assert build_positives(scores, "genetics") == {"A"}


def test_build_positives_custom_threshold() -> None:
    scores = {"A": {"drug": 0.5}, "B": {"drug": 0.6}}
    assert build_positives(scores, "drug", threshold=0.55) == {"B"}


def test_build_positives_rejects_bad_holdout() -> None:
    with pytest.raises(ValueError):
        build_positives({}, "expression")


def test_kfold_splits_partition_all_positives() -> None:
    positives = {f"G{i}" for i in range(23)}
    folds = kfold_splits(positives, n_folds=5, seed=42)
    assert len(folds) == 5
    assert set().union(*folds) == positives
    assert sum(len(f) for f in folds) == len(positives)


def test_kfold_splits_are_deterministic() -> None:
    positives = {f"G{i}" for i in range(20)}
    assert kfold_splits(positives, n_folds=5, seed=42) == kfold_splits(
        positives, n_folds=5, seed=42
    )


def test_kfold_splits_change_with_seed() -> None:
    positives = {f"G{i}" for i in range(20)}
    assert kfold_splits(positives, n_folds=5, seed=42) != kfold_splits(
        positives, n_folds=5, seed=1
    )


def test_kfold_splits_fold_sizes_differ_by_at_most_one() -> None:
    positives = {f"G{i}" for i in range(23)}
    sizes = [len(f) for f in kfold_splits(positives, n_folds=5)]
    assert max(sizes) - min(sizes) <= 1


def test_kfold_requires_enough_positives() -> None:
    with pytest.raises(ValueError, match="at least 5"):
        kfold_splits({"A", "B"}, n_folds=5)


def test_kfold_requires_at_least_two_folds() -> None:
    with pytest.raises(ValueError, match="n_folds must be >= 2"):
        kfold_splits({"A", "B", "C"}, n_folds=1)


def test_train_positives_excludes_test_fold() -> None:
    all_pos = {"A", "B", "C", "D"}
    assert train_positives(all_pos, {"C"}) == {"A", "B", "D"}
