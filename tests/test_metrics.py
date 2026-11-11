"""Tests for ranking metrics and statistical helpers."""

from __future__ import annotations

import math

import pytest

from genepriori.metrics import (
    auprc,
    benjamini_hochberg,
    bootstrap_mean_ci,
    evaluate_ranking,
    paired_bootstrap,
    rank_genes,
    recall_at_k,
    sign_test,
)


def test_rank_genes_descending() -> None:
    assert rank_genes({"A": 0.1, "B": 0.9, "C": 0.5}) == ["B", "C", "A"]


def test_rank_genes_breaks_ties_by_symbol() -> None:
    assert rank_genes({"C": 0.5, "A": 0.5, "B": 0.5}) == ["A", "B", "C"]


def test_recall_at_k_basic() -> None:
    assert math.isclose(recall_at_k(["A", "B", "C"], {"A", "C"}, k=2), 0.5)
    assert math.isclose(recall_at_k(["A", "B", "C"], {"A", "C"}, k=3), 1.0)


def test_recall_at_k_edge_cases() -> None:
    assert recall_at_k(["A"], set(), k=1) == 0.0
    assert recall_at_k(["A"], {"A"}, k=0) == 0.0


def test_auprc_hand_computed() -> None:
    # anchor (0,1) -> G1 at r=0.5,p=1 -> G3 at r=1,p=2/3
    expected = 0.5 * 1.0 + 0.5 * (0.5 + 2 / 3) / 2
    assert math.isclose(auprc(["G1", "G2", "G3"], {"G1", "G3"}), expected)


def test_auprc_perfect_ranking_is_one() -> None:
    assert math.isclose(auprc(["A", "B", "C"], {"A", "B"}), 1.0)


def test_auprc_empty_inputs() -> None:
    assert auprc([], {"A"}) == 0.0
    assert auprc(["A"], set()) == 0.0


def test_auprc_recomputes_from_scores() -> None:
    ranked = ["A", "B"]
    scores = {"A": 0.9, "B": 0.1}
    assert math.isclose(auprc(ranked, {"A"}, scores=scores), 1.0)


def test_auprc_warns_when_ranked_disagrees_with_scores() -> None:
    with pytest.warns(UserWarning, match="disagrees"):
        auprc(["B", "A"], {"A"}, scores={"A": 0.9, "B": 0.1})


def test_bootstrap_mean_ci_reports_mean_and_bounds() -> None:
    out = bootstrap_mean_ci([0.1, 0.2, 0.3, 0.4, 0.5], n_bootstrap=500, seed=7)
    assert math.isclose(out["mean"], 0.3)
    assert out["ci_low"] <= out["mean"] <= out["ci_high"]


def test_bootstrap_is_deterministic() -> None:
    values = [0.01, 0.02, 0.03, 0.05]
    assert bootstrap_mean_ci(values, n_bootstrap=300, seed=11) == bootstrap_mean_ci(
        values, n_bootstrap=300, seed=11
    )


def test_bootstrap_empty_values() -> None:
    out = bootstrap_mean_ci([])
    assert out == {"mean": 0.0, "ci_low": 0.0, "ci_high": 0.0, "p_one_sided": 1.0}


def test_paired_bootstrap_requires_alignment() -> None:
    with pytest.raises(ValueError, match="aligned"):
        paired_bootstrap([0.1, 0.2], [0.1])


def test_paired_bootstrap_reports_delta() -> None:
    out = paired_bootstrap([0.5, 0.4, 0.6], [0.4, 0.4, 0.3], n_bootstrap=200, seed=3)
    assert math.isclose(out["mean"], (0.1 + 0.0 + 0.3) / 3)
    assert out["n_items"] == 3.0


def test_sign_test_exact_binomial() -> None:
    # n=10, wins=9: two-sided p = 2 * (C(10,0) + C(10,1)) / 2^10
    out = sign_test(9, 1)
    assert math.isclose(out["p_value"], 2 * 11 / 1024)


def test_sign_test_no_trials() -> None:
    assert sign_test(0, 0)["p_value"] == 1.0


def test_sign_test_one_sided() -> None:
    two = sign_test(8, 2)["p_value"]
    greater = sign_test(8, 2, alternative="greater")["p_value"]
    assert greater <= two


def test_benjamini_hochberg_monotone_and_ordered() -> None:
    out = benjamini_hochberg([0.001, 0.008, 0.039, 0.041, 0.042])
    adjusted = out["adjusted"]
    assert math.isclose(adjusted[0], 0.005)
    assert math.isclose(adjusted[1], 0.02)
    assert all(math.isclose(a, 0.042) for a in adjusted[2:])


def test_benjamini_hochberg_preserves_input_order() -> None:
    out = benjamini_hochberg([0.5, 0.001])
    assert out["adjusted"][1] < out["adjusted"][0]


def test_benjamini_hochberg_empty() -> None:
    assert benjamini_hochberg([]) == {"adjusted": [], "rejected": []}


def test_evaluate_ranking_keys() -> None:
    out = evaluate_ranking({"A": 1.0, "B": 0.5, "C": 0.0}, {"A", "C"}, k=2)
    assert set(out) == {"auprc", "recall_at_2", "n_positives", "n_ranked"}
    assert out["n_positives"] == 2.0
    assert out["n_ranked"] == 3.0
