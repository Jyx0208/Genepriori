"""Tests for adjacency building, propagation and score blending."""

from __future__ import annotations

import math

import numpy as np
import pytest

from genepriori.network import (
    blend_scores,
    build_adjacency,
    degree_penalty,
    ensemble_propagation,
    extract_1hop_subgraph,
    heat_diffusion,
    neighbor_voting,
    personalized_pagerank,
    propagate,
    random_walk_with_restart,
)

TRIANGLE = [("A", "B", 1.0), ("B", "C", 1.0), ("A", "C", 1.0)]


def test_build_adjacency_is_symmetric() -> None:
    names, adj = build_adjacency(TRIANGLE)
    assert names == ["A", "B", "C"]
    dense = adj.toarray()
    assert np.allclose(dense, dense.T)


def test_build_adjacency_keeps_max_duplicate() -> None:
    names, adj = build_adjacency([("A", "B", 0.5), ("B", "A", 0.9)])
    assert names == ["A", "B"]
    assert math.isclose(adj[0, 1], 0.9)
    assert math.isclose(adj[1, 0], 0.9)


def test_build_adjacency_filters_low_scores_and_self_loops() -> None:
    names, adj = build_adjacency([("A", "B", 0.39), ("C", "D", 0.4), ("E", "E", 0.9)])
    assert set(names) == {"C", "D"}
    assert adj.shape == (2, 2)


def test_build_adjacency_empty() -> None:
    names, adj = build_adjacency([])
    assert names == []
    assert adj.shape == (0, 0)


def test_extract_1hop_keeps_isolated_seed() -> None:
    names, adj, degrees = extract_1hop_subgraph(["Z"], TRIANGLE)
    assert "Z" in names
    assert degrees["Z"] == 0.0
    assert adj.shape == (1, 1)


def test_extract_1hop_limits_neighbors() -> None:
    edges = [("S", f"N{i}", 1.0 - i / 100.0) for i in range(5)]
    names, _, _ = extract_1hop_subgraph(["S"], edges, max_neighbors_per_seed=2)
    assert set(names) == {"S", "N0", "N1"}


def test_extract_1hop_breaks_weight_ties_by_symbol() -> None:
    edges = [("S", "Z", 0.8), ("S", "A", 0.8)]
    names, _, _ = extract_1hop_subgraph(["S"], edges, max_neighbors_per_seed=1)
    assert set(names) == {"S", "A"}


def test_ppr_conserves_probability_mass() -> None:
    _, adj = build_adjacency(TRIANGLE)
    vec = personalized_pagerank(adj, {0: 1.0})
    assert math.isclose(float(vec.sum()), 1.0, rel_tol=1e-6)
    assert vec[0] > vec[1]


def test_rwr_conserves_probability_mass() -> None:
    _, adj = build_adjacency(TRIANGLE)
    vec = random_walk_with_restart(adj, {0: 1.0})
    assert math.isclose(float(vec.sum()), 1.0, rel_tol=1e-6)


def test_ppr_and_rwr_agree_on_seed_dominance() -> None:
    _, adj = build_adjacency(TRIANGLE)
    for vec in (
        personalized_pagerank(adj, {0: 1.0}),
        random_walk_with_restart(adj, {0: 1.0}),
    ):
        assert int(np.argmax(vec)) == 0


def test_neighbor_voting_ignores_self() -> None:
    _, adj = build_adjacency(TRIANGLE)
    vec = neighbor_voting(adj, {0: 1.0})
    assert vec[0] == 0.0
    assert math.isclose(vec[1], 0.5)
    assert math.isclose(vec[2], 0.5)


def test_heat_diffusion_is_nonnegative_and_normalized() -> None:
    _, adj = build_adjacency(TRIANGLE)
    vec = heat_diffusion(adj, {0: 1.0}, steps=5)
    assert np.all(vec >= 0.0)
    assert math.isclose(float(vec.sum()), 1.0, rel_tol=1e-6)


def test_ensemble_matches_shape_of_inputs() -> None:
    _, adj = build_adjacency(TRIANGLE)
    vec = ensemble_propagation(adj, {0: 1.0})
    assert vec.shape == (3,)
    assert math.isclose(float(vec.sum()), 1.0, rel_tol=1e-6)


def test_degree_penalty_decreases_with_degree() -> None:
    assert math.isclose(degree_penalty(0.0), 1.0)
    assert degree_penalty(10.0) < degree_penalty(1.0)


def test_blend_scores_formulas_unnormalized() -> None:
    out = blend_scores(
        {"A": 1.0},
        {"A": 0.5, "B": 0.2},
        {"A": 2.0, "B": 1.0},
        blend=0.7,
        network_discount=0.25,
        max_network_genes=80,
        normalize_network=False,
    )
    assert math.isclose(out["A"], 0.7 * 1.0 + 0.3 * 0.5)
    expected_b = 0.25 * 0.2 / (1.0 + math.log(2.0))
    assert math.isclose(out["B"], expected_b)


def test_blend_scores_normalizes_network_by_default() -> None:
    out = blend_scores(
        {"A": 1.0, "B": 0.5},
        {"A": 0.5, "B": 0.2},
        {"A": 1.0, "B": 1.0},
    )
    # network min-max: A -> 1.0, B -> 0.0
    assert math.isclose(out["A"], 0.7 * 1.0 + 0.3 * 1.0)
    assert math.isclose(out["B"], 0.7 * 0.5 + 0.0)


def test_blend_scores_top_n_tie_is_deterministic() -> None:
    out = blend_scores(
        {"S": 1.0},
        {"B": 0.2, "A": 0.2},
        {"A": 1.0, "B": 1.0},
        max_network_genes=1,
        normalize_network=False,
    )
    assert "A" in out
    assert "B" not in out


def test_blend_scores_drops_zero_network_genes() -> None:
    out = blend_scores({"S": 1.0}, {"Z": 0.0}, {"Z": 1.0})
    assert "Z" not in out


def test_propagate_none_returns_direct_scores() -> None:
    direct = {"A": 0.9, "B": 0.1}
    out = propagate(direct, TRIANGLE, algorithm="none")
    assert out == direct
    assert out is not direct


def test_propagate_empty_seeds_returns_input() -> None:
    direct = {"A": 0.0, "B": 0.0}
    assert propagate(direct, TRIANGLE, algorithm="ppr") == direct


def test_propagate_without_edges_returns_input() -> None:
    direct = {"A": 0.9}
    assert propagate(direct, [], algorithm="ppr") == direct


def test_propagate_rejects_unknown_algorithm() -> None:
    with pytest.raises(ValueError, match="Unsupported algorithm"):
        propagate({"A": 1.0}, TRIANGLE, algorithm="magic")


def test_propagate_is_deterministic() -> None:
    direct = {"A": 0.9, "C": 0.4}
    first = propagate(direct, TRIANGLE, algorithm="ppr")
    second = propagate(direct, TRIANGLE, algorithm="ppr")
    assert first == second


def test_propagate_all_supported_algorithms_run() -> None:
    direct = {"A": 0.9, "C": 0.4}
    for algo in ("ppr", "pagerank", "rwr", "neighbor_voting", "hd", "ensemble"):
        out = propagate(direct, TRIANGLE, algorithm=algo)
        assert set(out) >= {"A", "C"}
        assert all(math.isfinite(v) for v in out.values())
