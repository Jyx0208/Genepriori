"""Tests for evidence mapping, weight handling and score fusion."""

from __future__ import annotations

import math

import pytest

from genepriori.evidence import (
    OT_TO_STANDARD,
    STANDARD_TYPES,
    FusedScores,
    aggregate_ot_records,
    confidence_from_n_types,
    fuse_gene_scores,
    fuse_gene_scores_detailed,
    map_ot_type,
    min_max_normalize,
    normalize_holdout_type,
    renorm_weights,
)


def test_map_ot_type_covers_table_3() -> None:
    assert map_ot_type("known_drug") == "drug"
    assert map_ot_type("clinical") == "drug"
    assert map_ot_type("genetic_association") == "genetics"
    assert map_ot_type("rna_expression") == "expression"
    assert map_ot_type("affected_pathway") == "pathway"
    assert map_ot_type("literature") == "literature"
    assert map_ot_type("animal_model") == "animal_model"


def test_map_ot_type_is_case_and_space_insensitive() -> None:
    assert map_ot_type("  KNOWN_DRUG ") == "drug"


def test_map_ot_type_unknown_returns_none() -> None:
    assert map_ot_type("not_a_type") is None
    assert map_ot_type("") is None


def test_every_mapped_type_is_standard() -> None:
    assert set(OT_TO_STANDARD.values()) <= set(STANDARD_TYPES)


def test_normalize_holdout_aliases() -> None:
    for alias in ("drug", "drug_target", "known_drug", "DRUG"):
        assert normalize_holdout_type(alias) == "drug"
    for alias in ("genetics", "genetic", "genetic_association"):
        assert normalize_holdout_type(alias) == "genetics"


def test_normalize_holdout_rejects_unknown() -> None:
    with pytest.raises(ValueError, match="Unsupported holdout_type"):
        normalize_holdout_type("expression")


def test_min_max_normalize_basic() -> None:
    out = min_max_normalize({"a": 1.0, "b": 3.0, "c": 2.0})
    assert out == {"a": 0.0, "b": 1.0, "c": 0.5}


def test_min_max_normalize_constant_is_zero() -> None:
    assert min_max_normalize({"a": 2.0, "b": 2.0}) == {"a": 0.0, "b": 0.0}


def test_min_max_normalize_empty() -> None:
    assert min_max_normalize({}) == {}


def test_renorm_weights_sums_to_one() -> None:
    w = renorm_weights({"drug": 2.0, "genetics": 1.0, "pathway": 1.0})
    assert math.isclose(sum(w.values()), 1.0)
    assert math.isclose(w["drug"], 0.5)


def test_renorm_weights_clips_negatives() -> None:
    w = renorm_weights({"drug": -1.0, "genetics": 2.0, "pathway": 2.0})
    assert w["drug"] == 0.0
    assert math.isclose(sum(w.values()), 1.0)


def test_renorm_weights_masks_holdout() -> None:
    w = renorm_weights({"drug": 0.9, "genetics": 0.1}, holdout_type="drug")
    assert w["drug"] == 0.0
    assert math.isclose(w["genetics"], 1.0)


def test_renorm_weights_falls_back_to_uniform() -> None:
    w = renorm_weights({"drug": 1.0}, holdout_type="drug")
    assert w["drug"] == 0.0
    assert math.isclose(w["genetics"], 0.2)
    assert math.isclose(sum(w.values()), 1.0)


def test_renorm_weights_ignores_unknown_types() -> None:
    w = renorm_weights({"mystery": 5.0, "genetics": 1.0})
    assert set(w) == set(STANDARD_TYPES)
    assert math.isclose(w["genetics"], 1.0)


def test_confidence_formula() -> None:
    assert math.isclose(confidence_from_n_types(0), 0.35)
    assert math.isclose(confidence_from_n_types(1), 0.47)
    assert confidence_from_n_types(10) == 1.0


def test_fuse_gene_scores_matches_manual_computation() -> None:
    scores = {
        "G1": {"drug": 1.0, "genetics": 0.0},
        "G2": {"drug": 0.0, "genetics": 1.0},
    }
    out = fuse_gene_scores(scores, {"drug": 0.5, "genetics": 0.5})
    assert math.isclose(out["G1"], 0.5)
    assert math.isclose(out["G2"], 0.5)


def test_fuse_gene_scores_keeps_all_genes() -> None:
    scores = {"G1": {"genetics": 1.0}, "G2": {}, "G3": {"literature": 1.0}}
    out = fuse_gene_scores(scores, {"genetics": 0.5, "literature": 0.5})
    assert set(out) == {"G1", "G2", "G3"}
    assert out["G2"] == 0.0


def test_fuse_gene_scores_holdout_zeroes_contribution() -> None:
    scores = {"G1": {"drug": 1.0, "genetics": 1.0}, "G2": {"genetics": 0.0}}
    out = fuse_gene_scores(scores, {"drug": 0.5, "genetics": 0.5}, holdout_type="drug")
    # drug masked => G1 only carries genetics (renormalized to weight 1.0)
    assert math.isclose(out["G1"], 1.0)
    assert out["G2"] == 0.0


def test_fuse_gene_scores_ignores_unknown_evidence_types() -> None:
    scores = {"G1": {"mystery": 9.0, "genetics": 1.0}, "G2": {"genetics": 0.0}}
    out = fuse_gene_scores(scores, {"genetics": 1.0})
    assert math.isclose(out["G1"], 1.0)


def test_fuse_gene_scores_detailed_contributions_sum_to_score() -> None:
    scores = {
        "G1": {"drug": 0.9, "genetics": 0.2},
        "G2": {"drug": 0.7, "genetics": 0.1},
        "G3": {"genetics": 0.9},
    }
    detail = fuse_gene_scores_detailed(scores, {"drug": 0.5, "genetics": 0.5})
    assert isinstance(detail, FusedScores)
    for gene, score in detail.scores.items():
        assert math.isclose(sum(detail.contributions[gene].values()), score, abs_tol=1e-12)


def test_fuse_gene_scores_detailed_confidence_matches_n_types() -> None:
    scores = {
        "G1": {"drug": 1.0, "genetics": 1.0, "pathway": 1.0},
        "G2": {"drug": 0.5, "genetics": 0.5, "pathway": 0.5},
        "G3": {"drug": 0.0, "genetics": 0.0, "pathway": 0.0},
    }
    detail = fuse_gene_scores_detailed(scores, {"drug": 1 / 3, "genetics": 1 / 3, "pathway": 1 / 3})
    assert detail.n_types["G1"] == 3
    assert detail.n_types["G3"] == 0
    assert detail.confidence["G1"] == confidence_from_n_types(3)


def test_fuse_gene_scores_is_deterministic() -> None:
    scores = {"G2": {"genetics": 0.3}, "G1": {"genetics": 0.7}, "G3": {"genetics": 0.5}}
    first = fuse_gene_scores(scores, {"genetics": 1.0})
    second = fuse_gene_scores(scores, {"genetics": 1.0})
    assert first == second
    assert list(first) == ["G1", "G2", "G3"]


def test_aggregate_ot_records_takes_max_per_type() -> None:
    records = [
        {"gene": "G1", "evidence_type": "known_drug", "score": 0.4},
        {"gene": "G1", "evidence_type": "drugs", "score": 0.8},
        {"gene": "G1", "evidence_type": "literature", "score": 0.2},
    ]
    out = aggregate_ot_records(records)
    assert out["G1"]["drug"] == 0.8
    assert out["G1"]["literature"] == 0.2


def test_aggregate_ot_records_accepts_standard_names() -> None:
    out = aggregate_ot_records([{"gene": "G1", "evidence_type": "genetics", "score": 0.5}])
    assert out["G1"]["genetics"] == 0.5


def test_aggregate_ot_records_skips_unknown() -> None:
    out = aggregate_ot_records([{"gene": "G1", "evidence_type": "mystery", "score": 0.5}])
    assert out == {}
