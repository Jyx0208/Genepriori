"""Tests for the ranking and evaluation pipelines."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from genepriori.config import RunConfig
from genepriori.data.cache import Cache
from genepriori.data.opentargets import DiseaseEvidence
from genepriori.data.string import load_string_edges
from genepriori.data.toy import build_toy_cache
from genepriori.pipeline.evaluation import (
    adjust_families,
    compare_evaluations,
    evaluate_evidence,
)
from genepriori.pipeline.fusion import prioritize
from genepriori.pipeline.report import write_evaluation, write_ranking


@pytest.fixture()
def toy(tmp_path: Path):
    root = build_toy_cache(tmp_path / "cache")
    cache = Cache(root)
    edges = load_string_edges(cache=cache, offline=True)
    return cache, edges


def test_prioritize_offline_returns_ranking(toy) -> None:
    cache, _ = toy
    result = prioritize("MONDO_9000001", cache=cache, offline=True, network="ppr")
    assert result.ranked
    assert result.disease_name.startswith("synthetic")
    assert [row.rank for row in result.ranked] == list(range(1, len(result.ranked) + 1))


def test_ranking_is_sorted_by_score(toy) -> None:
    cache, _ = toy
    result = prioritize("MONDO_9000001", cache=cache, offline=True, network="ppr")
    scores = [row.score for row in result.ranked]
    assert scores == sorted(scores, reverse=True)


def test_network_discovers_genes_without_direct_evidence(toy) -> None:
    cache, _ = toy
    result = prioritize("MONDO_9000001", cache=cache, offline=True, network="ppr")
    discovered = [row for row in result.ranked if row.network_discovered]
    assert discovered, "expected network-discovered genes"
    for row in discovered:
        assert row.score > 0.0


def test_no_network_keeps_only_direct_genes(toy) -> None:
    cache, _ = toy
    result = prioritize(
        "MONDO_9000001", cache=cache, offline=True, network="none"
    )
    assert result.ranked
    assert not any(row.network_discovered for row in result.ranked)


def test_ranking_is_deterministic(toy) -> None:
    cache, _ = toy
    first = prioritize("MONDO_9000001", cache=cache, offline=True, network="rwr")
    second = prioritize("MONDO_9000001", cache=cache, offline=True, network="rwr")
    assert first.to_dict()["ranked"] == second.to_dict()["ranked"]


def test_result_csv_and_json_roundtrip(toy, tmp_path: Path) -> None:
    cache, _ = toy
    result = prioritize("MONDO_9000001", cache=cache, offline=True, network="none")
    csv_path = result.to_csv(tmp_path / "ranked.csv")
    json_path = result.to_json(tmp_path / "result.json")
    text = csv_path.read_text(encoding="utf-8").splitlines()
    assert text[0].startswith("rank,gene,score")
    assert len(text) == len(result.ranked) + 1
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["disease_id"] == "MONDO_9000001"


def test_build_ranking_weights_are_recorded(toy) -> None:
    cache, _ = toy
    result = prioritize(
        "MONDO_9000001", cache=cache, offline=True, network="ppr", weights="genetics"
    )
    assert result.config["weights"]["genetics"] == pytest.approx(0.5)
    assert result.config["network"]["algorithm"] == "ppr"


def test_evaluate_mask_vs_restore(toy) -> None:
    cache, edges = toy
    evidence = prioritization_evidence(cache)
    mask = evaluate_evidence(evidence, config=RunConfig(), edges=edges)
    restore_cfg = RunConfig()
    restore_cfg.evaluation.protocol = "restore"
    restore = evaluate_evidence(evidence, config=restore_cfg, edges=edges)

    assert mask.protocol == "mask"
    assert restore.protocol == "restore"
    assert all(f.n_excluded == 0 for f in mask.fold_results)
    assert all(f.n_excluded > 0 for f in restore.fold_results)
    # mask evaluates one fifth of the positives per fold, restore four fifths.
    assert mask.fold_results[0].n_eval_positives < restore.fold_results[0].n_eval_positives
    assert 0.0 <= mask.metric("auprc") <= 1.0
    assert 0.0 <= restore.metric("auprc") <= 1.0


def prioritization_evidence(cache: Cache) -> DiseaseEvidence:
    from genepriori.data.opentargets import fetch_disease_evidence

    return fetch_disease_evidence("MONDO_9000001", cache=cache, offline=True)


def test_evaluate_summary_has_ci(toy) -> None:
    cache, edges = toy
    result = evaluate_evidence(prioritization_evidence(cache), config=RunConfig(), edges=edges)
    summary = result.summary["auprc"]
    assert {"mean", "sd", "ci_low", "ci_high", "n"} <= set(summary)
    assert summary["ci_low"] <= summary["mean"] <= summary["ci_high"]
    assert summary["n"] == 5.0


def test_evaluate_raises_when_too_few_positives(toy) -> None:
    cache, edges = toy
    evidence = DiseaseEvidence(
        disease_id="MONDO_X",
        disease_name="tiny",
        gene_type_scores={
            "A": {"drug": 0.9, "genetics": 0.5},
            "B": {"drug": 0.8, "genetics": 0.4},
            "C": {"drug": 0.1, "genetics": 0.3},
        },
    )
    with pytest.raises(ValueError, match="at least 5"):
        evaluate_evidence(evidence, config=RunConfig(), edges=edges)


def test_compare_evaluations_reports_delta(toy) -> None:
    cache, edges = toy
    evidence = prioritization_evidence(cache)
    baseline = evaluate_evidence(
        evidence, config=RunConfig().with_holdout("drug"), edges=edges
    )
    candidate = evaluate_evidence(
        evidence,
        config=RunConfig().with_holdout("drug").with_weights("genetics"),
        edges=edges,
    )
    comparison = compare_evaluations([baseline], [candidate], n_bootstrap=100)
    assert comparison["n_diseases"] == 1
    assert comparison["delta_mean"] == pytest.approx(
        candidate.metric("auprc") - baseline.metric("auprc")
    )
    assert comparison["wins"] + comparison["losses"] + comparison["ties"] == 1


def test_compare_evaluations_requires_overlap() -> None:
    with pytest.raises(ValueError, match="No shared diseases"):
        compare_evaluations([], [], metric="auprc")


def test_adjust_families_adds_bh_fields() -> None:
    family = {
        "a": {"sign_test_p": 0.01},
        "b": {"sign_test_p": 0.04},
    }
    adjusted = adjust_families(family, alpha=0.05)
    assert adjusted["a"]["sign_test_p_adj"] == pytest.approx(0.02)
    assert adjusted["a"]["rejected_fdr"] is True
    assert "rejected_fdr" in adjusted["b"]


def test_write_ranking_and_evaluation_files(toy, tmp_path: Path) -> None:
    cache, edges = toy
    result = prioritize("MONDO_9000001", cache=cache, offline=True, network="none")
    files = write_ranking(result, tmp_path / "rank")
    assert {p.name for p in files} == {"ranked_genes.csv", "result.json"}
    assert all(p.exists() for p in files)

    evaluation = evaluate_evidence(prioritization_evidence(cache), config=RunConfig(), edges=edges)
    files = write_evaluation(evaluation, tmp_path / "eval", prefix="d1_drug")
    assert {p.name for p in files} == {"d1_drug_metrics.json", "d1_drug_folds.csv"}
    assert all(p.exists() for p in files)


def test_csv_contains_summary_rows(toy, tmp_path: Path) -> None:
    cache, edges = toy
    evaluation = evaluate_evidence(prioritization_evidence(cache), config=RunConfig(), edges=edges)
    path = evaluation.to_csv(tmp_path / "folds.csv")
    text = path.read_text(encoding="utf-8")
    for label in ("mean", "sd", "ci_low", "ci_high"):
        assert label in text
