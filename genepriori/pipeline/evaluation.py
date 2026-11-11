"""Dual-holdout cross-validation with explicit, documented protocols.

Two protocols are supported because the manuscript's wording ("its
holdout-type evidence was restored to scoring") admits two readings. Both are
implemented explicitly rather than guessed at:

``mask`` (strict)
    The holdout evidence type is masked for *every* gene. One ranking is
    produced; the folds partition the positives and each fold's metrics are
    computed against all non-positives. Reported mean/SD therefore measures
    how stable recovery is across positive subsets.

``restore`` (manuscript-literal)
    For fold *k*, the fold's positives are removed from the gold standard and
    their holdout-type evidence is restored, so they are neither scored as
    positives nor treated as negatives. The remaining positives are evaluated
    against all non-positives. Ranking is recomputed per fold.

In both protocols the evaluated positives never contribute their holdout
evidence, so the measured signal comes from complementary evidence types and
(optionally) network propagation.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Union

import numpy as np
from scipy import stats

from ..config import RunConfig
from ..data.cache import Cache
from ..data.opentargets import DiseaseEvidence
from ..data.registry import Disease
from ..evidence import fuse_gene_scores
from ..holdout import build_positives, kfold_splits, train_positives
from ..metrics import (
    auprc,
    benjamini_hochberg,
    paired_bootstrap,
    rank_genes,
    recall_at_k,
    sign_test,
)
from ..network import Edge, propagate
from .fusion import load_inputs

PathLike = Union[str, Path]


@dataclass
class FoldResult:
    """Metrics for one cross-validation fold."""

    fold: int
    n_eval_positives: int
    n_train_positives: int
    n_excluded: int
    metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class EvaluationResult:
    """Fold-level metrics plus a mean/SD/CI summary."""

    disease_id: str
    disease_name: str
    holdout: str
    protocol: str
    folds: int
    seed: int
    k: int
    fold_results: List[FoldResult] = field(default_factory=list)
    summary: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def metric(self, name: str = "auprc") -> float:
        return float(self.summary.get(name, {}).get("mean", 0.0))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "disease_id": self.disease_id,
            "disease_name": self.disease_name,
            "holdout": self.holdout,
            "protocol": self.protocol,
            "folds": self.folds,
            "seed": self.seed,
            "k": self.k,
            "summary": self.summary,
            "fold_results": [
                {
                    "fold": f.fold,
                    "n_eval_positives": f.n_eval_positives,
                    "n_train_positives": f.n_train_positives,
                    "n_excluded": f.n_excluded,
                    "metrics": f.metrics,
                }
                for f in self.fold_results
            ],
        }

    def to_json(self, path: PathLike) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return target

    def to_csv(self, path: PathLike) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        metric_names = sorted(self.summary)
        with open(target, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            header = ["fold", "n_eval_positives", "n_train_positives", "n_excluded", *metric_names]
            writer.writerow(header)
            for fold in self.fold_results:
                writer.writerow(
                    [
                        fold.fold,
                        fold.n_eval_positives,
                        fold.n_train_positives,
                        fold.n_excluded,
                        *[round(fold.metrics.get(m, 0.0), 8) for m in metric_names],
                    ]
                )
            for label, key in (("mean", "mean"), ("sd", "sd"), ("ci_low", "ci_low"), ("ci_high", "ci_high")):
                writer.writerow(
                    [label, "", "", "", *[round(self.summary[m].get(key, 0.0), 8) for m in metric_names]]
                )
        return target


def _summarize(values: Sequence[float]) -> Dict[str, float]:
    arr = np.asarray(list(values), dtype=float)
    n = int(arr.size)
    if n == 0:
        return {"mean": 0.0, "sd": 0.0, "ci_low": 0.0, "ci_high": 0.0, "n": 0.0}
    mean = float(arr.mean())
    if n == 1:
        return {"mean": mean, "sd": 0.0, "ci_low": mean, "ci_high": mean, "n": 1.0}
    sd = float(arr.std(ddof=1))
    if sd == 0.0:
        return {"mean": mean, "sd": 0.0, "ci_low": mean, "ci_high": mean, "n": float(n)}
    sem = sd / math.sqrt(n)
    low, high = stats.t.interval(0.95, n - 1, loc=mean, scale=sem)
    return {
        "mean": mean,
        "sd": sd,
        "ci_low": float(low),
        "ci_high": float(high),
        "n": float(n),
    }


def _propagate_scores(
    direct: Mapping[str, float],
    edges: Optional[Sequence[Edge]],
    config: RunConfig,
) -> Dict[str, float]:
    network = config.network
    if network.algorithm == "none" or not edges:
        return dict(direct)
    return propagate(
        direct,
        edges,
        algorithm=network.algorithm,
        damping=network.damping,
        restart_prob=network.restart_prob,
        max_neighbors_per_seed=network.max_neighbors_per_seed,
        min_score=network.min_score,
        blend=network.blend,
        network_discount=network.network_discount,
        max_network_genes=network.max_network_genes,
        heat_steps=network.heat_steps,
        heat_step_size=network.heat_step_size,
    )


def _fold_direct_scores(
    evidence: DiseaseEvidence,
    config: RunConfig,
    *,
    fold_genes: Iterable[str],
    protocol: str,
) -> Dict[str, float]:
    holdout = config.evaluation.holdout
    if protocol == "mask":
        return fuse_gene_scores(
            evidence.gene_type_scores, config.weights, holdout_type=holdout
        )
    restored = set(fold_genes)
    adjusted: Dict[str, Dict[str, float]] = {}
    for gene, tmap in evidence.gene_type_scores.items():
        if gene in restored:
            adjusted[gene] = dict(tmap)
        else:
            adjusted[gene] = {t: v for t, v in tmap.items() if t != holdout}
    return fuse_gene_scores(adjusted, config.weights)


def evaluate_evidence(
    evidence: DiseaseEvidence,
    *,
    config: Optional[RunConfig] = None,
    edges: Optional[Sequence[Edge]] = None,
) -> EvaluationResult:
    """Run dual-holdout K-fold evaluation on already-loaded evidence."""
    config = config or RunConfig()
    ev = config.evaluation
    positives = build_positives(
        evidence.gene_type_scores,
        ev.holdout,
        threshold=ev.positive_threshold,
    )
    if len(positives) < ev.folds:
        raise ValueError(
            f"Only {len(positives)} {ev.holdout}-positive genes for "
            f"{evidence.disease_id}; {ev.folds}-fold CV needs at least {ev.folds}. "
            "Lower --folds, lower --positive-threshold, or fetch more genes "
            "with a larger --max-genes."
        )

    splits = kfold_splits(positives, n_folds=ev.folds, seed=ev.seed)
    fold_results: List[FoldResult] = []

    for index, test_fold in enumerate(splits, start=1):
        train_fold = train_positives(positives, test_fold)
        if ev.protocol == "mask":
            eval_positives = set(test_fold)
            excluded: set = set()
        else:
            eval_positives = set(train_fold)
            excluded = set(test_fold)

        direct = _fold_direct_scores(
            evidence,
            config,
            fold_genes=test_fold,
            protocol=ev.protocol,
        )
        scores = _propagate_scores(direct, edges, config)
        ranked = rank_genes(scores)
        if excluded:
            ranked = [g for g in ranked if g not in excluded]

        metrics = {
            "auprc": auprc(ranked, eval_positives),
            f"recall_at_{ev.k}": recall_at_k(ranked, eval_positives, k=ev.k),
            "n_positives": float(len(eval_positives)),
            "n_ranked": float(len(ranked)),
        }
        fold_results.append(
            FoldResult(
                fold=index,
                n_eval_positives=len(eval_positives),
                n_train_positives=len(train_fold),
                n_excluded=len(excluded),
                metrics=metrics,
            )
        )

    metric_names = sorted({m for f in fold_results for m in f.metrics})
    summary = {
        name: _summarize([f.metrics.get(name, 0.0) for f in fold_results])
        for name in metric_names
    }
    return EvaluationResult(
        disease_id=evidence.disease_id,
        disease_name=evidence.disease_name,
        holdout=ev.holdout,
        protocol=ev.protocol,
        folds=ev.folds,
        seed=ev.seed,
        k=ev.k,
        fold_results=fold_results,
        summary=summary,
    )


def evaluate_disease(
    disease: Union[str, Disease],
    *,
    config: Optional[RunConfig] = None,
    cache: Optional[Cache] = None,
    cache_dir: Optional[PathLike] = None,
    client=None,
    evidence: Optional[DiseaseEvidence] = None,
    edges: Optional[Sequence[Edge]] = None,
    offline: bool = False,
    refresh: bool = False,
) -> EvaluationResult:
    """Load inputs for one disease and evaluate it."""
    _, _, loaded, loaded_edges = load_inputs(
        disease,
        config=config,
        cache=cache,
        cache_dir=cache_dir,
        client=client,
        evidence=evidence,
        edges=edges,
        offline=offline,
        refresh=refresh,
    )
    return evaluate_evidence(loaded, config=config, edges=loaded_edges)


def compare_evaluations(
    baseline: Sequence[EvaluationResult],
    candidate: Sequence[EvaluationResult],
    *,
    metric: str = "auprc",
    n_bootstrap: int = 2000,
    seed: int = 42,
) -> Dict[str, Any]:
    """Paired comparison of two method variants across diseases."""
    base_by_id = {r.disease_id: r for r in baseline}
    cand_by_id = {r.disease_id: r for r in candidate}
    shared = sorted(set(base_by_id) & set(cand_by_id))
    if not shared:
        raise ValueError("No shared diseases between baseline and candidate results.")

    deltas = [cand_by_id[d].metric(metric) - base_by_id[d].metric(metric) for d in shared]
    wins = sum(1 for d in deltas if d > 0)
    losses = sum(1 for d in deltas if d < 0)
    ties = len(deltas) - wins - losses
    bootstrap = paired_bootstrap(
        [cand_by_id[d].metric(metric) for d in shared],
        [base_by_id[d].metric(metric) for d in shared],
        n_bootstrap=n_bootstrap,
        seed=seed,
    )
    sign = sign_test(wins, losses)
    return {
        "metric": metric,
        "n_diseases": len(shared),
        "baseline_mean": float(np.mean([base_by_id[d].metric(metric) for d in shared])),
        "candidate_mean": float(np.mean([cand_by_id[d].metric(metric) for d in shared])),
        "delta_mean": float(np.mean(deltas)),
        "delta_bootstrap": bootstrap,
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "sign_test_p": sign["p_value"],
        "per_disease": {
            d: {
                "baseline": base_by_id[d].metric(metric),
                "candidate": cand_by_id[d].metric(metric),
                "delta": cand_by_id[d].metric(metric) - base_by_id[d].metric(metric),
            }
            for d in shared
        },
    }


def adjust_families(
    comparisons: Mapping[str, Dict[str, Any]],
    *,
    alpha: float = 0.05,
) -> Dict[str, Dict[str, Any]]:
    """Apply Benjamini-Hochberg within a family of comparisons (by metric)."""
    metrics = list(comparisons)
    pvalues = [float(comparisons[m]["sign_test_p"]) for m in metrics]
    adjusted = benjamini_hochberg(pvalues, alpha=alpha)
    for index, metric_name in enumerate(metrics):
        comparisons[metric_name]["sign_test_p_adj"] = adjusted["adjusted"][index]
        comparisons[metric_name]["rejected_fdr"] = adjusted["rejected"][index]
    return dict(comparisons)


def ranking_from_inputs(
    disease: Union[str, Disease],
    *,
    config: Optional[RunConfig] = None,
    cache: Optional[Cache] = None,
    cache_dir: Optional[PathLike] = None,
    client=None,
    evidence: Optional[DiseaseEvidence] = None,
    edges: Optional[Sequence[Edge]] = None,
    offline: bool = False,
    refresh: bool = False,
):
    """Convenience wrapper returning ``(resolved, config, evidence, edges)``."""
    return load_inputs(
        disease,
        config=config,
        cache=cache,
        cache_dir=cache_dir,
        client=client,
        evidence=evidence,
        edges=edges,
        offline=offline,
        refresh=refresh,
    )
