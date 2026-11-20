"""Select a final scoring configuration and rank genes after a debate."""

from __future__ import annotations

from statistics import mean
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from ..config import RunConfig
from ..data.cache import Cache
from ..data.registry import Disease
from ..leakage import Claim
from ..pipeline.evaluation import evaluate_evidence
from ..pipeline.fusion import RankedResult, build_ranking, load_inputs
from .debate import config_for_claim
from .progress import DebateProgress


def recommend_rankings(
    diseases: Sequence[Disease],
    reports: Mapping[str, Mapping[str, Any]],
    *,
    config: RunConfig,
    cache: Cache,
    progress: Optional[Callable[[str], None]] = None,
) -> Tuple[Dict[str, Any], List[RankedResult]]:
    status = DebateProgress(progress)
    candidates = {"baseline": config}
    if config.network.algorithm != "none":
        direct = config.to_dict()
        direct["network"]["algorithm"] = "none"
        candidates["evidence_only"] = RunConfig.from_dict(direct)

    origins: Dict[str, Dict[str, str]] = {}
    for holdout, report in reports.items():
        best = report.get("best")
        if best is None:
            continue
        claim = Claim(best["claim_id"], agent_name=best["agent"], intervention=best["intervention"])
        payload = config_for_claim(claim, config, holdout).to_dict()
        restored = config.weights[holdout]
        payload["weights"] = {
            kind: restored if kind == holdout else weight * (1 - restored)
            for kind, weight in payload["weights"].items()
        }
        if config.network.algorithm == "none":
            payload["network"]["algorithm"] = "none"
        name = f"{holdout}_winner"
        candidates[name] = RunConfig.from_dict(payload)
        origins[name] = {"claim_id": claim.claim_id, "agent": claim.agent_name, "holdout": holdout}

    evidence = []
    edges = None
    with status.waiting("Loading cached evidence and network for final ranking"):
        for disease in diseases:
            _, _, item, edges = load_inputs(
                disease, config=config, cache=cache, edges=edges, offline=True
            )
            evidence.append(item)

    scores: Dict[str, Dict[str, Any]] = {}
    for name, candidate in candidates.items():
        scores[name] = {"auprc": {}, **origins.get(name, {})}
        for holdout in reports:
            validation = candidate.with_holdout(holdout)
            with status.waiting(f"Comparing {name} on {holdout} holdout"):
                score = mean(
                    evaluate_evidence(item, config=validation, edges=edges).metric("auprc")
                    for item in evidence
                )
            scores[name]["auprc"][holdout] = score
            status(f"{name}: {holdout} AUPRC {score:.5f}")

    baseline = scores["baseline"]["auprc"]
    for record in scores.values():
        deltas = {holdout: score - baseline[holdout] for holdout, score in record["auprc"].items()}
        record["qualifies"] = all(delta >= -1e-12 for delta in deltas.values())
        record["mean_gain"] = mean(
            delta / baseline[holdout] if baseline[holdout] else delta
            for holdout, delta in deltas.items()
        )
    selected = max(
        (name for name in scores if scores[name]["qualifies"]),
        key=lambda name: scores[name]["mean_gain"],
    )
    status(f"Selected scoring configuration: {selected}")
    final = candidates[selected]
    rankings = []
    for item in evidence:
        with status.waiting(f"Ranking genes for {item.disease_id} and comparing with the baseline"):
            ranking = build_ranking(item, config=final, edges=edges)
            baseline_ranking = (
                ranking if selected == "baseline" else build_ranking(item, config=config, edges=edges)
            )
            baseline_ranks = {row.gene: row.rank for row in baseline_ranking.ranked}
            for row in ranking.ranked:
                row.baseline_rank = baseline_ranks.get(row.gene)
            rankings.append(ranking)
    selection = {
        "selected": selected,
        "holdouts": list(reports),
        "rule": (
            "Require AUPRC at least as high as the configured baseline on every requested "
            "holdout. Maximize mean relative AUPRC gain, using absolute gain when baseline "
            "AUPRC is zero. Prefer the baseline when scores tie."
        ),
        "claim_weight_restoration": (
            "Restore each winner's held-out evidence to its configured baseline share "
            "and preserve the relative weights of the other evidence types."
        ),
        "validation": "Exploratory comparison on the debate data; no independent validation.",
        "candidates": scores,
        "config": final.to_dict(),
    }
    return selection, rankings
