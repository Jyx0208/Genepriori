"""Three-round, six-step claim debate orchestration.

Each round runs: propose -> audit -> score/select -> execute -> accept/reject
-> feedback. The Governor owns the budget and the acceptance criteria; the
LeakageAuditor is a hard gate before any claim is executed.
"""

from __future__ import annotations

import logging
from dataclasses import fields
from typing import Any, Dict, List, Optional, Sequence

from ..config import NetworkConfig, RunConfig
from ..data.cache import Cache
from ..data.opentargets import DiseaseEvidence, fetch_disease_evidence
from ..data.registry import Disease
from ..data.string import load_string_edges
from ..leakage import Claim, LeakageAuditor
from ..pipeline.evaluation import EvaluationResult, evaluate_evidence
from .base import DebateContext, Proposal, ScoredClaim
from .governor import Governor, GovernorConfig
from .heuristic import default_agents
from .llm import LLMAgent, LLMClient

LOG = logging.getLogger(__name__)

_NETWORK_FIELDS = {f.name for f in fields(NetworkConfig)}

PERSONAS = {
    "CausalGenetics": "You prioritise causal genetic evidence and conservative propagation.",
    "NetworkMechanism": "You explore propagation algorithms and their parameters.",
    "Translational": "You balance clinical/drug evidence against biological support.",
    "ConsensusDesigner": "You synthesise the strongest claims proposed so far.",
}


def config_for_claim(claim: Claim, base: RunConfig, holdout: str) -> RunConfig:
    """Translate a claim's intervention into a validated run configuration."""
    payload = base.to_dict()
    weights = dict(payload["weights"])
    weights.update(
        {
            str(k): float(v)
            for k, v in (claim.intervention.get("evidence_type_weights") or {}).items()
        }
    )
    weights[holdout] = 0.0
    payload["weights"] = weights
    payload["evaluation"]["holdout"] = holdout

    algorithm = claim.intervention.get("network_algorithm")
    if algorithm is not None:
        payload["network"]["algorithm"] = str(algorithm)
    params = claim.intervention.get("network_params") or {}
    for key, value in params.items():
        if key in _NETWORK_FIELDS:
            payload["network"][key] = value
    return RunConfig.from_dict(payload)


def _summarize(results: Dict[str, EvaluationResult], metric: str) -> Dict[str, float]:
    values = [result.metric(metric) for result in results.values()]
    return {
        "mean": sum(values) / len(values) if values else 0.0,
        "per_disease": {key: result.metric(metric) for key, result in results.items()},
    }


def _execute(
    proposal: Proposal,
    *,
    base: RunConfig,
    evidence: Dict[str, DiseaseEvidence],
    edges,
    holdout: str,
) -> Dict[str, EvaluationResult]:
    config = config_for_claim(proposal.claim, base, holdout)
    results: Dict[str, EvaluationResult] = {}
    for disease_id, item in evidence.items():
        results[disease_id] = evaluate_evidence(item, config=config, edges=edges)
    return results


def run_debate(
    diseases: Sequence[Disease],
    *,
    config: Optional[RunConfig] = None,
    cache: Optional[Cache] = None,
    rounds: int = 3,
    budget: int = 12,
    mode: str = "heuristic",
    holdout: str = "drug",
    offline: bool = False,
    refresh: bool = False,
    client=None,
    llm_client: Optional[LLMClient] = None,
) -> Dict[str, Any]:
    """Run the full debate for one holdout and return a JSON-serialisable report."""
    config = config or RunConfig()
    cache = cache if cache is not None else Cache()
    base = config.with_holdout(holdout)
    governor = Governor(
        GovernorConfig(budget=budget, rounds=rounds, seed=config.evaluation.seed)
    )
    auditor = LeakageAuditor()

    evidence: Dict[str, DiseaseEvidence] = {}
    for disease in diseases:
        evidence[disease.disease_id] = fetch_disease_evidence(
            disease,
            client=client,
            cache=cache,
            max_genes=base.max_genes,
            refresh=refresh,
            offline=offline,
        )

    edges = None
    if base.network.algorithm != "none":
        edges = load_string_edges(
            version=base.string_version,
            min_combined_score=base.min_combined_score,
            cache=cache,
            refresh=refresh,
            offline=offline,
        )

    baseline_results = {
        disease_id: evaluate_evidence(item, config=base, edges=edges)
        for disease_id, item in evidence.items()
    }
    baseline_auprc = {k: r.metric("auprc") for k, r in baseline_results.items()}
    baseline_recall = {
        k: r.metric(f"recall_at_{base.evaluation.k}") for k, r in baseline_results.items()
    }

    agents = default_agents()
    if mode == "llm":
        llm_client = llm_client or LLMClient()
        agents = [
            LLMAgent(
                agent.name,
                PERSONAS.get(agent.name, "You propose gene-prioritisation strategies."),
                client=llm_client,
                fallback=agent,
            )
            for agent in agents
        ]

    leaderboard: List[ScoredClaim] = []
    decisions: List[Dict[str, Any]] = []
    seen_fingerprints: set = set()

    for round_no in range(1, rounds + 1):
        context = DebateContext(
            diseases=list(diseases),
            evidence=evidence,
            edges=edges,
            config=base,
            holdout=holdout,
            round=round_no,
            leaderboard=leaderboard,
            feedback=(
                f"best so far: {leaderboard[-1].claim.claim_id}"
                if leaderboard
                else "no claims executed yet"
            ),
        )
        proposals: List[Proposal] = []
        for agent in agents:
            try:
                proposals.extend(agent.propose(context))
            except Exception as exc:  # noqa: BLE001 - one agent must not stop the round
                LOG.warning("agent %s failed in round %d: %s", agent.name, round_no, exc)

        audited: List[Proposal] = []
        for proposal in proposals:
            ok, reason = auditor.audit(proposal.claim, holdout)
            if not ok:
                decisions.append(
                    {
                        "round": round_no,
                        "claim_id": proposal.claim.claim_id,
                        "agent": proposal.claim.agent_name,
                        "title": proposal.claim.title,
                        "score": 0.0,
                        "predicted_score": governor.score(proposal),
                        "accepted": False,
                        "reason": f"leakage audit failed: {reason}",
                        "intervention": proposal.claim.intervention,
                    }
                )
                continue
            fingerprint = governor.fingerprint(proposal)
            proposal.novelty = 0.0 if fingerprint in seen_fingerprints else 1.0
            seen_fingerprints.add(fingerprint)
            audited.append(proposal)

        slots = governor.round_budget(round_no)
        if round_no == 1:
            slots = max(0, slots - 1)  # the baseline occupies one slot
        selected = governor.select(audited, slots)

        for proposal in selected:
            predicted = governor.score(proposal)
            try:
                results = _execute(
                    proposal,
                    base=base,
                    evidence=evidence,
                    edges=edges,
                    holdout=holdout,
                )
            except (ValueError, TypeError) as exc:
                decisions.append(
                    {
                        "round": round_no,
                        "claim_id": proposal.claim.claim_id,
                        "agent": proposal.claim.agent_name,
                        "title": proposal.claim.title,
                        "score": 0.0,
                        "predicted_score": predicted,
                        "accepted": False,
                        "reason": f"execution failed: {exc}",
                        "intervention": proposal.claim.intervention,
                    }
                )
                continue

            auprc = _summarize(results, "auprc")
            recall = _summarize(results, f"recall_at_{base.evaluation.k}")
            decision = governor.decide(baseline_auprc, auprc["per_disease"])
            entry = ScoredClaim(
                claim=proposal.claim,
                round=round_no,
                predicted_score=predicted,
                measured_auprc=auprc["mean"],
                measured_recall=recall["mean"],
                accepted=decision.accepted,
                reason=decision.reason,
                per_disease=auprc["per_disease"],
                metrics={
                    "executed": True,
                    "auprc": auprc["mean"],
                    "recall": recall["mean"],
                    "decision": decision.to_dict(),
                },
            )
            leaderboard.append(entry)
            decisions.append(
                {
                    "round": round_no,
                    "claim_id": proposal.claim.claim_id,
                    "agent": proposal.claim.agent_name,
                    "title": proposal.claim.title,
                    "score": round(entry.measured_auprc, 8),
                    "predicted_score": round(predicted, 8),
                    "accepted": decision.accepted,
                    "reason": decision.reason,
                    "delta": round(decision.mean_delta, 8),
                    "win_rate": round(decision.win_rate, 4),
                    "ci_low": round(decision.ci_low, 8),
                    "intervention": proposal.claim.intervention,
                }
            )

    accepted = [entry for entry in leaderboard if entry.accepted]
    best = max(accepted, key=lambda entry: entry.measured_auprc) if accepted else None

    return {
        "mode": mode,
        "holdout": holdout,
        "rounds": rounds,
        "budget": budget,
        "diseases": [{"disease_id": d.disease_id, "name": d.name} for d in diseases],
        "baseline": {
            "auprc": _summarize(baseline_results, "auprc"),
            "recall": _summarize(baseline_results, f"recall_at_{base.evaluation.k}"),
            "weights": dict(base.weights),
            "network": base.network.algorithm,
        },
        "decisions": decisions,
        "leaderboard": [
            {
                "claim_id": entry.claim.claim_id,
                "agent": entry.claim.agent_name,
                "round": entry.round,
                "auprc": entry.measured_auprc,
                "recall": entry.measured_recall,
                "accepted": entry.accepted,
                "reason": entry.reason,
                "intervention": entry.claim.intervention,
            }
            for entry in leaderboard
        ],
        "best": (
            {
                "claim_id": best.claim.claim_id,
                "agent": best.claim.agent_name,
                "title": best.claim.title,
                "auprc": best.measured_auprc,
                "recall": best.measured_recall,
                "intervention": best.claim.intervention,
                "per_disease": best.per_disease,
            }
            if best is not None
            else None
        ),
    }
