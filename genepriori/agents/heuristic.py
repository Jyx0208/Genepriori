"""Heuristic agents: deterministic claim generators requiring no LLM.

The four personas are: CausalGenetics emphasises genetic evidence, NetworkMechanism explores propagation configurations, Translational
balances biological and clinical signals, and ConsensusDesigner only appears
from round 2 and synthesises the current leaderboard.
"""

from __future__ import annotations

from typing import Dict, List

from .base import Agent, DebateContext, Proposal, make_claim

BASE_TRANSLATIONAL = {
    "drug": 0.30,
    "genetics": 0.25,
    "expression": 0.05,
    "pathway": 0.20,
    "literature": 0.15,
    "animal_model": 0.05,
}


class CausalGeneticsAgent(Agent):
    name = "CausalGenetics"

    def propose(self, context: DebateContext) -> List[Proposal]:
        round_no = context.round
        best = context.best()
        genetics_weight = 0.55
        if best is not None and best.claim.agent_name == self.name:
            genetics_weight = min(0.75, genetics_weight + 0.05 * round_no)
        tail = (1.0 - genetics_weight) / 4.0
        weights = {
            "drug": 0.05,
            "genetics": genetics_weight,
            "expression": tail,
            "pathway": tail,
            "literature": tail,
            "animal_model": 0.0,
        }
        proposals = [
            make_claim(
                claim_id=f"cg-r{round_no}-a",
                agent_name=self.name,
                title=f"Genetics-first fusion (w={genetics_weight:.2f})",
                hypothesis=(
                    "Genetic association evidence is the most disease-specific signal; "
                    "up-weighting it should improve recovery across holdouts."
                ),
                weights=weights,
                holdout=context.holdout,
                algorithm="ppr",
                network_params={"damping": 0.85, "blend": 0.70},
                cost=1.0,
                risk_tags=["genetics-heavy"],
                falsification_test="No AUPRC gain over balanced weights on any disease.",
                predicted_auprc=0.010,
                predicted_recall=0.005,
            ),
            make_claim(
                claim_id=f"cg-r{round_no}-b",
                agent_name=self.name,
                title="Genetics-first with RWR propagation",
                hypothesis="RWR with a moderate restart probability spreads genetic signal locally.",
                weights=weights,
                holdout=context.holdout,
                algorithm="rwr",
                network_params={"restart_prob": 0.30, "blend": 0.70},
                cost=1.1,
                risk_tags=["genetics-heavy", "network"],
                falsification_test="RWR fails to beat PPR on network-discovered positives.",
                predicted_auprc=0.008,
                predicted_recall=0.004,
            ),
        ]
        if round_no >= 2:
            weights_hd = dict(weights)
            proposals.append(
                make_claim(
                    claim_id=f"cg-r{round_no}-c",
                    agent_name=self.name,
                    title="Genetics-first with heat diffusion",
                    hypothesis="Diffusion over the normalized Laplacian reduces hub dominance.",
                    weights=weights_hd,
                    holdout=context.holdout,
                    algorithm="hd",
                    network_params={"heat_steps": 12},
                    cost=0.9,
                    risk_tags=["genetics-heavy", "network"],
                    falsification_test="Diffusion underperforms the same weights without propagation.",
                    predicted_auprc=0.006,
                    predicted_recall=0.003,
                )
            )
        return proposals


class NetworkMechanismAgent(Agent):
    name = "NetworkMechanism"

    CONFIGS = [
        ("ppr", {"damping": 0.75, "blend": 0.65}),
        ("rwr", {"restart_prob": 0.20, "blend": 0.65}),
        ("ensemble", {"damping": 0.85, "restart_prob": 0.30, "blend": 0.70}),
    ]

    def propose(self, context: DebateContext) -> List[Proposal]:
        round_no = context.round
        weights = {
            "drug": 0.15,
            "genetics": 0.20,
            "expression": 0.15,
            "pathway": 0.20,
            "literature": 0.20,
            "animal_model": 0.10,
        }
        proposals: List[Proposal] = []
        for index, (algorithm, params) in enumerate(self.CONFIGS):
            if index >= max(1, 3 - (round_no - 1)):
                continue
            params = dict(params)
            if round_no >= 2:
                params["max_network_genes"] = 120
            proposals.append(
                make_claim(
                    claim_id=f"nm-r{round_no}-{algorithm}",
                    agent_name=self.name,
                    title=f"{algorithm.upper()} propagation tuned for recall",
                    hypothesis=(
                        "Lower damping/restart widens the explored neighbourhood and "
                        "surfaces network-discovered candidates."
                    ),
                    weights=weights,
                    holdout=context.holdout,
                    algorithm=algorithm,
                    network_params=params,
                    cost=1.2,
                    risk_tags=["network", "hub-bias"],
                    falsification_test="No Recall@20 improvement over the no-network baseline.",
                    predicted_auprc=0.004,
                    predicted_recall=0.010,
                )
            )
        return proposals


class TranslationalAgent(Agent):
    name = "Translational"

    def propose(self, context: DebateContext) -> List[Proposal]:
        round_no = context.round
        weights = dict(BASE_TRANSLATIONAL)
        if round_no >= 2:
            weights["drug"] += 0.05
            weights["literature"] -= 0.05
        return [
            make_claim(
                claim_id=f"tr-r{round_no}-a",
                agent_name=self.name,
                title="Clinical-translational balance",
                hypothesis=(
                    "Drug and pathway evidence carry the most translational signal; "
                    "balancing them favours actionable targets."
                ),
                weights=weights,
                holdout=context.holdout,
                algorithm="ppr",
                network_params={"damping": 0.85, "blend": 0.60},
                cost=1.0,
                risk_tags=["clinical"],
                falsification_test="No Recall@20 gain over balanced weights.",
                predicted_auprc=0.006,
                predicted_recall=0.008,
            ),
            make_claim(
                claim_id=f"tr-r{round_no}-b",
                agent_name=self.name,
                title="Translational with suppressed literature",
                hypothesis=(
                    "Literature evidence is circular; damping it exposes whether other "
                    "signals carry the performance."
                ),
                weights={**weights, "literature": 0.02, "pathway": weights["pathway"] + 0.13},
                holdout=context.holdout,
                algorithm="ppr",
                network_params={"damping": 0.85, "blend": 0.70},
                cost=1.0,
                risk_tags=["literature-tautology"],
                falsification_test="AUPRC drops sharply, confirming literature dominance.",
                predicted_auprc=0.002,
                predicted_recall=0.004,
            ),
        ]


class ConsensusDesignerAgent(Agent):
    name = "ConsensusDesigner"

    def propose(self, context: DebateContext) -> List[Proposal]:
        if context.round < 2:
            return []
        executed = [entry for entry in context.leaderboard if entry.metrics.get("executed")]
        if not executed:
            return []
        top = sorted(executed, key=lambda entry: entry.measured_auprc, reverse=True)[:2]
        merged: Dict[str, float] = {}
        for entry in top:
            weights = entry.claim.intervention.get("evidence_type_weights", {}) or {}
            for key, value in weights.items():
                merged[key] = merged.get(key, 0.0) + float(value) / len(top)
        if not merged:
            return []
        best_network = top[0].claim.intervention.get("network_algorithm", "ppr")
        best_params = dict(top[0].claim.intervention.get("network_params") or {})
        return [
            make_claim(
                claim_id=f"cd-r{context.round}",
                agent_name=self.name,
                title="Consensus of the top-2 executed claims",
                hypothesis="Averaging the best-performing weight vectors is more robust than either.",
                weights=merged,
                holdout=context.holdout,
                algorithm=str(best_network),
                network_params=best_params,
                cost=1.1,
                risk_tags=["consensus"],
                falsification_test="Consensus fails to beat both parents on mean AUPRC.",
                predicted_auprc=0.012,
                predicted_recall=0.010,
            )
        ]


def default_agents() -> List[Agent]:
    return [
        CausalGeneticsAgent(),
        NetworkMechanismAgent(),
        TranslationalAgent(),
        ConsensusDesignerAgent(),
    ]
