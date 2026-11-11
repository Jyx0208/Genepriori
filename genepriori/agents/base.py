"""Shared types for the multi-agent claim debate."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Sequence

from ..config import RunConfig
from ..data.opentargets import DiseaseEvidence
from ..data.registry import Disease
from ..evidence import STANDARD_TYPES
from ..leakage import Claim
from ..network import Edge


@dataclass
class Proposal:
    """A claim together with the agent's own predicted effect."""

    claim: Claim
    predicted_auprc: float = 0.0
    predicted_recall: float = 0.0
    novelty: float = 0.5


@dataclass
class ScoredClaim:
    """A claim after prediction, execution and governance."""

    claim: Claim
    round: int
    predicted_score: float = 0.0
    measured_auprc: float = 0.0
    measured_recall: float = 0.0
    accepted: bool = False
    reason: str = ""
    per_disease: Dict[str, float] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DebateContext:
    """Everything an agent needs to propose claims."""

    diseases: Sequence[Disease]
    evidence: Dict[str, DiseaseEvidence]
    edges: Optional[Sequence[Edge]]
    config: RunConfig
    holdout: str
    round: int = 1
    leaderboard: List[ScoredClaim] = field(default_factory=list)
    feedback: str = ""

    def best(self) -> Optional[ScoredClaim]:
        accepted = [entry for entry in self.leaderboard if entry.accepted]
        if not accepted:
            return None
        return max(accepted, key=lambda entry: entry.measured_auprc)


class Agent(Protocol):
    """Protocol implemented by heuristic and LLM agents."""

    name: str

    def propose(self, context: DebateContext) -> List[Proposal]: ...


def sanitize_weights(
    weights: Dict[str, float],
    holdout: str,
) -> Dict[str, float]:
    """Zero the holdout type and renormalize; raises if nothing is left."""
    cleaned = {t: max(0.0, float(weights.get(t, 0.0))) for t in STANDARD_TYPES}
    cleaned[holdout] = 0.0
    total = sum(cleaned.values())
    if total <= 0:
        raise ValueError(
            f"Claim assigns no weight to non-holdout evidence types for {holdout!r}."
        )
    return {t: v / total for t, v in cleaned.items()}


def make_claim(
    *,
    claim_id: str,
    agent_name: str,
    title: str,
    hypothesis: str,
    weights: Dict[str, float],
    holdout: str,
    algorithm: str = "ppr",
    network_params: Optional[Dict[str, Any]] = None,
    cost: float = 1.0,
    risk_tags: Optional[Sequence[str]] = None,
    falsification_test: str = "",
    predicted_auprc: float = 0.0,
    predicted_recall: float = 0.0,
) -> Proposal:
    """Build a Proposal with holdout-safe weights."""
    safe_weights = sanitize_weights(weights, holdout)
    intervention: Dict[str, Any] = {
        "evidence_type_weights": safe_weights,
        "network_algorithm": algorithm,
        "network_params": dict(network_params or {}),
    }
    claim = Claim(
        claim_id=claim_id,
        agent_name=agent_name,
        title=title,
        hypothesis=hypothesis,
        intervention=intervention,
        expected_delta={"auprc": float(predicted_auprc), "recall": float(predicted_recall)},
        risk_tags=list(risk_tags or ()),
        cost_estimate=float(cost),
        falsification_test=falsification_test,
    )
    return Proposal(
        claim=claim,
        predicted_auprc=float(predicted_auprc),
        predicted_recall=float(predicted_recall),
    )
