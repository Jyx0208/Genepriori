"""Governor: budget allocation, claim scoring and acceptance criteria."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from ..metrics import bootstrap_mean_ci
from .base import Proposal


@dataclass
class GovernorConfig:
    """Parameters of the governance protocol (Methods)."""

    budget: int = 12
    rounds: int = 3
    min_delta: float = 1e-4
    majority_win_rate: float = 0.60
    strong_win_rate: float = 0.80
    auprc_weight: float = 1.0
    recall_weight: float = 0.5
    cost_weight: float = 0.05
    novelty_weight: float = 0.02
    n_bootstrap: int = 2000
    seed: int = 42


@dataclass
class Decision:
    accepted: bool
    reason: str
    mean_delta: float = 0.0
    win_rate: float = 0.0
    ci_low: float = 0.0
    ci_high: float = 0.0
    p_one_sided: float = 1.0
    n_diseases: int = 0

    def to_dict(self) -> Dict[str, float]:
        return {
            "accepted": self.accepted,
            "reason": self.reason,
            "mean_delta": self.mean_delta,
            "win_rate": self.win_rate,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "p_one_sided": self.p_one_sided,
            "n_diseases": float(self.n_diseases),
        }


class Governor:
    """Allocates the claim budget and enforces the three acceptance criteria."""

    def __init__(self, config: Optional[GovernorConfig] = None) -> None:
        self.config = config or GovernorConfig()

    # ----------------------------------------------------------------- budget
    def round_budget(self, round_no: int) -> int:
        """40% / 35% / remaining split, with at least one claim per round."""
        total = max(0, int(self.config.budget))
        rounds = max(1, int(self.config.rounds))
        if round_no < 1 or round_no > rounds:
            return 0
        if rounds == 1:
            return total
        if rounds == 2:
            share = 0.55 if round_no == 1 else 0.45
        else:
            shares = {1: 0.40, 2: 0.35}
            share = shares.get(round_no)
        if share is not None:
            return max(1, int(math.floor(total * share)))
        allocated = sum(
            max(1, int(math.floor(total * (0.40 if r == 1 else 0.35))))
            for r in range(1, rounds)
        )
        return max(0, total - allocated)

    # ----------------------------------------------------------------- scoring
    def score(self, proposal: Proposal) -> float:
        cfg = self.config
        return (
            cfg.auprc_weight * proposal.predicted_auprc
            + cfg.recall_weight * proposal.predicted_recall
            - cfg.cost_weight * proposal.claim.cost_estimate
            + cfg.novelty_weight * proposal.novelty
        )

    def select(self, proposals: Sequence[Proposal], k: int) -> List[Proposal]:
        """Rank by predicted score, dropping duplicate interventions."""
        scored = sorted(
            proposals,
            key=lambda p: (-self.score(p), p.claim.claim_id),
        )
        chosen: List[Proposal] = []
        seen = set()
        for proposal in scored:
            fingerprint = self.fingerprint(proposal)
            if fingerprint in seen:
                continue
            seen.add(fingerprint)
            chosen.append(proposal)
            if len(chosen) >= k:
                break
        return chosen

    @staticmethod
    def fingerprint(proposal: Proposal) -> str:
        intervention = proposal.claim.intervention or {}
        payload = {
            "weights": {k: round(float(v), 6) for k, v in (intervention.get("evidence_type_weights") or {}).items()},
            "algorithm": intervention.get("network_algorithm"),
            "params": {
                k: round(float(v), 6) if isinstance(v, (int, float)) else v
                for k, v in (intervention.get("network_params") or {}).items()
            },
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]

    # -------------------------------------------------------------- acceptance
    def decide(
        self,
        baseline: Dict[str, float],
        candidate: Dict[str, float],
    ) -> Decision:
        """Compare per-disease AUPRC maps and apply the three criteria.

        1. Majority rule: win rate >= 60% and mean delta above ``min_delta``.
        2. Statistical support: bootstrap 95% CI lower bound > 0.
        3. Strong override: win rate >= 80% and mean delta above 2x threshold.
        """
        shared = sorted(set(baseline) & set(candidate))
        if not shared:
            return Decision(False, "no shared diseases")
        deltas = [float(candidate[d]) - float(baseline[d]) for d in shared]
        mean_delta = sum(deltas) / len(deltas)
        wins = sum(1 for d in deltas if d > 0)
        win_rate = wins / len(deltas)
        stats = bootstrap_mean_ci(
            deltas,
            n_bootstrap=self.config.n_bootstrap,
            seed=self.config.seed,
        )

        if win_rate >= self.config.strong_win_rate and mean_delta > 2 * self.config.min_delta:
            return Decision(
                True,
                "strong override: consistent wins with large mean gain",
                mean_delta,
                win_rate,
                stats["ci_low"],
                stats["ci_high"],
                stats["p_one_sided"],
                len(deltas),
            )
        if stats["ci_low"] > 0.0:
            return Decision(
                True,
                "statistically supported: bootstrap CI excludes zero",
                mean_delta,
                win_rate,
                stats["ci_low"],
                stats["ci_high"],
                stats["p_one_sided"],
                len(deltas),
            )
        if win_rate >= self.config.majority_win_rate and mean_delta > self.config.min_delta:
            return Decision(
                True,
                "majority rule: consistent improvement across diseases",
                mean_delta,
                win_rate,
                stats["ci_low"],
                stats["ci_high"],
                stats["p_one_sided"],
                len(deltas),
            )
        reason = (
            "no criterion met: "
            f"mean delta {mean_delta:+.5f}, win rate {win_rate:.2f}, "
            f"CI low {stats['ci_low']:+.5f}"
        )
        return Decision(
            False,
            reason,
            mean_delta,
            win_rate,
            stats["ci_low"],
            stats["ci_high"],
            stats["p_one_sided"],
            len(deltas),
        )
