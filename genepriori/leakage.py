"""LeakageAuditor policy checks (Methods; claim-level, not residual-correlation free)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Tuple

from .evidence import normalize_holdout_type

SUPPORTED_NETWORK_ALGORITHMS = {
    "",
    "none",
    "ppr",
    "rwr",
    "neighbor_voting",
    "hd",
    "ensemble",
}


@dataclass
class Claim:
    """Claim object for the leakage audit workflow."""

    claim_id: str
    agent_name: str = "unspecified"
    title: str = ""
    hypothesis: str = ""
    intervention: Dict[str, Any] = field(default_factory=dict)
    expected_delta: Dict[str, float] = field(default_factory=dict)
    risk_tags: list = field(default_factory=list)
    cost_estimate: float = 1.0
    falsification_test: str = ""


class LeakageAuditor:
    """Hard-constraint auditor for *direct* holdout-type leakage in a claim.

    Pass criteria (policy checks):
      1. Global evidence_type_weights[holdout] must be 0.
      2. disease_specific_weights[*][holdout] must be 0.
      3. allow_holdout_evidence must not be true.
      4. network_algorithm (global and per-disease) must be in the supported set.

    This does **not** remove semantic collinearity among non-holdout evidence
    types (e.g., literature vs drug annotations). Positive genes may still be
    visible through complementary evidence during evaluation by design.
    """

    name = "LeakageAuditor"

    def audit(self, claim: Claim, holdout_type: str) -> Tuple[bool, str]:
        holdout = normalize_holdout_type(holdout_type)
        intervention = claim.intervention or {}
        weights = intervention.get("evidence_type_weights", {}) or {}

        if float(weights.get(holdout, 0.0) or 0.0) > 0.0:
            return False, f"holdout evidence type '{holdout}' has non-zero weight"

        if bool(intervention.get("allow_holdout_evidence", False)):
            return False, "intervention explicitly enables holdout evidence"

        algo = str(intervention.get("network_algorithm", "")).strip().lower()
        if algo not in SUPPORTED_NETWORK_ALGORITHMS:
            return False, f"unsupported network algorithm: {algo}"

        ds_weights = intervention.get("disease_specific_weights", {}) or {}
        if not isinstance(ds_weights, dict):
            return False, "disease_specific_weights must be a dict"
        for disease_id, dsw in ds_weights.items():
            if not isinstance(dsw, dict):
                return False, f"disease_specific_weights[{disease_id}] must be a dict"
            if float(dsw.get(holdout, 0.0) or 0.0) > 0.0:
                return (
                    False,
                    f"holdout evidence type '{holdout}' has non-zero "
                    f"disease-specific weight for {disease_id}",
                )

        ds_network = intervention.get("disease_specific_network", {}) or {}
        if not isinstance(ds_network, dict):
            return False, "disease_specific_network must be a dict"
        for disease_id, cfg in ds_network.items():
            if not isinstance(cfg, dict):
                return False, f"disease_specific_network[{disease_id}] must be a dict"
            d_algo = str(cfg.get("network_algorithm", algo)).strip().lower()
            if d_algo not in SUPPORTED_NETWORK_ALGORITHMS:
                return (
                    False,
                    f"unsupported disease-specific network algorithm "
                    f"for {disease_id}: {d_algo}",
                )

        return True, "passed"


def apply_holdout_mask_to_weights(
    weights: Mapping[str, float],
    holdout_type: str,
) -> Dict[str, float]:
    """Convenience wrapper used at execution time (in addition to claim audit)."""
    from .evidence import renorm_weights

    return renorm_weights(weights, holdout_type=holdout_type)
