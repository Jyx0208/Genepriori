"""Tests for agents, the Governor and the debate orchestration."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from genepriori.agents.base import DebateContext, make_claim, sanitize_weights
from genepriori.agents.debate import config_for_claim, run_debate
from genepriori.agents.governor import Governor, GovernorConfig
from genepriori.agents.heuristic import (
    CausalGeneticsAgent,
    ConsensusDesignerAgent,
    NetworkMechanismAgent,
    TranslationalAgent,
    default_agents,
)
from genepriori.agents.llm import LLMAgent, LLMClient, parse_claims
from genepriori.config import RunConfig
from genepriori.data.cache import Cache
from genepriori.data.registry import resolve_disease
from genepriori.data.toy import build_toy_cache
from genepriori.evidence import STANDARD_TYPES
from genepriori.leakage import LeakageAuditor


@pytest.fixture()
def context(tmp_path: Path) -> DebateContext:
    root = build_toy_cache(tmp_path / "cache")
    from genepriori.data.opentargets import fetch_disease_evidence

    cache = Cache(root)
    evidence = fetch_disease_evidence("MONDO_9000001", cache=cache, offline=True)
    return DebateContext(
        diseases=[resolve_disease("MONDO_9000001")],
        evidence={"MONDO_9000001": evidence},
        edges=None,
        config=RunConfig(),
        holdout="drug",
    )


# -------------------------------------------------------------------- weights
def test_sanitize_weights_zeroes_holdout_and_renormalizes() -> None:
    out = sanitize_weights({"drug": 5.0, "genetics": 1.0, "pathway": 1.0, "expression": 1.0}, "drug")
    assert out["drug"] == 0.0
    assert math.isclose(sum(out.values()), 1.0)


def test_sanitize_weights_rejects_degenerate() -> None:
    with pytest.raises(ValueError, match="no weight to non-holdout"):
        sanitize_weights({"drug": 1.0}, "drug")


def test_make_claim_records_intervention() -> None:
    proposal = make_claim(
        claim_id="x",
        agent_name="A",
        title="t",
        hypothesis="h",
        weights={"genetics": 1.0},
        holdout="drug",
        algorithm="rwr",
        network_params={"restart_prob": 0.2},
        predicted_auprc=0.01,
    )
    claim = proposal.claim
    assert claim.intervention["evidence_type_weights"]["drug"] == 0.0
    assert claim.intervention["network_algorithm"] == "rwr"
    assert claim.expected_delta["auprc"] == 0.01
    assert LeakageAuditor().audit(claim, "drug")[0]


# -------------------------------------------------------------------- agents
def test_all_heuristic_claims_pass_audit(context: DebateContext) -> None:
    auditor = LeakageAuditor()
    for agent in default_agents():
        for proposal in agent.propose(context):
            ok, reason = auditor.audit(proposal.claim, context.holdout)
            assert ok, f"{agent.name}: {reason}"
            weights = proposal.claim.intervention["evidence_type_weights"]
            assert set(weights) == set(STANDARD_TYPES)
            assert weights[context.holdout] == 0.0


def test_agents_are_deterministic(context: DebateContext) -> None:
    first = [p.claim.intervention for p in CausalGeneticsAgent().propose(context)]
    second = [p.claim.intervention for p in CausalGeneticsAgent().propose(context)]
    assert first == second


def test_causal_genetics_respects_holdout_genetics(context: DebateContext) -> None:
    context.holdout = "genetics"
    for proposal in CausalGeneticsAgent().propose(context):
        assert proposal.claim.intervention["evidence_type_weights"]["genetics"] == 0.0


def test_network_agent_explores_algorithms(context: DebateContext) -> None:
    algorithms = {
        p.claim.intervention["network_algorithm"] for p in NetworkMechanismAgent().propose(context)
    }
    assert {"ppr", "rwr", "ensemble"} <= algorithms


def test_translational_agent_has_literature_contrast(context: DebateContext) -> None:
    proposals = TranslationalAgent().propose(context)
    assert len(proposals) == 2
    assert (
        proposals[0].claim.intervention["evidence_type_weights"]["literature"]
        > proposals[1].claim.intervention["evidence_type_weights"]["literature"]
    )


def test_consensus_designer_silent_in_round_one(context: DebateContext) -> None:
    assert ConsensusDesignerAgent().propose(context) == []


# ------------------------------------------------------------------ governor
def test_round_budget_split() -> None:
    governor = Governor(GovernorConfig(budget=12, rounds=3))
    assert governor.round_budget(1) == 4
    assert governor.round_budget(2) == 4
    assert governor.round_budget(3) == 4
    assert governor.round_budget(4) == 0


def test_round_budget_two_rounds() -> None:
    governor = Governor(GovernorConfig(budget=10, rounds=2))
    assert governor.round_budget(1) == 5
    assert governor.round_budget(2) == 4


def test_score_prefers_predicted_gain() -> None:
    governor = Governor()
    high = make_claim(
        claim_id="a", agent_name="x", title="", hypothesis="", weights={"genetics": 1.0},
        holdout="drug", predicted_auprc=0.02, predicted_recall=0.02,
    )
    low = make_claim(
        claim_id="b", agent_name="x", title="", hypothesis="", weights={"genetics": 1.0},
        holdout="drug", predicted_auprc=0.001, predicted_recall=0.001,
    )
    assert governor.score(high) > governor.score(low)


def test_select_deduplicates_identical_interventions() -> None:
    governor = Governor()
    a = make_claim(
        claim_id="a", agent_name="x", title="", hypothesis="", weights={"genetics": 1.0},
        holdout="drug", algorithm="ppr", network_params={"damping": 0.85},
    )
    b = make_claim(
        claim_id="b", agent_name="y", title="", hypothesis="", weights={"genetics": 1.0},
        holdout="drug", algorithm="ppr", network_params={"damping": 0.85},
    )
    assert len(governor.select([a, b], 2)) == 1


def test_decide_accepts_consistent_wins() -> None:
    governor = Governor(GovernorConfig(n_bootstrap=200))
    baseline = {"d1": 0.10, "d2": 0.10, "d3": 0.10, "d4": 0.10, "d5": 0.10}
    candidate = {"d1": 0.12, "d2": 0.13, "d3": 0.11, "d4": 0.14, "d5": 0.12}
    decision = governor.decide(baseline, candidate)
    assert decision.accepted
    assert decision.win_rate == 1.0


def test_decide_rejects_no_change() -> None:
    governor = Governor(GovernorConfig(n_bootstrap=200))
    baseline = {"d1": 0.1, "d2": 0.1, "d3": 0.1}
    decision = governor.decide(baseline, dict(baseline))
    assert not decision.accepted
    assert "no criterion met" in decision.reason


def test_decide_rejects_mixed_results() -> None:
    governor = Governor(GovernorConfig(n_bootstrap=200))
    baseline = {"d1": 0.1, "d2": 0.1, "d3": 0.1, "d4": 0.1}
    candidate = {"d1": 0.09, "d2": 0.11, "d3": 0.09, "d4": 0.11}
    assert not governor.decide(baseline, candidate).accepted


# -------------------------------------------------------------- claim config
def test_config_for_claim_applies_intervention() -> None:
    proposal = make_claim(
        claim_id="x", agent_name="a", title="", hypothesis="",
        weights={"genetics": 0.6, "pathway": 0.4}, holdout="drug",
        algorithm="rwr", network_params={"restart_prob": 0.2, "blend": 0.6},
    )
    config = config_for_claim(proposal.claim, RunConfig(), "drug")
    assert config.network.algorithm == "rwr"
    assert config.network.restart_prob == 0.2
    assert config.network.blend == 0.6
    assert config.weights["drug"] == 0.0
    assert config.evaluation.holdout == "drug"


# ------------------------------------------------------------------ LLM bits
def test_parse_claims_accepts_valid_json(context: DebateContext) -> None:
    payload = """
    {"claims": [{
      "title": "t", "hypothesis": "h",
      "evidence_type_weights": {"drug": 0.5, "genetics": 0.5},
      "network_algorithm": "ppr", "network_params": {"damping": 0.8},
      "expected_auprc_gain": 0.01, "expected_recall_gain": 0.02,
      "risk_tags": ["x"], "falsification_test": "f"
    }]}
    """
    proposals = parse_claims(payload, context=context, agent_name="LLM")
    assert len(proposals) == 1
    assert proposals[0].claim.intervention["evidence_type_weights"]["drug"] == 0.0


def test_parse_claims_strips_markdown_fences(context: DebateContext) -> None:
    payload = '```json\n{"claims": []}\n```'
    assert parse_claims(payload, context=context, agent_name="LLM") == []


def test_parse_claims_rejects_bad_shape(context: DebateContext) -> None:
    with pytest.raises(ValueError):
        parse_claims("[]", context=context, agent_name="LLM")


def test_llm_agent_falls_back_on_failure(context: DebateContext) -> None:
    class BrokenClient(LLMClient):
        def __init__(self):  # noqa: D401 - bypass env
            self.api_key = "x"

        def chat(self, system: str, user: str) -> str:
            raise RuntimeError("boom")

    agent = LLMAgent("CG", "persona", client=BrokenClient(), fallback=CausalGeneticsAgent())
    proposals = agent.propose(context)
    assert proposals
    # Fallback proposals keep the heuristic agent's provenance.
    assert proposals[0].claim.agent_name == "CausalGenetics"


# -------------------------------------------------------------------- debate
def test_run_debate_smoke(tmp_path: Path) -> None:
    root = build_toy_cache(tmp_path / "cache")
    report = run_debate(
        [resolve_disease("MONDO_9000001"), resolve_disease("MONDO_9000002")],
        config=RunConfig(),
        cache=Cache(root),
        rounds=1,
        budget=2,
        mode="heuristic",
        holdout="drug",
        offline=True,
    )
    assert report["holdout"] == "drug"
    assert report["baseline"]["auprc"]["mean"] > 0
    assert report["decisions"]
    assert isinstance(report["decisions"][0]["accepted"], bool)
    assert report["leaderboard"]
    assert report["best"] is None or report["best"]["auprc"] >= 0
