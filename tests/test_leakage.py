"""Tests for the LeakageAuditor policy checks."""

from __future__ import annotations

import math

import pytest

from genepriori.leakage import (
    SUPPORTED_NETWORK_ALGORITHMS,
    Claim,
    LeakageAuditor,
    apply_holdout_mask_to_weights,
)


def _claim(**intervention) -> Claim:
    return Claim(claim_id="c1", intervention=intervention)


@pytest.fixture()
def auditor() -> LeakageAuditor:
    return LeakageAuditor()


def test_clean_claim_passes(auditor: LeakageAuditor) -> None:
    claim = _claim(
        evidence_type_weights={"drug": 0.0, "genetics": 0.5, "literature": 0.5},
        network_algorithm="ppr",
    )
    ok, reason = auditor.audit(claim, "drug")
    assert ok, reason
    assert reason == "passed"


def test_claim_without_intervention_passes(auditor: LeakageAuditor) -> None:
    ok, _ = auditor.audit(Claim(claim_id="c1"), "genetics")
    assert ok


def test_positive_holdout_weight_rejected(auditor: LeakageAuditor) -> None:
    ok, reason = auditor.audit(_claim(evidence_type_weights={"drug": 0.3}), "drug")
    assert not ok
    assert "non-zero weight" in reason


def test_negative_holdout_weight_rejected(auditor: LeakageAuditor) -> None:
    ok, reason = auditor.audit(_claim(evidence_type_weights={"drug": -0.3}), "drug")
    assert not ok
    assert "non-zero weight" in reason


def test_allow_flag_rejected(auditor: LeakageAuditor) -> None:
    ok, reason = auditor.audit(_claim(allow_holdout_evidence=True), "drug")
    assert not ok
    assert "enables holdout evidence" in reason


def test_unsupported_algorithm_rejected(auditor: LeakageAuditor) -> None:
    ok, reason = auditor.audit(_claim(network_algorithm="magic"), "drug")
    assert not ok
    assert "unsupported network algorithm" in reason


def test_supported_algorithms_accepted(auditor: LeakageAuditor) -> None:
    for algo in SUPPORTED_NETWORK_ALGORITHMS:
        ok, reason = auditor.audit(_claim(network_algorithm=algo), "drug")
        assert ok, f"{algo}: {reason}"


def test_disease_specific_holdout_weight_rejected(auditor: LeakageAuditor) -> None:
    claim = _claim(disease_specific_weights={"EFO_1": {"drug": 0.2}})
    ok, reason = auditor.audit(claim, "drug")
    assert not ok
    assert "disease-specific weight" in reason


def test_disease_specific_weights_must_be_dict(auditor: LeakageAuditor) -> None:
    ok, reason = auditor.audit(_claim(disease_specific_weights=("EFO_1",)), "drug")
    assert not ok
    assert "must be a dict" in reason


def test_empty_disease_specific_weights_is_allowed(auditor: LeakageAuditor) -> None:
    ok, reason = auditor.audit(_claim(disease_specific_weights={}), "drug")
    assert ok, reason


def test_disease_specific_entry_must_be_dict(auditor: LeakageAuditor) -> None:
    ok, reason = auditor.audit(_claim(disease_specific_weights={"EFO_1": 0.2}), "drug")
    assert not ok
    assert "must be a dict" in reason


def test_disease_specific_algorithm_inherits_global(auditor: LeakageAuditor) -> None:
    claim = _claim(network_algorithm="rwr", disease_specific_network={"EFO_1": {}})
    ok, reason = auditor.audit(claim, "drug")
    assert ok, reason


def test_disease_specific_bad_algorithm_rejected(auditor: LeakageAuditor) -> None:
    claim = _claim(
        network_algorithm="ppr",
        disease_specific_network={"EFO_1": {"network_algorithm": "magic"}},
    )
    ok, reason = auditor.audit(claim, "drug")
    assert not ok
    assert "disease-specific network algorithm" in reason


def test_audit_normalizes_holdout_alias(auditor: LeakageAuditor) -> None:
    claim = _claim(evidence_type_weights={"drug": 0.4})
    ok, _ = auditor.audit(claim, "known_drug")
    assert not ok


def test_audit_rejects_unknown_holdout(auditor: LeakageAuditor) -> None:
    with pytest.raises(ValueError):
        auditor.audit(Claim(claim_id="c1"), "expression")


def test_apply_holdout_mask_zeroes_and_renormalizes() -> None:
    masked = apply_holdout_mask_to_weights({"drug": 0.6, "genetics": 0.4}, "drug")
    assert masked["drug"] == 0.0
    assert math.isclose(masked["genetics"], 1.0)
    assert math.isclose(sum(masked.values()), 1.0)
