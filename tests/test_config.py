"""Tests for run configuration and weight presets."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from genepriori.config import (
    WEIGHT_PRESETS,
    EvaluationConfig,
    LLMConfig,
    NetworkConfig,
    RunConfig,
    resolve_weights,
)
from genepriori.evidence import STANDARD_TYPES


def test_all_presets_cover_six_types_and_sum_to_one() -> None:
    for name, preset in WEIGHT_PRESETS.items():
        assert set(preset) == set(STANDARD_TYPES), name
        assert math.isclose(sum(preset.values()), 1.0), name


def test_resolve_weights_by_preset_name() -> None:
    assert resolve_weights("balanced") == WEIGHT_PRESETS["balanced"]


def test_resolve_weights_rejects_unknown_preset() -> None:
    with pytest.raises(ValueError, match="Unknown weight preset"):
        resolve_weights("turbo")


def test_resolve_weights_rejects_unknown_type() -> None:
    with pytest.raises(ValueError, match="Unknown evidence type"):
        resolve_weights({"mystery": 1.0})


def test_resolve_weights_rejects_negative() -> None:
    with pytest.raises(ValueError, match="Negative weight"):
        resolve_weights({"drug": -1.0, "genetics": 2.0})


def test_resolve_weights_rejects_all_zero() -> None:
    with pytest.raises(ValueError, match="all zero"):
        resolve_weights({"drug": 0.0})


def test_network_config_defaults_match_methods() -> None:
    cfg = NetworkConfig()
    assert cfg.algorithm == "ppr"
    assert cfg.damping == 0.85
    assert cfg.restart_prob == 0.30
    assert cfg.blend == 0.70
    assert cfg.network_discount == 0.25
    assert cfg.max_network_genes == 80


def test_network_config_rejects_bad_algorithm() -> None:
    with pytest.raises(ValueError, match="Unsupported network algorithm"):
        NetworkConfig(algorithm="magic")


def test_network_config_normalizes_algorithm_case() -> None:
    assert NetworkConfig(algorithm="PPR").algorithm == "ppr"
    assert NetworkConfig(algorithm="").algorithm == "none"


def test_network_config_rejects_bad_damping() -> None:
    with pytest.raises(ValueError, match="damping"):
        NetworkConfig(damping=1.5)


def test_evaluation_config_normalizes_holdout() -> None:
    assert EvaluationConfig(holdout="known_drug").holdout == "drug"
    assert EvaluationConfig(holdout="genetic_association").holdout == "genetics"


def test_evaluation_config_rejects_bad_protocol() -> None:
    with pytest.raises(ValueError, match="cv protocol"):
        EvaluationConfig(protocol="leaky")


def test_evaluation_config_rejects_single_fold() -> None:
    with pytest.raises(ValueError, match="folds"):
        EvaluationConfig(folds=1)


def test_run_config_roundtrip_dict() -> None:
    cfg = RunConfig(weights="genetics", max_genes=1500)
    restored = RunConfig.from_dict(cfg.to_dict())
    assert restored.weights == cfg.weights
    assert restored.max_genes == 1500
    assert restored.evaluation.holdout == "drug"


def test_run_config_from_partial_dict() -> None:
    cfg = RunConfig.from_dict({"weights": "translational", "network": {"algorithm": "rwr"}})
    assert cfg.weights == WEIGHT_PRESETS["translational"]
    assert cfg.network.algorithm == "rwr"
    assert cfg.evaluation.folds == 5


def test_run_config_rejects_unknown_top_level_key() -> None:
    with pytest.raises(ValueError, match="Unknown configuration key"):
        RunConfig.from_dict({"weight": "balanced"})


def test_run_config_rejects_unknown_network_key() -> None:
    with pytest.raises(ValueError, match="Unknown network key"):
        RunConfig.from_dict({"network": {"dampening": 0.5}})


def test_run_config_yaml_roundtrip(tmp_path: Path) -> None:
    cfg = RunConfig(weights="network")
    path = cfg.save(tmp_path / "run.yaml")
    restored = RunConfig.load(path)
    assert restored.to_dict() == cfg.to_dict()


def test_run_config_json_roundtrip(tmp_path: Path) -> None:
    cfg = RunConfig(weights="balanced")
    path = cfg.save(tmp_path / "run.json")
    restored = RunConfig.load(path)
    assert restored.to_dict() == cfg.to_dict()


def test_with_weights_and_holdout_return_copies() -> None:
    cfg = RunConfig()
    changed = cfg.with_weights("genetics").with_holdout("genetics")
    assert cfg.weights == WEIGHT_PRESETS["balanced"]
    assert changed.weights == WEIGHT_PRESETS["genetics"]
    assert changed.evaluation.holdout == "genetics"
    assert cfg.evaluation.holdout == "drug"


def test_llm_config_defaults_are_deepseek() -> None:
    cfg = LLMConfig()
    assert cfg.base_url == "https://api.deepseek.com/v1"
    assert cfg.model == "deepseek-chat"
    assert cfg.api_key_env == "DEEPSEEK_API_KEY"


def test_llm_config_rejects_bad_base_url() -> None:
    with pytest.raises(ValueError, match="base_url"):
        LLMConfig(base_url="ftp://example.com")


def test_llm_config_rejects_bad_temperature() -> None:
    with pytest.raises(ValueError, match="temperature"):
        LLMConfig(temperature=3.0)


def test_run_config_llm_roundtrip() -> None:
    cfg = RunConfig(llm=LLMConfig(model="deepseek-reasoner", temperature=0.1))
    restored = RunConfig.from_dict(cfg.to_dict())
    assert restored.llm.model == "deepseek-reasoner"
    assert restored.llm.temperature == 0.1


def test_run_config_rejects_unknown_llm_key() -> None:
    with pytest.raises(ValueError, match="Unknown llm key"):
        RunConfig.from_dict({"llm": {"api_key": "oops"}})


def test_run_config_yaml_carries_llm_block(tmp_path: Path) -> None:
    cfg = RunConfig(llm=LLMConfig(base_url="https://example.com/v1", model="demo"))
    path = cfg.save(tmp_path / "run.yaml")
    text = path.read_text(encoding="utf-8")
    assert "llm:" in text
    assert "api_key_env" in text
    assert "sk-" not in text
    restored = RunConfig.load(path)
    assert restored.llm.model == "demo"
