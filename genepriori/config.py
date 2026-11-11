"""Run configuration: evidence weights, network parameters, evaluation settings.

A configuration can be built in Python, loaded from YAML/JSON, or assembled
from named weight presets. It is intentionally small so that a run can be
described in a single file and recorded verbatim in the run manifest.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Union

import yaml

from .evidence import STANDARD_TYPES, normalize_holdout_type

PathLike = Union[str, "Path"]


def _uniform(types) -> Dict[str, float]:
    selected = list(types)
    share = 1.0 / len(selected)
    return {t: (share if t in selected else 0.0) for t in STANDARD_TYPES}


WEIGHT_PRESETS: Dict[str, Dict[str, float]] = {
    "balanced": _uniform(STANDARD_TYPES),
    "genetics": {
        "drug": 0.05,
        "genetics": 0.50,
        "expression": 0.10,
        "pathway": 0.15,
        "literature": 0.15,
        "animal_model": 0.05,
    },
    "translational": {
        "drug": 0.30,
        "genetics": 0.30,
        "expression": 0.05,
        "pathway": 0.10,
        "literature": 0.20,
        "animal_model": 0.05,
    },
    "network": {
        "drug": 0.10,
        "genetics": 0.15,
        "expression": 0.15,
        "pathway": 0.25,
        "literature": 0.20,
        "animal_model": 0.15,
    },
    "no_literature": _uniform([t for t in STANDARD_TYPES if t != "literature"]),
    "literature_only": {t: (1.0 if t == "literature" else 0.0) for t in STANDARD_TYPES},
}

DEFAULT_WEIGHTS = dict(WEIGHT_PRESETS["balanced"])


def resolve_weights(weights: Union[str, Mapping[str, float], None]) -> Dict[str, float]:
    """Accept a preset name, an explicit weight mapping, or ``None``."""
    if weights is None:
        return dict(DEFAULT_WEIGHTS)
    if isinstance(weights, str):
        key = weights.strip().lower()
        if key not in WEIGHT_PRESETS:
            raise ValueError(
                f"Unknown weight preset {weights!r}. "
                f"Expected one of {sorted(WEIGHT_PRESETS)} or an explicit mapping."
            )
        return dict(WEIGHT_PRESETS[key])
    cleaned: Dict[str, float] = {t: 0.0 for t in STANDARD_TYPES}
    for key, value in weights.items():
        if key not in STANDARD_TYPES:
            raise ValueError(
                f"Unknown evidence type {key!r} in weights. "
                f"Expected a subset of {list(STANDARD_TYPES)}."
            )
        value = float(value)
        if value < 0:
            raise ValueError(f"Negative weight for {key!r}: {value}.")
        cleaned[key] = value
    if sum(cleaned.values()) <= 0:
        raise ValueError("Weights must not be all zero.")
    return cleaned


@dataclass
class NetworkConfig:
    """Network propagation parameters (Methods defaults)."""

    algorithm: str = "ppr"
    damping: float = 0.85
    restart_prob: float = 0.30
    max_neighbors_per_seed: int = 30
    min_score: float = 0.40
    blend: float = 0.70
    network_discount: float = 0.25
    max_network_genes: int = 80
    heat_steps: int = 10
    heat_step_size: float = 0.25

    def __post_init__(self) -> None:
        from .network import SUPPORTED_ALGORITHMS

        algo = (self.algorithm or "").strip().lower()
        if algo not in SUPPORTED_ALGORITHMS:
            raise ValueError(
                f"Unsupported network algorithm {self.algorithm!r}. "
                f"Expected one of {sorted(SUPPORTED_ALGORITHMS - {''})} or 'none'."
            )
        self.algorithm = algo or "none"
        if not 0.0 < self.damping < 1.0:
            raise ValueError(f"damping must be in (0, 1), got {self.damping}.")
        if not 0.0 < self.restart_prob < 1.0:
            raise ValueError(f"restart_prob must be in (0, 1), got {self.restart_prob}.")
        if not 0.0 <= self.blend <= 1.0:
            raise ValueError(f"blend must be in [0, 1], got {self.blend}.")
        if self.max_network_genes < 0:
            raise ValueError("max_network_genes must be non-negative.")
        if self.max_neighbors_per_seed < 0:
            raise ValueError("max_neighbors_per_seed must be non-negative.")


@dataclass
class EvaluationConfig:
    """Dual-holdout cross-validation settings."""

    holdout: str = "drug"
    folds: int = 5
    seed: int = 42
    positive_threshold: float = 0.30
    k: int = 20
    n_bootstrap: int = 2000
    protocol: str = "mask"

    def __post_init__(self) -> None:
        self.holdout = normalize_holdout_type(self.holdout)
        if self.folds < 2:
            raise ValueError(f"folds must be >= 2, got {self.folds}.")
        if self.k <= 0:
            raise ValueError(f"k must be positive, got {self.k}.")
        if self.n_bootstrap <= 0:
            raise ValueError(f"n_bootstrap must be positive, got {self.n_bootstrap}.")
        protocol = (self.protocol or "").strip().lower()
        if protocol not in {"mask", "restore"}:
            raise ValueError(
                f"Unsupported cv protocol {self.protocol!r}; expected 'mask' or 'restore'."
            )
        self.protocol = protocol
        if not 0.0 <= self.positive_threshold < 1.0:
            raise ValueError(
                f"positive_threshold must be in [0, 1), got {self.positive_threshold}."
            )


@dataclass
class LLMConfig:
    """OpenAI-compatible chat endpoint used by the debate agents.

    The API key itself is never stored here: ``api_key_env`` names the
    environment variable to read (after ``GENEPRIORI_LLM_API_KEY`` and the
    provider-specific fallbacks).
    """

    base_url: str = "https://api.deepseek.com/v1"
    model: str = "deepseek-chat"
    api_key_env: str = "DEEPSEEK_API_KEY"
    temperature: float = 0.2
    timeout: float = 120.0
    max_retries: int = 2

    def __post_init__(self) -> None:
        self.base_url = (self.base_url or "").rstrip("/")
        if not self.base_url.startswith(("http://", "https://")):
            raise ValueError(
                f"llm.base_url must start with http:// or https://, got {self.base_url!r}."
            )
        self.model = (self.model or "").strip()
        if not self.model:
            raise ValueError("llm.model must not be empty.")
        if not 0.0 <= float(self.temperature) <= 2.0:
            raise ValueError(f"llm.temperature must be in [0, 2], got {self.temperature}.")
        if float(self.timeout) <= 0:
            raise ValueError(f"llm.timeout must be positive, got {self.timeout}.")
        if int(self.max_retries) < 0:
            raise ValueError("llm.max_retries must be non-negative.")


@dataclass
class RunConfig:
    """Everything needed to reproduce a ranking run."""

    weights: Dict[str, float] = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))
    network: NetworkConfig = field(default_factory=NetworkConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    max_genes: int = 2000
    page_size: int = 500
    string_version: str = "12.5"
    min_combined_score: int = 400
    cache_dir: Optional[str] = None
    ot_max_age_days: Optional[float] = None

    def __post_init__(self) -> None:
        self.weights = resolve_weights(self.weights)
        if self.max_genes <= 0:
            raise ValueError(f"max_genes must be positive, got {self.max_genes}.")
        if self.page_size <= 0:
            raise ValueError(f"page_size must be positive, got {self.page_size}.")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save(self, path: PathLike) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.suffix.lower() in {".yaml", ".yml"}:
            target.write_text(yaml.safe_dump(self.to_dict(), sort_keys=False), encoding="utf-8")
        else:
            target.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return target

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RunConfig":
        data = dict(payload or {})
        network_raw = data.pop("network", {}) or {}
        evaluation_raw = data.pop("evaluation", {}) or {}
        llm_raw = data.pop("llm", {}) or {}
        weights_raw = data.pop("weights", None)

        network = _build(NetworkConfig, network_raw, "network")
        evaluation = _build(EvaluationConfig, evaluation_raw, "evaluation")
        llm = _build(LLMConfig, llm_raw, "llm")
        allowed = {f.name for f in fields(cls)} - {"weights", "network", "evaluation", "llm"}
        unknown = set(data) - allowed
        if unknown:
            raise ValueError(
                f"Unknown configuration key(s): {sorted(unknown)}. "
                f"Expected a subset of {sorted(allowed)} plus weights/network/evaluation/llm."
            )
        return cls(
            weights=weights_raw if weights_raw is not None else dict(DEFAULT_WEIGHTS),
            network=network,
            evaluation=evaluation,
            llm=llm,
            **data,
        )

    @classmethod
    def load(cls, path: PathLike) -> "RunConfig":
        source = Path(path)
        text = source.read_text(encoding="utf-8")
        payload = yaml.safe_load(text) if source.suffix.lower() in {".yaml", ".yml"} else json.loads(text)
        if not isinstance(payload, Mapping):
            raise ValueError(f"{source} must contain a mapping at the top level.")
        return cls.from_dict(payload)

    def with_weights(self, weights: Union[str, Mapping[str, float]]) -> "RunConfig":
        clone = RunConfig.from_dict(self.to_dict())
        clone.weights = resolve_weights(weights)
        return clone

    def with_holdout(self, holdout: str) -> "RunConfig":
        clone = RunConfig.from_dict(self.to_dict())
        clone.evaluation.holdout = normalize_holdout_type(holdout)
        return clone


def _build(cls, payload: Mapping[str, Any], label: str):
    if not isinstance(payload, Mapping):
        raise ValueError(f"{label} must be a mapping, got {type(payload).__name__}.")
    allowed = {f.name for f in fields(cls)}
    unknown = set(payload) - allowed
    if unknown:
        raise ValueError(
            f"Unknown {label} key(s): {sorted(unknown)}. Expected a subset of {sorted(allowed)}."
        )
    return cls(**dict(payload))
