"""Ranking pipeline: evidence -> fusion -> optional propagation -> ranked list.

This is the user-facing entry point behind ``genepriori rank`` and
``genepriori.prioritize``. It deliberately contains no evaluation logic: it
produces the ranking a researcher would act on, using every available
evidence type (no holdout masking). Leakage-controlled measurement lives in
:mod:`genepriori.pipeline.evaluation`.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

from ..config import RunConfig
from ..data.cache import Cache
from ..data.opentargets import DiseaseEvidence, evidence_summary, fetch_disease_evidence
from ..data.registry import Disease, resolve_disease
from ..data.string import load_string_edges, ppi_summary
from ..evidence import STANDARD_TYPES, fuse_gene_scores_detailed
from ..metrics import rank_genes
from ..network import Edge, propagate

PathLike = Union[str, Path]


@dataclass
class RankedGene:
    """One row of a prioritization result."""

    rank: int
    gene: str
    score: float
    confidence: float
    n_evidence_types: int
    evidence: Dict[str, float] = field(default_factory=dict)
    network_discovered: bool = False
    ensembl_id: str = ""

    def to_row(self) -> Dict[str, Any]:
        row: Dict[str, Any] = {
            "rank": self.rank,
            "gene": self.gene,
            "score": round(self.score, 8),
            "confidence": round(self.confidence, 4),
            "n_evidence_types": self.n_evidence_types,
            "network_discovered": int(self.network_discovered),
            "ensembl_id": self.ensembl_id,
        }
        for evidence_type in STANDARD_TYPES:
            row[evidence_type] = round(self.evidence.get(evidence_type, 0.0), 8)
        return row


@dataclass
class RankedResult:
    """Ranked genes plus the metadata needed to reproduce them."""

    disease_id: str
    disease_name: str
    ranked: List[RankedGene]
    config: Dict[str, Any] = field(default_factory=dict)
    evidence_summary: Dict[str, Any] = field(default_factory=dict)
    network_summary: Dict[str, Any] = field(default_factory=dict)

    def top(self, n: int = 20) -> List[RankedGene]:
        return self.ranked[: max(0, int(n))]

    def scores(self) -> Dict[str, float]:
        return {row.gene: row.score for row in self.ranked}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "disease_id": self.disease_id,
            "disease_name": self.disease_name,
            "n_genes": len(self.ranked),
            "config": self.config,
            "evidence_summary": self.evidence_summary,
            "network_summary": self.network_summary,
            "ranked": [row.to_row() for row in self.ranked],
        }

    def to_csv(self, path: PathLike) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        columns = list(self.ranked[0].to_row()) if self.ranked else [
            "rank",
            "gene",
            "score",
            "confidence",
            "n_evidence_types",
            "network_discovered",
            "ensembl_id",
            *STANDARD_TYPES,
        ]
        with open(target, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            for row in self.ranked:
                writer.writerow(row.to_row())
        return target

    def to_json(self, path: PathLike) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return target


def build_ranking(
    evidence: DiseaseEvidence,
    *,
    config: Optional[RunConfig] = None,
    edges: Optional[Sequence[Edge]] = None,
) -> RankedResult:
    """Rank genes for an already-loaded :class:`DiseaseEvidence`."""
    config = config or RunConfig()
    detail = fuse_gene_scores_detailed(evidence.gene_type_scores, config.weights)

    direct = detail.scores
    final: Dict[str, float] = dict(direct)
    network_stats: Dict[str, Any] = {}
    network_cfg = config.network

    if network_cfg.algorithm != "none" and edges:
        final = propagate(
            direct,
            edges,
            algorithm=network_cfg.algorithm,
            damping=network_cfg.damping,
            restart_prob=network_cfg.restart_prob,
            max_neighbors_per_seed=network_cfg.max_neighbors_per_seed,
            min_score=network_cfg.min_score,
            blend=network_cfg.blend,
            network_discount=network_cfg.network_discount,
            max_network_genes=network_cfg.max_network_genes,
            heat_steps=network_cfg.heat_steps,
            heat_step_size=network_cfg.heat_step_size,
        )
        network_stats = ppi_summary(list(edges))
        network_stats["algorithm"] = network_cfg.algorithm

    ordered = rank_genes(final)
    rows: List[RankedGene] = []
    for position, gene in enumerate(ordered, start=1):
        direct_score = float(direct.get(gene, 0.0))
        rows.append(
            RankedGene(
                rank=position,
                gene=gene,
                score=float(final[gene]),
                confidence=float(detail.confidence.get(gene, 0.35)),
                n_evidence_types=int(detail.n_types.get(gene, 0)),
                evidence=dict(detail.contributions.get(gene, {})),
                network_discovered=direct_score <= 0.0,
                ensembl_id=evidence.ensembl_ids.get(gene, ""),
            )
        )

    return RankedResult(
        disease_id=evidence.disease_id,
        disease_name=evidence.disease_name,
        ranked=rows,
        config={
            "weights": dict(config.weights),
            "network": {
                "algorithm": network_cfg.algorithm,
                "damping": network_cfg.damping,
                "restart_prob": network_cfg.restart_prob,
                "blend": network_cfg.blend,
                "network_discount": network_cfg.network_discount,
                "max_network_genes": network_cfg.max_network_genes,
                "max_neighbors_per_seed": network_cfg.max_neighbors_per_seed,
                "min_score": network_cfg.min_score,
            },
            "max_genes": config.max_genes,
            "string_version": config.string_version,
            "min_combined_score": config.min_combined_score,
        },
        evidence_summary=evidence_summary(evidence),
        network_summary=network_stats,
    )


def load_inputs(
    disease: Union[str, Disease],
    *,
    config: Optional[RunConfig] = None,
    cache: Optional[Cache] = None,
    cache_dir: Optional[PathLike] = None,
    client=None,
    evidence: Optional[DiseaseEvidence] = None,
    edges: Optional[Sequence[Edge]] = None,
    offline: bool = False,
    refresh: bool = False,
    max_age_days: Optional[float] = None,
) -> tuple[Disease, RunConfig, DiseaseEvidence, Optional[Sequence[Edge]]]:
    """Resolve a disease and load its evidence plus (optionally) the PPI network."""
    base = config or RunConfig()
    if cache is None:
        cache = Cache(cache_dir if cache_dir is not None else base.cache_dir)
    resolved = disease if isinstance(disease, Disease) else resolve_disease(disease)

    if evidence is None:
        evidence = fetch_disease_evidence(
            resolved,
            client=client,
            cache=cache,
            max_genes=base.max_genes,
            page_size=base.page_size,
            refresh=refresh,
            offline=offline,
            max_age_days=max_age_days if max_age_days is not None else base.ot_max_age_days,
        )
    if resolved.name and resolved.name != resolved.disease_id and not evidence.disease_name:
        evidence.disease_name = resolved.name

    if edges is None and base.network.algorithm != "none":
        edges = load_string_edges(
            version=base.string_version,
            min_combined_score=base.min_combined_score,
            cache=cache,
            refresh=refresh,
            offline=offline,
        )
    return resolved, base, evidence, edges


def prioritize(
    disease: Union[str, Disease],
    *,
    config: Optional[RunConfig] = None,
    weights: Optional[Union[str, Mapping[str, float]]] = None,
    network: Optional[str] = None,
    cache: Optional[Cache] = None,
    cache_dir: Optional[PathLike] = None,
    client=None,
    evidence: Optional[DiseaseEvidence] = None,
    edges: Optional[Sequence[Edge]] = None,
    offline: bool = False,
    refresh: bool = False,
    max_age_days: Optional[float] = None,
) -> RankedResult:
    """Rank genes for one disease.

    ``disease`` accepts a registry key/name (see ``genepriori list-diseases``)
    or any EFO/MONDO identifier. Weights and network settings come from
    ``config`` unless overridden by ``weights`` / ``network``.
    """
    base = config or RunConfig()
    if weights is not None:
        base = base.with_weights(weights)
    if network is not None:
        payload = base.to_dict()
        payload["network"]["algorithm"] = network.strip().lower() or "none"
        base = RunConfig.from_dict(payload)

    _, base, evidence, edges = load_inputs(
        disease,
        config=base,
        cache=cache,
        cache_dir=cache_dir,
        client=client,
        evidence=evidence,
        edges=edges,
        offline=offline,
        refresh=refresh,
        max_age_days=max_age_days,
    )
    return build_ranking(evidence, config=base, edges=edges)
