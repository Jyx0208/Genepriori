"""Deterministic synthetic dataset used by ``genepriori demo``.

The data is artificial but follows the real cache layout: Open Targets-style
evidence files under ``diseases/`` and a STRING-style edge cache under
``ppi/v12.5/``. Literature evidence is deliberately correlated with drug
evidence so that the leakage story is visible even in the demo.
"""

from __future__ import annotations

import gzip
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

TOY_DISEASES = (
    ("MONDO_9000001", "synthetic breast carcinoma"),
    ("MONDO_9000002", "synthetic neurodegenerative disease"),
)
TOY_STRING_VERSION = "12.5"
TOY_MIN_COMBINED_SCORE = 400
N_GENES = 260
STANDARD_TYPES = ("drug", "genetics", "expression", "pathway", "literature", "animal_model")


def _genes() -> List[str]:
    return [f"GENE{i:03d}" for i in range(1, N_GENES + 1)]


def _build_network(rng: random.Random, genes: List[str]) -> List[Tuple[str, str, float]]:
    edges: Dict[Tuple[str, str], float] = {}
    for i, gene in enumerate(genes):
        for offset in (1, 7, 31):
            partner = genes[(i + offset) % len(genes)]
            a, b = sorted((gene, partner))
            weight = round(0.45 + 0.5 * rng.random(), 3)
            edges[(a, b)] = max(edges.get((a, b), 0.0), weight)
    return [(a, b, w) for (a, b), w in sorted(edges.items())]


def _build_disease(
    rng: random.Random,
    genes: List[str],
    disease_id: str,
    name: str,
    emphasis: Dict[str, float],
) -> Dict:
    gene_type_scores: Dict[str, Dict[str, float]] = {}
    ensembl: Dict[str, str] = {}
    # Genes are grouped into modules that share a latent priority, and the
    # synthetic network is denser within modules. This gives network
    # propagation real signal to work with, as in a real PPI network.
    module_size = 10
    module_priority = [rng.random() for _ in range((len(genes) + module_size - 1) // module_size)]
    for i, gene in enumerate(genes):
        # Real Open Targets releases only annotate a subset of genes; leaving
        # some genes without direct evidence is what lets network propagation
        # discover new candidates.
        if i % 9 in (0, 4, 7):
            continue
        latent = module_priority[i // module_size] * (rng.random() ** 1.5)
        if i % 5 == 0:
            latent = min(1.0, latent + 0.10)
        scores: Dict[str, float] = {}
        for evidence_type in STANDARD_TYPES:
            base = emphasis.get(evidence_type, 0.2)
            value = base * latent + 0.12 * rng.random()
            if evidence_type == "literature":
                value = 0.65 * scores.get("drug", 0.0) + 0.35 * rng.random()
            scores[evidence_type] = round(min(1.0, max(0.0, value)), 4)
        gene_type_scores[gene] = scores
        ensembl[gene] = f"ENSG{i:011d}"
    return {
        "schema_version": 1,
        "disease_id": disease_id,
        "disease_name": name,
        "gene_type_scores": gene_type_scores,
        "ensembl_ids": ensembl,
        "n_rows": len(gene_type_scores),
        "total_associations": len(gene_type_scores),
        "fetched_at": "2026-01-01T00:00:00+00:00",
        "source": "synthetic demo data (not Open Targets)",
    }


def _write_edges(path: Path, edges: List[Tuple[str, str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8", compresslevel=9) as handle:
        handle.write("gene1\tgene2\tweight\n")
        for a, b, w in edges:
            handle.write(f"{a}\t{b}\t{w:.4f}\n")


def build_toy_cache(root: Path | str, *, force: bool = True) -> Path:
    """Write the synthetic cache under ``root`` and return the path."""
    root = Path(root)
    marker = root / "diseases" / f"{TOY_DISEASES[0][0]}.json"
    if marker.exists() and not force:
        return root

    rng = random.Random(7)
    genes = _genes()
    edges = _build_network(rng, genes)
    _write_edges(
        root / "ppi" / f"v{TOY_STRING_VERSION}" / f"9606.edges.min{TOY_MIN_COMBINED_SCORE}.tsv.gz",
        edges,
    )

    emphases = (
        {"drug": 0.75, "genetics": 0.55, "expression": 0.30, "pathway": 0.35,
         "literature": 0.60, "animal_model": 0.25},
        {"drug": 0.55, "genetics": 0.85, "expression": 0.25, "pathway": 0.45,
         "literature": 0.40, "animal_model": 0.55},
    )
    manifest: Dict[str, Dict] = {}
    for (disease_id, name), emphasis in zip(TOY_DISEASES, emphases):
        payload = _build_disease(rng, genes, disease_id, name, emphasis)
        key = f"diseases/{disease_id}.json"
        target = root / key
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        manifest[key] = {
            "path": key,
            "fetched_at": payload["fetched_at"],
            "url": None,
            "source": payload["source"],
        }

    edge_key = f"ppi/v{TOY_STRING_VERSION}/9606.edges.min{TOY_MIN_COMBINED_SCORE}.tsv.gz"
    manifest[edge_key] = {
        "path": edge_key,
        "fetched_at": "2026-01-01T00:00:00+00:00",
        "url": None,
        "source": "synthetic demo data (not STRING)",
        "n_edges": len(edges),
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return root
