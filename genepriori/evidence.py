"""Evidence-type mapping and weighted fusion (Methods)."""

from __future__ import annotations

from typing import Dict, Iterable, Mapping, MutableMapping, Optional

# Manuscript Table 3: Open Targets fine-grained types -> six standardized types.
# When multiple fine-grained scores exist for the same gene and category, keep max.
OT_TO_STANDARD: Dict[str, str] = {
    "known_drug": "drug",
    "drugs": "drug",
    "genetic_association": "genetics",
    "somatic_mutation": "genetics",
    "genetic_literature": "genetics",
    "rna_expression": "expression",
    "gene_expression": "expression",
    "proteomics": "expression",
    "affected_pathway": "pathway",
    "pathway": "pathway",
    "literature": "literature",
    "animal_model": "animal_model",
}

STANDARD_TYPES = (
    "drug",
    "genetics",
    "expression",
    "pathway",
    "literature",
    "animal_model",
)

# Holdout protocol names used in the manuscript.
HOLDOUT_TO_STANDARD = {
    "drug": "drug",
    "drug_target": "drug",
    "known_drug": "drug",
    "genetics": "genetics",
    "genetic": "genetics",
    "genetic_association": "genetics",
}


def map_ot_type(ot_type: str) -> Optional[str]:
    """Map an Open Targets evidence type string to a GenePriori standard type."""
    key = (ot_type or "").strip().lower()
    return OT_TO_STANDARD.get(key)


def normalize_holdout_type(holdout_type: str) -> str:
    key = (holdout_type or "").strip().lower()
    if key not in HOLDOUT_TO_STANDARD:
        raise ValueError(
            f"Unsupported holdout_type={holdout_type!r}. "
            f"Expected one of {sorted(set(HOLDOUT_TO_STANDARD))}."
        )
    return HOLDOUT_TO_STANDARD[key]


def min_max_normalize(scores: Mapping[str, float]) -> Dict[str, float]:
    if not scores:
        return {}
    vals = list(scores.values())
    lo, hi = min(vals), max(vals)
    if hi <= lo:
        return {g: 0.0 for g in scores}
    return {g: (v - lo) / (hi - lo) for g, v in scores.items()}


def renorm_weights(
    weights: Mapping[str, float],
    *,
    holdout_type: Optional[str] = None,
) -> Dict[str, float]:
    """Non-negative weights summing to 1; holdout type forced to 0 then renorm.

    Safety: negative weights are clipped to 0. If all remaining mass is 0,
    fall back to uniform weights over non-holdout standard types.
    """
    w: Dict[str, float] = {t: max(0.0, float(weights.get(t, 0.0))) for t in STANDARD_TYPES}
    masked = None
    if holdout_type is not None:
        masked = normalize_holdout_type(holdout_type)
        w[masked] = 0.0
    total = sum(w.values())
    if total <= 0:
        remain = [t for t in STANDARD_TYPES if t != masked]
        if not remain:
            remain = list(STANDARD_TYPES)
        u = 1.0 / len(remain)
        return {t: (u if t in remain else 0.0) for t in STANDARD_TYPES}
    return {t: w[t] / total for t in STANDARD_TYPES}


def confidence_from_n_types(n_types: int) -> float:
    """Manuscript: confidence(g) = min(1.0, 0.35 + 0.12 * n_types(g))."""
    return min(1.0, 0.35 + 0.12 * float(n_types))


def fuse_gene_scores(
    gene_type_scores: Mapping[str, Mapping[str, float]],
    weights: Mapping[str, float],
    *,
    holdout_type: Optional[str] = None,
) -> Dict[str, float]:
    """Weighted sum of min-max-normalized per-type scores.

    ``gene_type_scores``: gene -> {standard_type -> raw score}.
    Returns a score dict only; ranking order is determined by callers
    (``metrics.rank_genes`` breaks ties deterministically by gene symbol).
    """
    w = renorm_weights(weights, holdout_type=holdout_type)

    # Collect per-type raw maps then normalize within type.
    type_to_gene: Dict[str, Dict[str, float]] = {t: {} for t in STANDARD_TYPES}
    for gene, tmap in gene_type_scores.items():
        for t, s in tmap.items():
            if t not in type_to_gene:
                continue
            prev = type_to_gene[t].get(gene)
            # max if multiple contributions
            type_to_gene[t][gene] = s if prev is None else max(prev, s)

    type_norm = {t: min_max_normalize(m) for t, m in type_to_gene.items()}

    genes = set(gene_type_scores)
    out: Dict[str, float] = {}
    for g in genes:
        score = 0.0
        n_types = 0
        for t, wt in w.items():
            if wt <= 0:
                continue
            v = type_norm[t].get(g)
            if v is None:
                continue
            score += wt * v
            if v > 0:
                n_types += 1
        # confidence is available to callers via confidence_from_n_types if needed
        _ = confidence_from_n_types(n_types)
        out[g] = float(score)

    return out


def aggregate_ot_records(
    records: Iterable[Mapping[str, object]],
) -> Dict[str, Dict[str, float]]:
    """Aggregate OT-like records into gene -> standard_type -> max score.

    Each record needs keys: gene, evidence_type (OT or standard), score.
    """
    by_gene: Dict[str, Dict[str, float]] = {}
    for rec in records:
        gene = str(rec["gene"])
        raw_type = str(rec["evidence_type"])
        score = float(rec["score"])
        std = map_ot_type(raw_type) or (raw_type if raw_type in STANDARD_TYPES else None)
        if std is None:
            continue
        slot = by_gene.setdefault(gene, {})
        slot[std] = score if std not in slot else max(slot[std], score)
    return by_gene
