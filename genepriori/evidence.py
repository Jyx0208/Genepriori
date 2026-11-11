"""Evidence-type mapping and weighted fusion (Methods)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Optional

# Manuscript Table 3: Open Targets fine-grained types -> six standardized types.
# When multiple fine-grained scores exist for the same gene and category, keep max.
# ``clinical`` is the current name of the drug/clinical datatype (the older
# ``known_drug`` label is kept as an alias for archived responses).
OT_TO_STANDARD: Dict[str, str] = {
    "known_drug": "drug",
    "drugs": "drug",
    "clinical": "drug",
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


@dataclass(frozen=True)
class FusedScores:
    """Detailed output of :func:`fuse_gene_scores_detailed`.

    ``scores``: gene -> fused score.
    ``confidence``: gene -> confidence in [0.35, 1.0] from the number of
    non-zero contributing evidence types.
    ``n_types``: gene -> number of contributing evidence types.
    ``contributions``: gene -> {standard type -> weighted contribution}, i.e.
    ``weight * min_max_normalized_score``. Values sum to ``scores[gene]``.
    """

    scores: Dict[str, float]
    confidence: Dict[str, float]
    n_types: Dict[str, int]
    contributions: Dict[str, Dict[str, float]]


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
    """Min-max scale to [0, 1] over the provided genes.

    A type whose scores are constant (including a single-gene type) has no
    discriminative power and normalizes to all zeros.
    """
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


def fuse_gene_scores_detailed(
    gene_type_scores: Mapping[str, Mapping[str, float]],
    weights: Mapping[str, float],
    *,
    holdout_type: Optional[str] = None,
) -> FusedScores:
    """Weighted sum of min-max-normalized per-type scores, with details.

    ``gene_type_scores``: gene -> {standard_type -> raw score}.
    Per-type scores are min-max normalized across the provided genes before
    weighting, matching the Methods ("Scores are min-max normalized before
    aggregation"). Unknown evidence types are ignored.
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
            type_to_gene[t][gene] = float(s) if prev is None else max(prev, float(s))

    type_norm = {t: min_max_normalize(m) for t, m in type_to_gene.items()}

    scores: Dict[str, float] = {}
    confidence: Dict[str, float] = {}
    n_types: Dict[str, int] = {}
    contributions: Dict[str, Dict[str, float]] = {}
    for g in sorted(gene_type_scores):
        score = 0.0
        count = 0
        breakdown: Dict[str, float] = {}
        for t, wt in w.items():
            if wt <= 0:
                continue
            v = type_norm[t].get(g)
            if v is None:
                continue
            contribution = wt * v
            score += contribution
            if contribution > 0:
                breakdown[t] = contribution
                count += 1
        scores[g] = float(score)
        n_types[g] = count
        confidence[g] = confidence_from_n_types(count)
        contributions[g] = breakdown

    return FusedScores(
        scores=scores,
        confidence=confidence,
        n_types=n_types,
        contributions=contributions,
    )


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
    Use :func:`fuse_gene_scores_detailed` when per-gene confidence or the
    evidence breakdown is needed.
    """
    return fuse_gene_scores_detailed(
        gene_type_scores,
        weights,
        holdout_type=holdout_type,
    ).scores


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
