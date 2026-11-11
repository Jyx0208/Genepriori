"""GenePriori: leakage-aware gene prioritization.

Most users only need the high-level entry point::

    from genepriori import prioritize
    result = prioritize("breast carcinoma")
    result.to_csv("ranked_genes.csv")

Lower-level building blocks (evidence fusion, network propagation, holdout
evaluation, metrics) are re-exported here for scripting.
"""

from __future__ import annotations

__version__ = "0.2.0"

from .evidence import (
    HOLDOUT_TO_STANDARD,
    OT_TO_STANDARD,
    STANDARD_TYPES,
    FusedScores,
    aggregate_ot_records,
    confidence_from_n_types,
    fuse_gene_scores,
    fuse_gene_scores_detailed,
    map_ot_type,
    min_max_normalize,
    normalize_holdout_type,
    renorm_weights,
)
from .holdout import build_positives, kfold_splits, train_positives
from .leakage import Claim, LeakageAuditor, apply_holdout_mask_to_weights
from .metrics import (
    auprc,
    benjamini_hochberg,
    bootstrap_mean_ci,
    evaluate_ranking,
    paired_bootstrap,
    rank_genes,
    recall_at_k,
    sign_test,
)
from .network import (
    Edge,
    blend_scores,
    build_adjacency,
    degree_penalty,
    ensemble_propagation,
    extract_1hop_subgraph,
    heat_diffusion,
    neighbor_voting,
    personalized_pagerank,
    propagate,
    random_walk_with_restart,
)

__all__ = [
    "__version__",
    # evidence
    "OT_TO_STANDARD",
    "STANDARD_TYPES",
    "HOLDOUT_TO_STANDARD",
    "FusedScores",
    "map_ot_type",
    "normalize_holdout_type",
    "min_max_normalize",
    "renorm_weights",
    "confidence_from_n_types",
    "fuse_gene_scores",
    "fuse_gene_scores_detailed",
    "aggregate_ot_records",
    # holdout
    "build_positives",
    "kfold_splits",
    "train_positives",
    # leakage
    "Claim",
    "LeakageAuditor",
    "apply_holdout_mask_to_weights",
    # metrics
    "rank_genes",
    "recall_at_k",
    "auprc",
    "evaluate_ranking",
    "bootstrap_mean_ci",
    "paired_bootstrap",
    "sign_test",
    "benjamini_hochberg",
    # network
    "Edge",
    "build_adjacency",
    "extract_1hop_subgraph",
    "personalized_pagerank",
    "random_walk_with_restart",
    "neighbor_voting",
    "heat_diffusion",
    "ensemble_propagation",
    "degree_penalty",
    "blend_scores",
    "propagate",
    # high-level
    "prioritize",
]


def prioritize(*args, **kwargs):
    """Rank genes for a disease (requires the data layer).

    Imported lazily so that ``import genepriori`` stays cheap and does not
    require network access.
    """
    from .pipeline.fusion import prioritize as _prioritize

    return _prioritize(*args, **kwargs)


def __getattr__(name: str):
    # Keep the package importable while optional subpackages are absent.
    if name in {"RankedResult", "RankedGene"}:
        from .pipeline import fusion as _fusion

        return getattr(_fusion, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
