# GenePriori

Source code for:

> **GenePriori: a leakage-aware multi-agent framework for bias-robust gene prioritization with translational oncology relevance**

This repository provides **reference implementations** of the core components
described in the manuscript Methods:

1. Evidence-type mapping and weighted fusion (`genepriori/evidence.py`)
2. Dual-holdout positive-set construction and fold splitting (`genepriori/holdout.py`)
3. LeakageAuditor policy checks (`genepriori/leakage.py`)
4. Ranking metrics: AUPRC, Recall@K, bootstrap CI (`genepriori/metrics.py`)
5. RWR / PPR network propagation with degree penalty (`genepriori/network.py`)

It does **not** ship a full end-to-end multi-agent production system, raw
third-party bulk downloads, API keys, or AI-generated manuscript figures.

## What this code is / is not

| Included | Not included |
|---|---|
| Evidence-type mapping (manuscript Table 3) | Full Open Targets GraphQL client cache dumps |
| Holdout weight masking + renormalization | Guaranteed bit-for-bit regeneration of every paper figure |
| RWR / PPR + degree penalty (Methods) | Large intermediate run artifacts |
| LeakageAuditor **policy checks** (claim-level) | Claims of "leakage-free" evaluation |
| AUPRC, Recall@K, bootstrap helpers | Live LLM debate logs / prompt packages by default |
|  | Unrelated prototype agent stacks |

**Important evaluation note:**
Positive genes remain visible through *non-holdout* evidence types.
"Leakage-aware" means the **held-out evidence type is masked** and **claims are
audited against direct use of that type**, not that all semantic collinearity
among remaining evidence is removed.

## Installation

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Requires Python 3.9+ with `numpy` and `scipy`.

## Usage example (synthetic data)

```python
from genepriori.evidence import fuse_gene_scores
from genepriori.network import propagate

# Synthetic scores: gene -> {evidence_type -> score}
scores = {
    "G1": {"drug": 0.9, "genetics": 0.2, "literature": 0.8},
    "G2": {"drug": 0.7, "genetics": 0.1, "literature": 0.5},
    "G3": {"genetics": 0.9, "literature": 0.3},
}
weights = {
    "drug": 0.3, "genetics": 0.3, "expression": 0.1,
    "pathway": 0.1, "literature": 0.1, "animal_model": 0.1,
}
fused = fuse_gene_scores(scores, weights, holdout_type="drug")

# Network propagation on a tiny synthetic graph
edges = [("G1", "G3", 0.9), ("G2", "G3", 0.7)]
ranked = propagate(fused, edges, algorithm="ppr", damping=0.85)
```

These examples use synthetic data. Numbers are for pipeline sanity checks only
and are **not** manuscript results.

## Manuscript results

Reported summary metrics are **not included** in this repository. They are
available from the authors upon reasonable request, as described in the
manuscript's data-availability statement.

## Reproducing the full study (high level)

Full study reproduction requires:

1. Open Targets Platform evidence (manuscript: December 2024 release) for the
   19 diseases used in the study.
2. STRING v12.0 human PPI edges with `combined_score >= 400`.
3. Dual-holdout 5-fold CV (`drug` / `genetic_association` style holdouts;
   positive threshold 0.30) as described in Methods.
4. Optional LLM debate (DeepSeek) **or** deterministic heuristic claim sets;
   the manuscript reports both, with heuristics often strongest on ranking
   metrics.

## Citation

Please cite the manuscript when using these materials.

## License

Apache-2.0 (see `LICENSE`).

## Third-party resources

- Open Targets: https://platform.opentargets.org/
- STRING: https://string-db.org/
