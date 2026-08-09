# GenePriori

Python tools for evidence fusion, holdout evaluation, leakage checks, and
network propagation in gene prioritization.

## Features

- Evidence-type mapping and weighted score fusion
- Dual-holdout positive sets and deterministic fold splitting
- Policy checks for direct use of held-out evidence
- AUPRC, Recall@K, and bootstrap confidence intervals
- PPR and RWR propagation with degree correction

The leakage checks cover direct use of the selected holdout type. They do not
remove correlations among the remaining evidence types.

## Install

Requires Python 3.9 or later.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Example

```python
from genepriori.evidence import fuse_gene_scores
from genepriori.network import propagate

scores = {
    "G1": {"drug": 0.9, "genetics": 0.2},
    "G2": {"drug": 0.7, "genetics": 0.1},
    "G3": {"genetics": 0.9},
}
weights = {"drug": 0.5, "genetics": 0.5}

fused = fuse_gene_scores(scores, weights, holdout_type="drug")
edges = [("G1", "G3", 0.9), ("G2", "G3", 0.7)]
propagated = propagate(fused, edges, algorithm="ppr", damping=0.85)
```

The example uses synthetic data.

## Data

Third-party datasets and API credentials are not included. Data can be obtained
from [Open Targets](https://platform.opentargets.org/) and
[STRING](https://string-db.org/), subject to their respective terms.

## License

Apache-2.0. See [LICENSE](LICENSE).
