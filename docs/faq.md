# FAQ and troubleshooting

## Installation

**`pip install -e .` fails on Python 3.9.**
The package targets ≥3.9, but the test matrix actively covers 3.9. If you hit
a syntax error, check that you are not on an old editable install
(`pip install -e . --force-reinstall`).

**Do I need an API key?**
No. Open Targets and STRING are public. An LLM key is only needed for
`genepriori debate --mode llm`.

## Data

**How large is the STRING download?**
The human links file is several hundred MB compressed. It is downloaded once
and converted to a compact cached edge list; later runs read that cache.

**Can I work completely offline?**
Yes, after one fetch:

```bash
genepriori fetch --disease "breast carcinoma"
genepriori fetch-ppi
genepriori rank --disease "breast carcinoma" --offline
```

`--offline` never touches the network and raises with a hint when data is
missing. `genepriori demo` is offline by construction.

**Where is the cache?**
`~/.genepriori` by default. Override with `GENEPRIORI_HOME`, `--cache-dir`, or
`cache_dir` in the config. `manifest.json` records every artifact with URL,
timestamp, size and SHA-256.

**Which disease identifiers work?**
Any EFO/MONDO identifier accepted by the current Open Targets release, plus
the 19 registry names. Legacy EFO identifiers are increasingly retired — use
`genepriori search-disease` to find the current ID.

## Evaluation

**`Only N positives ... needs at least 5`.**
The disease has fewer than `--folds` positives at the 0.30 threshold. Options:
lower `--positive-threshold`, lower `--folds`, or raise `--max-genes`.

**Why do `mask` and `restore` give such different numbers?**
They evaluate different things. `mask` splits positives into five subsets and
measures recovery of each subset against all non-positives. `restore` removes
one fifth from the gold standard and evaluates the remaining four fifths while
the removed genes become unlabeled positives with their holdout evidence
restored. Compare numbers only within the same protocol.

**Why is AUPRC so low?**
AUPRC depends on prevalence. With ~2,000 candidate genes and a few dozen
positives, a random ranker already scores ~positives/candidates. Always report
the prevalence (`n_positives` / `n_ranked`) alongside AUPRC.

**Why doesn't network propagation improve my numbers?**
Propagation helps when the PPI neighbourhood carries signal the direct
evidence does not. On synthetic or weakly structured data it can be neutral or
slightly negative. Also check the blend settings: the default is conservative
(`0.7 × direct + 0.3 × network`, discovered genes capped at 80).

## Outputs

**How do I know which gene has no direct evidence?**
`network_discovered = 1` in `ranked_genes.csv`.

**What does `confidence` mean?**
`min(1, 0.35 + 0.12 × n_evidence_types)`. It measures evidence breadth, not
calibrated probability.

**Can I audit a past run?**
`run_manifest.json` records package version, Python, platform, arguments,
configuration, input disease IDs and the cached data provenance.

## Development

**How do I run the tests?**

```bash
pip install -e ".[dev]"
pytest
```

The suite is fully offline and uses a synthetic cache built in a temporary
directory.

**How do I regenerate the shipped demo data?**

```bash
python scripts/make_toy_data.py
```

**How do I add a new evidence type?**
Add it to `STANDARD_TYPES` in `genepriori/evidence.py` and to the mapping in
`OT_TO_STANDARD`. Note that this changes weight vectors and configuration
validation; update the presets in `genepriori/config.py` to match.
