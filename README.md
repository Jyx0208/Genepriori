# GenePriori

**Leakage-aware gene prioritization for translational genomics.**

GenePriori ranks genes for a disease by fusing Open Targets evidence with
STRING network propagation, and measures performance under holdouts that
prevent the most common evaluation artefact in gene prioritization:
using literature-encoded associations to "predict" associations that were
themselves derived from the literature.

- **Ready to use**: one command returns a ranked gene table for any EFO/MONDO disease.
- **Leakage-aware**: drug-target and genetic-association holdouts are masked at scoring time, with an auditor and cross-validation.
- **Reproducible**: deterministic seeds, cached inputs with checksums, and a run manifest for every output.
- **Extensible**: the multi-agent claim debate from the manuscript is included, with heuristic agents that need no API key.

---

## Install

```bash
pip install -e .
```

Requires Python 3.9+. Dependencies: `numpy`, `scipy`, `requests`, `PyYAML`.

Verify the installation:

```bash
genepriori --version
genepriori doctor
```

---

## 60-second offline demo

No network, no API key, no downloads — synthetic data is generated locally:

```bash
genepriori demo
```

Output:

```
synthetic breast carcinoma: 253 genes ranked
rank  gene     score     confidence  types  network
----  -------  --------  ----------  -----  -------
1     GENE066  0.98439   1.00        6
2     GENE108  0.844228  1.00        6
3     GENE110  0.814582  1.00        6

Leakage-controlled evaluation (5-fold):
auprc                       holdout   protocol  mean    sd      ci_low  ci_high  R@20
--------------------------  --------  --------  ------  ------  ------  -------  ------
synthetic breast carcinoma  drug      mask      0.2162  0.0950  0.0983  0.3341   0.4444
synthetic breast carcinoma  genetics  mask      0.2225  0.1316  0.0591  0.3859   0.7267
```

Files are written to `demo_results/` (`rank/ranked_genes.csv`,
`evaluation/metrics.json`, `run_manifest.json`).

> The demo data is synthetic and deliberately small. It exists to prove the
> pipeline runs end to end; it says nothing about real disease performance.

---

## Real data workflow

```bash
# 1. STRING human PPI network (~200 MB download, cached once)
genepriori fetch-ppi

# 2. Open Targets evidence for a disease (2,000 genes by default)
genepriori fetch --disease "breast carcinoma"

# 3. Rank genes
genepriori rank --disease "breast carcinoma" --network ppr -o results/breast

# 4. Leakage-controlled evaluation (dual holdout, 5-fold)
genepriori evaluate --disease "breast carcinoma" --holdout both -o results/eval
```

Any ontology identifier works too:

```bash
genepriori rank --disease MONDO_0004989
genepriori search-disease "triple-negative breast"
```

Built-in registry (19 diseases, 7 solid tumours):

```bash
genepriori list-diseases
genepriori list-diseases --category oncology
```

---

## What the outputs mean

`ranked_genes.csv` contains one row per gene:

| column | meaning |
|---|---|
| `rank` | position in the final ranking |
| `gene` | HGNC symbol from Open Targets |
| `score` | final score (direct evidence blended with network propagation) |
| `confidence` | 0.35 + 0.12 × number of contributing evidence types, capped at 1 |
| `n_evidence_types` | how many of the six evidence types contribute |
| `network_discovered` | 1 when the gene has no direct evidence and was found through the PPI |
| `drug … animal_model` | weighted contribution of each evidence type to the score |
| `ensembl_id` | Ensembl gene identifier |

Every run also writes `result.json` and `run_manifest.json` (package version,
Python, parameters, data provenance, timestamps).

---

## Python API

```python
from genepriori import prioritize

result = prioritize(
    "breast carcinoma",
    weights="translational",   # preset: balanced | genetics | translational | network | no_literature | literature_only
    network="ppr",             # ppr | rwr | neighbor_voting | hd | ensemble | none
)
print(result.top(10))
result.to_csv("ranked_genes.csv")
```

Low-level building blocks are exported as well:

```python
from genepriori import (
    fuse_gene_scores, fuse_gene_scores_detailed, propagate,
    build_positives, kfold_splits, evaluate_ranking,
    Claim, LeakageAuditor,
)

fused = fuse_gene_scores(
    {"G1": {"drug": 0.9, "genetics": 0.2}, "G2": {"genetics": 0.9}},
    {"drug": 0.5, "genetics": 0.5},
    holdout_type="drug",          # drug weight forced to 0, rest renormalized
)
propagated = propagate(fused, [("G1", "G2", 0.9)], algorithm="ppr", damping=0.85)
```

---

## Core concepts

### Six standardized evidence types

Open Targets fine-grained evidence is mapped to `drug`, `genetics`,
`expression`, `pathway`, `literature`, `animal_model`; the maximum score per
type and gene is retained. Scores are min-max normalized per type, then
weighted-summed with non-negative weights.

### Dual holdouts

| holdout | positives | masked evidence |
|---|---|---|
| `drug` | genes with drug score > 0.30 | drug |
| `genetics` | genes with genetics score > 0.30 | genetics |

The holdout type's weight is forced to zero and the remaining weights are
renormalized, so measured performance comes from complementary signals.

### Two evaluation protocols

The manuscript's wording admits two readings, so both are implemented
explicitly (`--protocol`):

- **`mask`** (default, strictest): holdout evidence is masked for every gene.
  The folds partition the positives; each fold measures recovery against all
  non-positives. Reported mean/SD reflects stability across positive subsets.
- **`restore`** (manuscript-literal): for each fold, that fold's positives are
  removed from the gold standard and their holdout evidence is restored; the
  remaining positives are evaluated. Ranking is recomputed per fold.

In both protocols, evaluated positives never contribute their holdout
evidence.

### What leakage control does *not* do

The auditor blocks direct use of the holdout type. It cannot remove semantic
collinearity among the remaining evidence types — for example, literature
annotations still encode published drug–gene relationships. This is a known
limitation and is reported as such.

### Network propagation

Propagation runs on a 1-hop subgraph of seed genes (top 30 neighbours per
seed, edges with score ≥ 0.4). Seed scores are blended
`0.7 × direct + 0.3 × network`; network-discovered genes receive
`0.25 × network × 1/(1 + ln(degree + 1))`, capped at 80 genes. Network scores
are min-max normalized before blending so the blend weight is meaningful.

---

## CLI reference

| command | purpose |
|---|---|
| `genepriori demo` | offline synthetic end-to-end run |
| `genepriori doctor` | check environment, cache, network access |
| `genepriori list-diseases` | built-in 19-disease registry |
| `genepriori search-disease TEXT` | resolve a disease name to an ID |
| `genepriori fetch --disease D` | download Open Targets evidence |
| `genepriori fetch-ppi` | download and cache STRING |
| `genepriori rank --disease D` | ranked genes + manifest |
| `genepriori evaluate --disease D` | dual-holdout 5-fold CV |
| `genepriori benchmark --diseases oncology` | compare method variants across diseases |
| `genepriori debate --holdout both` | multi-agent claim debate (`--mode llm` optional) |
| `genepriori config --write run.yaml` | template configuration |

Global flags work before or after the subcommand: `--cache-dir`, `--offline`,
`--json`, `-v`, `-q`.

Run `genepriori <command> --help` for all options.

---

## Multi-agent debate

```bash
genepriori debate --diseases oncology --rounds 3 --budget 12 --holdout both
```

Four agents (CausalGenetics, NetworkMechanism, Translational,
ConsensusDesigner) propose parameterized claims over three rounds. Each round:
propose → audit → score/select → execute → accept/reject → feedback. The
Governor allocates a 12-claim budget (40%/35%/remaining), scores claims from
predicted AUPRC/Recall gains, cost and novelty, and accepts claims only when
one of three criteria holds (majority rule, bootstrap CI above zero, or a
strong consistent override). The LeakageAuditor is a hard gate.

LLM mode uses any OpenAI-compatible endpoint (DeepSeek by default):

```bash
export GENEPRIORI_LLM_API_KEY=...          # or DEEPSEEK_API_KEY / OPENAI_API_KEY
export GENEPRIORI_LLM_BASE_URL=https://api.deepseek.com/v1
export GENEPRIORI_LLM_MODEL=deepseek-chat
genepriori debate --mode llm
```

If a call fails, the agent falls back to its heuristic counterpart, so the
debate always completes.

---

## Data sources and versions

| source | default | pin for reproduction |
|---|---|---|
| Open Targets Platform | current release, live GraphQL API | `--ot-release` is recorded in the manifest; fetch date is stored per disease |
| STRING | v12.5 | `--string-version 12.0` (manuscript) |

Downloads are cached under `~/.genepriori` (override with `GENEPRIORI_HOME`,
`--cache-dir`, or `cache_dir` in the config). Each artifact is registered in
`manifest.json` with URL, timestamp, size and SHA-256.

---

## Reproducing the manuscript

The original experiment code was not preserved; this package re-implements the
described methods. Because Open Targets and STRING releases have moved on,
numbers will not match the paper exactly. See
[docs/reproduction.md](docs/reproduction.md) for the recommended settings and
a list of known sources of divergence.

---

## Project status

- Algorithmic core, data layer, CLI, evaluation and debate: implemented and tested.
- Test suite: `pytest` (offline, no network required).
- Not included: web UI, non-human organisms, automated PyPI publishing.

## Citation

See [CITATION.cff](CITATION.cff).

## License

Apache-2.0. See [LICENSE](LICENSE). Third-party data remains subject to the
terms of Open Targets and STRING.
