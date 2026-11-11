# GenePriori

Rank genes for any disease by fusing Open Targets evidence with STRING network
propagation, and measure performance under holdouts that prevent literature
circularity from inflating the results.

```bash
genepriori rank --disease "breast carcinoma" --network ppr
```

---

## Install from GitHub

Requires Python 3.9+ and `git`.

```bash
# 1. Clone
git clone https://github.com/Jyx0208/Genepriori.git
cd Genepriori

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 3. Install (editable, so updates are picked up with git pull)
pip install -e .

# 4. Verify
genepriori --version
genepriori doctor
```

`genepriori doctor` checks Python, dependencies, the cache directory, network
access and the LLM configuration. Everything else works offline once data is
cached.

---

## Configure DeepSeek

The multi-agent debate runs with heuristic agents out of the box. To use the
LLM agents, add a DeepSeek key (any OpenAI-compatible endpoint works).

**Option A — `.env` file (recommended)**

```bash
cp .env.example .env
# then edit .env:
# DEEPSEEK_API_KEY=sk-...
```

The file is read from the working directory or from `~/.genepriori/.env`.
`.env` is git-ignored.

**Option B — environment variable**

```bash
export DEEPSEEK_API_KEY=sk-...          # macOS / Linux
$env:DEEPSEEK_API_KEY="sk-..."          # Windows PowerShell
```

Accepted names, in priority order: `GENEPRIORI_LLM_API_KEY`,
`DEEPSEEK_API_KEY`, `OPENAI_API_KEY`.

**Option C — run configuration**

```yaml
llm:
  base_url: https://api.deepseek.com/v1
  model: deepseek-chat
  api_key_env: DEEPSEEK_API_KEY
```

The key itself is never written to the configuration file.

**Verify**

```bash
genepriori doctor
```

```
llm   OK   deepseek-chat @ https://api.deepseek.com/v1 | key sk-a...1234
```

`genepriori debate` uses the LLM whenever a key is available (`--mode auto`),
falls back to the heuristic agents when it is not, and `--mode llm` stops with
a clear message if the key is missing.

---

## Quick start: run it offline first

```bash
genepriori demo
```

This uses synthetic data, needs no network and writes to `demo_results/`:

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

If this prints a table, the installation is good.

---

## Usage

### 1. Download the PPI network (once)

```bash
genepriori fetch-ppi
```

Downloads the STRING human network, filters to `combined_score >= 400`, and
caches a compact edge list. The first run downloads a few hundred MB;
later runs read the cache.

### 2. Download disease evidence

```bash
genepriori fetch --disease "breast carcinoma"
```

Fetches up to 2,000 associated genes from Open Targets and caches the scores.
Don't know the exact name?

```bash
genepriori search-disease "triple-negative breast"
genepriori list-diseases                       # 19 built-in diseases
genepriori list-diseases --category oncology
```

Any ontology ID also works:

```bash
genepriori fetch --disease MONDO_0004989
```

### 3. Rank genes

```bash
genepriori rank --disease "breast carcinoma" --network ppr -o results/breast
```

```
breast carcinoma (MONDO_0004989) - 1000 genes
rank  gene    score     confidence  types  network
----  ------  --------  ----------  -----  -------
1     PTEN    0.660046  0.95        5
2     BRCA1   0.551209  0.83        4
3     TP53    0.522721  0.83        4
4     ESR1    0.484185  0.83        4
5     BRCA2   0.483169  0.71        3
```

Common options:

| option | values | default |
|---|---|---|
| `--weights` | `balanced`, `genetics`, `translational`, `network`, `no_literature`, `literature_only`, or a YAML file | `balanced` |
| `--network` | `ppr`, `rwr`, `neighbor_voting`, `hd`, `ensemble`, `none` | `ppr` |
| `--top` | rows printed | `20` |
| `--max-genes` | Open Targets cap | `2000` |
| `-o` | output directory | `results/rank_<id>` |

Use your own weights:

```bash
cat > my_weights.yaml <<'EOF'
drug: 0.3
genetics: 0.3
expression: 0.05
pathway: 0.2
literature: 0.1
animal_model: 0.05
EOF
genepriori rank --disease MONDO_0004989 --weights my_weights.yaml
```

### 4. Read the output

`ranked_genes.csv` — one row per gene:

| column | meaning |
|---|---|
| `rank` | position in the final ranking |
| `gene` | HGNC symbol |
| `score` | final score (direct evidence blended with network) |
| `confidence` | 0.35 + 0.12 × number of contributing evidence types, capped at 1 |
| `n_evidence_types` | how many of the six types contribute |
| `network_discovered` | 1 = no direct evidence, found through the PPI |
| `drug`, `genetics`, `expression`, `pathway`, `literature`, `animal_model` | weighted contribution of each type |
| `ensembl_id` | Ensembl gene ID |

`result.json` — the same data in JSON.
`run_manifest.json` — version, parameters, data provenance, timestamp.

### 5. Measure it with leakage control

```bash
genepriori evaluate --disease "breast carcinoma" --holdout both -o results/eval
```

```
auprc             holdout   protocol  mean    sd      ci_low  ci_high  R@20
----------------  --------  --------  ------  ------  ------  -------  ------
breast carcinoma  drug      mask      0.0224  0.0128  0.0065  0.0383   0.0758
breast carcinoma  genetics  mask      0.1637  0.0140  0.1463  0.1812   0.0168
```

What it does: positives are genes with the holdout evidence type above 0.30.
That evidence type is masked during scoring, so the model has to recover them
from the other five types plus the network. Two protocols:

- `mask` (default) — holdout evidence masked for every gene; folds partition
  the positives to report stability.
- `restore` — one fifth of positives is removed from the gold standard and its
  holdout evidence restored; the rest is evaluated (`--protocol restore`).

Useful flags:

```bash
genepriori evaluate --disease MONDO_0004989 --holdout genetics --folds 5
genepriori evaluate --disease MONDO_0004989 --positive-threshold 0.2
```

If you see `Only N positives ... needs at least 5`, lower
`--positive-threshold`, lower `--folds`, or fetch more genes with
`--max-genes`.

### 6. Compare method variants across diseases

```bash
genepriori benchmark --diseases oncology --holdout both \
    --variants baseline,ppr,rwr,genetics_ppr -o results/benchmark
```

Runs every variant on every selected disease and reports paired bootstrap
deltas, win rates, sign tests and BH-adjusted p-values. `--diseases` accepts
`oncology`, `registry` (all 19), a category, or a comma-separated list.

### 7. Multi-agent debate

```bash
genepriori debate --diseases oncology --rounds 3 --budget 12 --holdout both
```

Four agents propose parameterised claims over three rounds: propose → audit →
score/select → execute → accept/reject → feedback. A Governor allocates the
budget (40% / 35% / remaining of 12 claims) and accepts a claim only when one
of three criteria holds (majority rule, bootstrap CI above zero, or a strong
consistent override). The LeakageAuditor is a hard gate before execution.

Modes:

| flag | behaviour |
|---|---|
| `--mode auto` (default) | LLM agents if `DEEPSEEK_API_KEY` is set, otherwise heuristic |
| `--mode llm` | require the LLM; fail with instructions if no key |
| `--mode heuristic` | never call an LLM |

Override the endpoint per run:

```bash
genepriori debate --mode llm \
    --llm-model deepseek-reasoner \
    --llm-base-url https://api.deepseek.com/v1
```

If an LLM call fails mid-debate, that agent falls back to its heuristic
version, the error is recorded in `debate.json` under `llm_errors`, and the CLI
prints a warning. The run always completes.

---

## Python API

```python
from genepriori import prioritize

result = prioritize("breast carcinoma", weights="translational", network="ppr")
for row in result.top(10):
    print(row.rank, row.gene, round(row.score, 4), row.network_discovered)

result.to_csv("ranked_genes.csv")
```

Lower-level pieces, if you want to build your own pipeline:

```python
from genepriori import fuse_gene_scores, propagate, evaluate_ranking

fused = fuse_gene_scores(
    {"G1": {"drug": 0.9, "genetics": 0.2}, "G2": {"genetics": 0.9}},
    {"drug": 0.5, "genetics": 0.5},
    holdout_type="drug",          # drug weight forced to 0, rest renormalized
)
propagated = propagate(fused, [("G1", "G2", 0.9)], algorithm="ppr", damping=0.85)
print(evaluate_ranking(propagated, {"G1"}))
```

---

## Command reference

| command | what it does |
|---|---|
| `genepriori demo` | offline synthetic run |
| `genepriori doctor` | check environment, cache, network |
| `genepriori list-diseases` | built-in registry |
| `genepriori search-disease TEXT` | name → ontology ID |
| `genepriori fetch --disease D` | download Open Targets evidence |
| `genepriori fetch-ppi` | download STRING network |
| `genepriori rank --disease D` | ranked genes |
| `genepriori evaluate --disease D` | dual-holdout K-fold CV |
| `genepriori benchmark` | compare variants across diseases |
| `genepriori debate` | multi-agent claim debate |
| `genepriori config --write run.yaml` | configuration template |

Global flags work before or after the subcommand: `--cache-dir`, `--offline`,
`--json`, `-v`, `-q`. Run `genepriori <command> --help` for all options.

---

## Troubleshooting

**`No cached evidence ...`** — run `genepriori fetch --disease <id>` first, or
drop `--offline`.

**`no API key found`** — set `DEEPSEEK_API_KEY`, add it to `.env`, or run
`--mode heuristic`. `genepriori doctor` shows whether a key is detected.

**`Only N positives ... needs at least 5`** — lower `--positive-threshold` or
`--folds`, or increase `--max-genes`.

**Where is the cache?** `~/.genepriori`. Override with `GENEPRIORI_HOME`,
`--cache-dir`, or `cache_dir` in the config. `manifest.json` records every
download with URL, timestamp, size and SHA-256.

**Old EFO IDs no longer resolve** — Open Targets moved to MONDO. Use
`genepriori search-disease` to find the current ID.

---

## License

Apache-2.0. Third-party data remains subject to the terms of Open Targets and
STRING.
