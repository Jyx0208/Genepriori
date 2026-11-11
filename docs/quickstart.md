# Quick start

## 1. Install

```bash
git clone https://github.com/Jyx0208/Genepriori.git
cd Genepriori
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
pip install -e .
genepriori --version
```

## 2. Prove it works offline

```bash
genepriori demo
```

This generates a synthetic cache in `~/.genepriori/demo-cache`, ranks 253
genes, runs a leakage-controlled 5-fold evaluation, and writes everything to
`demo_results/`. No network access is used.

Inspect the outputs:

```bash
head -5 demo_results/rank/ranked_genes.csv
cat demo_results/evaluation/MONDO_9000001_drug_metrics.json
```

## 3. Fetch real data

```bash
genepriori doctor            # environment + cache + API reachability
genepriori fetch-ppi         # STRING v12.5 human network, cached once
genepriori fetch --disease "breast carcinoma"
```

If you do not know the exact name:

```bash
genepriori search-disease "breast carcinoma"
genepriori list-diseases
```

## 4. Rank genes

```bash
genepriori rank --disease "breast carcinoma" --network ppr --top 20 -o results/breast
```

Outputs:

```
results/breast/
├── ranked_genes.csv      # full ranking with evidence breakdown
├── result.json           # machine-readable result
└── run_manifest.json     # versions, parameters, provenance
```

## 5. Measure it properly

```bash
genepriori evaluate --disease "breast carcinoma" --holdout both -o results/eval
```

You get AUPRC, Recall@20, SD across folds and a t-based 95% CI for both
holdouts. Read the `protocol` column: `mask` is the strict default; `restore`
follows the manuscript's wording literally.

## 6. Go further

```bash
# Compare several method variants over the oncology registry
genepriori benchmark --diseases oncology --holdout both \
    --variants baseline,ppr,rwr,genetics_ppr -o results/benchmark

# Run the multi-agent claim debate (heuristic agents, no API key)
genepriori debate --diseases oncology --rounds 3 --budget 12 --holdout both
```

## Troubleshooting

- `No cached evidence ...` — run `genepriori fetch --disease <id>` first, or
  drop `--offline`.
- `Only N positives ... needs at least 5` — the disease has too few positives
  at the 0.30 threshold. Lower `--positive-threshold` (e.g. 0.2), reduce
  `--folds`, or increase `--max-genes`.
- Downloads look stuck — STRING's human links file is large; the first
  `fetch-ppi` can take several minutes. Subsequent runs read the cached edge
  list.
