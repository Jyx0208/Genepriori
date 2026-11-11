# Reproducing the manuscript

The original experiment scripts were not preserved. This package re-implements
the methods described in the manuscript, so the code released here is
complete and runnable, but it is a **re-implementation**, not the original
code that produced the published numbers.

## Version differences

| component | manuscript | default here |
|---|---|---|
| Open Targets | December 2024 release | current live release |
| STRING | v12.0 | v12.5 |
| Python | 3.13 | ≥3.9 (tested on 3.12) |
| NumPy / SciPy | 2.4 / 1.17 | current |

Open Targets has also migrated disease identifiers from EFO to MONDO. Many
legacy EFO identifiers no longer resolve (for example `EFO_0000305` for breast
carcinoma is now `MONDO_0004989`). The built-in registry uses current MONDO
identifiers.

## Pinning the STRING release

```bash
genepriori fetch-ppi --string-version 12.0 --min-combined-score 400
genepriori rank --disease MONDO_0004989 --string-version 12.0
```

`genepriori doctor` and every `run_manifest.json` record what was actually
used.

## Reproducing the paper's tables

```bash
# 19 diseases, both holdouts, all variants
genepriori fetch --all
genepriori benchmark --diseases registry --holdout both \
    --variants baseline,ppr,rwr,genetics_ppr,translational_ppr \
    --n-bootstrap 2000 -o results/paper_benchmark

# Multi-agent debate
genepriori debate --diseases registry --rounds 3 --budget 12 \
    --holdout both --mode heuristic -o results/paper_debate
```

Runtime: the full benchmark is dominated by evidence fetches on the first run
(19 diseases × up to 2,000 genes) and by 19 × 2 holdouts × 5 folds × variants
of ranking. On a single workstation expect tens of minutes to a few hours.

## Known sources of divergence

1. **Release drift.** Evidence scores change between Open Targets releases,
   which changes positive sets and therefore every metric. The current release
   also renamed the drug datatype from `known_drug` to `clinical`; both are
   mapped to the `drug` standard type here.
2. **Identifier migration.** EFO → MONDO means the exact disease node may
   differ (for example the generic "multiple sclerosis" node is not indexed in
   the current release; relapsing-remitting MS is used instead).
3. **Evaluation protocol.** The manuscript's description of restoring
   holdout evidence is ambiguous. Both readings are available
   (`--protocol mask` / `--protocol restore`); they produce different
   absolute numbers, so compare like with like.
4. **Gene symbols as keys.** The original pipeline may have keyed genes by
   Ensembl identifier. Symbol collisions are merged by maximum score here.
5. **Network normalization.** Propagation scores are min-max normalized before
   blending so that the blend weight is meaningful; a raw-probability
   implementation would behave differently.
6. **Random seeds.** Fold assignment uses seed 42 and is deterministic, but
   bootstrap draws and dictionary ordering can differ from the original code.

## What *is* deterministic here

- Fold splitting: alphabetical sort → `random.Random(42).shuffle` → round-robin.
- Ranking ties: broken by gene symbol.
- Neighbour selection and top-N cuts: tie-broken by gene symbol.
- Bootstrap: seeded, chunked, reproducible for a fixed seed.

Re-running the same command twice on the same cache produces byte-identical
`ranked_genes.csv` and `result.json`.
