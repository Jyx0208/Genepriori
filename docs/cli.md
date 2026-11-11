# CLI reference

Global flags are accepted before or after the subcommand:

| flag | meaning |
|---|---|
| `--cache-dir DIR` | override the cache directory |
| `--offline` | never touch the network; fail if data is missing |
| `--json` | machine-readable output instead of tables |
| `-v` / `-vv` | info / debug logging |
| `-q` | errors only |
| `--version` | print the version |

---

## `genepriori doctor`

Checks Python, dependencies, cache writability, cached data, and Open Targets
reachability. Exit code 1 if any check fails.

```bash
genepriori doctor
genepriori doctor --offline --json
```

## `genepriori list-diseases`

Lists the 19 built-in diseases (7 solid tumours plus neurodegenerative,
psychiatric, autoimmune, respiratory, cardiovascular, metabolic and renal
entries).

```bash
genepriori list-diseases
genepriori list-diseases --category oncology
genepriori list-diseases --json
```

## `genepriori search-disease`

Resolves a free-text name through the Open Targets search API.

```bash
genepriori search-disease "triple-negative breast"
genepriori search-disease "lupus" --limit 5
```

## `genepriori fetch`

Downloads Open Targets evidence and caches it as `diseases/<id>.json`.

```bash
genepriori fetch --disease "breast carcinoma"
genepriori fetch --disease MONDO_0004989 --max-genes 5000
genepriori fetch --all          # whole registry (slow)
genepriori fetch --disease "asthma" --refresh
```

| option | default | meaning |
|---|---|---|
| `--disease` | required (repeatable) | registry name or ontology ID |
| `--all` | off | fetch the whole registry |
| `--max-genes` | 2000 | cap on associated genes per disease |
| `--refresh` | off | ignore cached evidence |

## `genepriori fetch-ppi`

Downloads STRING (human) and writes a compact cached edge list.

```bash
genepriori fetch-ppi
genepriori fetch-ppi --string-version 12.0 --min-combined-score 700
```

| option | default |
|---|---|
| `--string-version` | `12.5` |
| `--min-combined-score` | `400` |

The first run downloads a large file (hundreds of MB). Later runs read the
cached edge list (`ppi/v<version>/9606.edges.min<score>.tsv.gz`).

## `genepriori rank`

Produces the ranked gene table.

```bash
genepriori rank --disease "breast carcinoma" --network ppr -o results/breast
genepriori rank --disease MONDO_0004989 --weights genetics --network rwr --top 50
genepriori rank --disease MONDO_0004989 --weights my_weights.yaml
genepriori rank --disease MONDO_0004989 --json --top 10
```

| option | default | meaning |
|---|---|---|
| `--disease` | required | registry name or ontology ID |
| `-o/--out` | `results/rank_<id>` | output directory |
| `--weights` | `balanced` | preset name or YAML/JSON mapping |
| `--network` | `ppr` | `ppr`, `rwr`, `neighbor_voting`, `hd`, `ensemble`, `none` |
| `--top` | 20 | rows printed to the console |
| `--config` | none | run configuration file |
| `--max-genes` | 2000 | Open Targets cap |
| `--string-version` | 12.5 | STRING release |
| `--min-combined-score` | 400 | STRING threshold |

Weight presets: `balanced`, `genetics`, `translational`, `network`,
`no_literature`, `literature_only`.

## `genepriori evaluate`

Dual-holdout K-fold evaluation.

```bash
genepriori evaluate --disease "breast carcinoma"
genepriori evaluate --disease MONDO_0004989 --holdout genetics --folds 5
genepriori evaluate --disease MONDO_0004989 --protocol restore
```

| option | default | meaning |
|---|---|---|
| `--disease` | required | may be combined with `--diseases` |
| `--diseases` | none | additional diseases (repeatable) |
| `--holdout` | `both` | `drug`, `genetics`, `both` |
| `--folds` | 5 | must be ≤ number of positives |
| `--protocol` | `mask` | `mask` or `restore` |
| `--k` | 20 | Recall@K |
| `--positive-threshold` | 0.30 | positive-set cut-off |
| `--weights`, `--network`, `--config` | as in `rank` | |
| `-o/--out` | `results/evaluation` | output directory |

Outputs per disease and holdout: `<id>_<holdout>_metrics.json` and
`<id>_<holdout>_folds.csv` (with `mean`, `sd`, `ci_low`, `ci_high` rows).

## `genepriori benchmark`

Compares method variants across diseases with paired statistics.

```bash
genepriori benchmark --diseases oncology --holdout both \
    --variants baseline,ppr,rwr,genetics_ppr -o results/benchmark
```

Variants: `baseline` (balanced, no network), `ppr`, `rwr`, `genetics_ppr`,
`translational_ppr`. `baseline` is always included.

| option | default |
|---|---|
| `--diseases` | `oncology` (use `registry` for all 19) |
| `--holdout` | `both` |
| `--variants` | `baseline,ppr,rwr` |
| `--n-bootstrap` | 2000 |
| `--alpha` | 0.05 |

Outputs `benchmark.csv` (long format), `comparison.json` (paired bootstrap,
win rate, sign test, BH-adjusted p-values) and `run_manifest.json`.

## `genepriori debate`

Runs the multi-agent claim debate.

```bash
genepriori debate --diseases oncology --rounds 3 --budget 12 --holdout both
genepriori debate --diseases MONDO_0004989 --holdout genetics --mode llm
```

| option | default |
|---|---|
| `--diseases` | `oncology` |
| `--rounds` | 3 |
| `--budget` | 12 |
| `--mode` | `auto` (`llm` if a key is found, else `heuristic`) |
| `--holdout` | `both` |
| `--llm-model` | `deepseek-chat` |
| `--llm-base-url` | `https://api.deepseek.com/v1` |
| `--llm-api-key` | none (prefer `DEEPSEEK_API_KEY` or `.env`) |

API key resolution: `GENEPRIORI_LLM_API_KEY`, then `DEEPSEEK_API_KEY` (or the
variable named by `llm.api_key_env`), then `OPENAI_API_KEY`. A `.env` file in
the working directory or in `~/.genepriori/` is loaded automatically.

In `auto` mode the report records both `mode_requested` and `mode`; failed LLM
calls fall back to the heuristic agent and are listed under `llm_errors`.

## `genepriori demo`

Offline synthetic end-to-end run. Writes `demo_results/rank`,
`demo_results/evaluation` and `run_manifest.json`.

## `genepriori config`

```bash
genepriori config                      # print the default configuration
genepriori config --write run.yaml     # write an editable template
genepriori rank --disease X --config run.yaml
```

---

## Exit codes

| code | meaning |
|---|---|
| 0 | success |
| 1 | `doctor` found a failing check, or no search hits |
| 2 | invalid input, missing data, or a failed download |
| 130 | interrupted |
