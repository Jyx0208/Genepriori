# Changelog

All notable changes to this project are documented here.

## 0.2.0 — 2026-11-11

A complete, installable tool built around the core algorithm modules. The
public API of the five existing modules is preserved.

### Added

- **Packaging**: `pyproject.toml`, `genepriori` console script, package
  re-exports, Python ≥3.9.
- **Data layer** (`genepriori/data/`):
  - Open Targets GraphQL client with pagination, rate limiting, retries and caching;
  - STRING v12.x download, ID mapping, threshold filtering and a compact cached edge list;
  - local cache with a SHA-256 manifest;
  - 19-disease registry using current MONDO identifiers.
- **Pipelines** (`genepriori/pipeline/`): ranking, dual-holdout K-fold
  evaluation with `mask` and `restore` protocols, and CSV/JSON/manifest writers.
- **CLI**: `demo`, `doctor`, `list-diseases`, `search-disease`, `fetch`,
  `fetch-ppi`, `rank`, `evaluate`, `benchmark`, `debate`, `config`.
- **Multi-agent debate** (`genepriori/agents/`): four heuristic agents, a
  Governor with budget allocation and three acceptance criteria, a
  three-round six-step workflow, and an OpenAI-compatible LLM mode
  (DeepSeek by default) with heuristic fallback.
- **LLM configuration**: `llm:` block in run configurations, `.env` loading,
  `DEEPSEEK_API_KEY` / `GENEPRIORI_LLM_API_KEY` / `OPENAI_API_KEY` resolution,
  `--mode auto|llm|heuristic`, per-run `--llm-model` / `--llm-base-url`
  overrides, and an LLM status line in `genepriori doctor`.
- **Statistics**: paired bootstrap, exact binomial sign test and
  Benjamini–Hochberg FDR.
- **Propagation**: `neighbor_voting`, heat diffusion (`hd`) and `ensemble`
  algorithms, matching the auditor's whitelist.
- **Docs**: README rewrite, quick start, CLI reference, methods, reproduction
  notes and FAQ; GitHub Actions CI.
- **Synthetic demo data** and `genepriori demo` for a fully offline first run.

### Changed

- Network scores are min-max normalized before blending, so `blend` has a real
  effect (raw propagation probability masses are ~1/N).
- Neighbour and top-N selections now break ties by gene symbol, making output
  deterministic regardless of input ordering.
- `extract_1hop_subgraph` retains isolated seed genes instead of silently
  dropping them.
- `bootstrap_mean_ci` is vectorized in chunks; results remain seeded and
  deterministic.
- `auprc` warns when the supplied ranking disagrees with the supplied scores.
- Negative weights are rejected by the leakage auditor (they were silently
  clipped before).
- `fuse_gene_scores` no longer computes and discards a confidence value;
  `fuse_gene_scores_detailed` exposes scores, confidence and the per-type
  breakdown.

### Fixed

- **Drug evidence mapping.** Current Open Targets releases expose the drug
  datatype as `clinical` rather than `known_drug`. Without mapping it, drug
  evidence and drug-holdout positives were silently empty on live data.
- Removed dead code in `evidence.py` and `network.py`.
- Propagation now warns instead of silently returning a non-converged iterate.
- `propagate` documents and accepts the `pagerank` alias and `none`.
- `holdout.kfold_splits` rejects `n_folds < 2`.

## 0.1.0 — initial public release

Core modules: evidence fusion, holdout construction, leakage audit, metrics
and network propagation.
