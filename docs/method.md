# Methods implemented

This page documents exactly what the code computes, so that results can be
audited without reading the source. Defaults follow the manuscript unless
noted.

## 1. Evidence mapping

Open Targets fine-grained evidence types are mapped to six standardized types:

| Open Targets type | Standard type |
|---|---|
| `known_drug`, `drugs`, `clinical` | `drug` |
| `genetic_association`, `somatic_mutation`, `genetic_literature` | `genetics` |
| `rna_expression`, `gene_expression`, `proteomics` | `expression` |
| `affected_pathway`, `pathway` | `pathway` |
| `literature` | `literature` |
| `animal_model` | `animal_model` |

When several fine-grained scores map to the same standard type for the same
gene, the **maximum** is kept. The client reads Open Targets
`datatypeScores`; `datasourceScores` are ignored by default
(`include_datasources=True` enables them).

Genes are keyed by `approvedSymbol`. Duplicate symbols are merged by taking the
maximum per evidence type; `ensembl_ids` retains one identifier for
traceability.

## 2. Score fusion

For gene *g*, standardized type *t*, weight *w_t* ≥ 0, normalized score *x_{g,t}* ∈ [0, 1]:

```
direct(g) = Σ_t w_t · x_{g,t}
```

- Each type is min-max normalized **across the genes of the run**.
  A type with a constant value (including a single-gene type) contributes zero.
- Weights are clipped at zero, missing types are zero, and weights are
  renormalized to sum to 1.
- Under a holdout, `w_holdout = 0` and the remainder is renormalized. If all
  remaining weight is zero, uniform weights over the non-holdout types are used.
- `confidence(g) = min(1, 0.35 + 0.12 · n_types(g))`, where `n_types` counts
  evidence types with a strictly positive contribution.

Ties are broken by gene symbol (ascending), so rankings are deterministic.

## 3. Network propagation

Seeds are genes with `direct(g) > 0`.

**Subgraph.** 1-hop neighbourhood of the seeds; at most
`max_neighbors_per_seed = 30` highest-weight neighbours per seed; edges with
`combined_score < 400` (weight < 0.4) removed. Isolated seeds are retained as
zero-degree nodes.

**Transition matrix.** Column-stochastic `W[:, j] = A[:, j] / deg(j)` with zero
columns left at zero.

**Algorithms**

| name | iteration |
|---|---|
| `ppr` (alias `pagerank`) | `p_{t+1} = d · W p_t + (1-d) · p0`, `d = 0.85` |
| `rwr` | `p_{t+1} = (1-r) · W p_t + r · p0`, `r = 0.30` |
| `neighbor_voting` | `x = A x`, L1-normalized after each step |
| `hd` | `p_{t+1} = p_t - 0.25 · L p_t`, `L = I - D^{-1/2} A D^{-1/2}` |
| `ensemble` | mean of the L1-normalized vectors above |
| `none` / `""` | no propagation |

Convergence for `ppr`/`rwr`: L2 norm < `1e-6`, at most 100 iterations; a
`RuntimeWarning` is emitted if the limit is hit.

**Blending.** Network scores are min-max normalized over the subgraph first
(otherwise PPR's ~1/N scale makes the blend weight meaningless).

```
seed:              final(g) = 0.70 · direct(g) + 0.30 · net(g)
network-discovered: final(g) = 0.25 · net(g) · 1/(1 + ln(deg(g)+1))
```

At most `max_network_genes = 80` network-discovered genes are added, ranked by
the penalized score with symbol tie-breaking.

## 4. Dual holdouts

Positive sets use a score **strictly greater than** `0.30` for the holdout
type:

- `drug`: `known_drug` evidence (standard type `drug`)
- `genetics`: `genetic_association` evidence (standard type `genetics`)

Folds: positives sorted alphabetically, shuffled with `seed = 42`, split
round-robin into `K = 5` disjoint folds.

## 5. Evaluation protocols

### `mask` (default)

One ranking, computed with the holdout weight zeroed for **every** gene.
Fold *k* evaluates the fold's positives against all non-positives. Mean/SD
across folds describes stability across positive subsets; the ranking itself
does not change between folds.

### `restore` (manuscript-literal)

For fold *k*:

1. the fold's positives are removed from the gold standard;
2. their holdout evidence is restored before scoring;
3. the remaining positives are evaluated against all non-positives;
4. the removed genes are excluded from the ranked list entirely.

Ranking is recomputed per fold because the restored genes differ.

In both protocols, genes being evaluated never contribute their holdout
evidence.

## 6. Metrics

**AUPRC** — trapezoidal integration of precision over recall, starting from the
anchor `(recall = 0, precision = 1)`. This differs slightly from
`sklearn.metrics.average_precision_score` (step-wise). The curve ends at the
maximum achieved recall.

**Recall@K** — hits in the top *K* divided by the total number of positives.
Default `K = 20`.

**Confidence interval** — t-based 95% interval of the fold mean:
`mean ± t(0.975, K-1) · sd / sqrt(K)`.

**Bootstrap** — percentile CI over `n_bootstrap = 2000` resamples; one-sided
p-value is the fraction of resampled means ≤ 0.

**Sign test** — exact binomial test on win/loss counts (ties excluded).

**BH-FDR** — Benjamini–Hochberg adjustment, applied within a family of
comparisons and returned in input order.

## 7. Governor (debate)

- Budget `B = 12` claims over 3 rounds: `floor(0.40B)`, `floor(0.35B)`, rest.
  The baseline occupies one slot in round 1.
- Claim score:
  `1.0·E[ΔAUPRC] + 0.5·E[ΔRecall@20] - 0.05·cost + 0.02·novelty`.
- Acceptance (any one):
  1. majority rule — win rate ≥ 0.60 and mean ΔAUPRC > 1e-4;
  2. statistical support — bootstrap 95% CI lower bound > 0;
  3. strong override — win rate ≥ 0.80 and mean ΔAUPRC > 2e-4.

Duplicate interventions (same weights, algorithm and parameters) are only
executed once.

## 8. Leakage auditor

Rejects a claim when any of the following holds:

1. `evidence_type_weights[holdout] ≠ 0` (global);
2. `allow_holdout_evidence` is true;
3. `network_algorithm` is not in the supported set;
4. a disease-specific weight map assigns non-zero weight to the holdout type;
5. a disease-specific network config names an unsupported algorithm.

It does **not** remove correlations among non-holdout evidence types. Positive
genes remain visible through complementary evidence by design.
