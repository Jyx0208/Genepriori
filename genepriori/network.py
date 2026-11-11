"""RWR / PPR network propagation with degree penalty (Methods)."""

from __future__ import annotations

import math
import warnings
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy import sparse

from .evidence import min_max_normalize

Edge = Tuple[str, str, float]

# Algorithms accepted by ``propagate`` and by the LeakageAuditor whitelist.
SUPPORTED_ALGORITHMS = frozenset(
    {
        "",
        "none",
        "ppr",
        "pagerank",
        "rwr",
        "neighbor_voting",
        "hd",
        "ensemble",
    }
)


def build_adjacency(
    edges: Iterable[Edge],
    *,
    min_score: float = 0.4,
) -> Tuple[List[str], sparse.csr_matrix]:
    """Build symmetric weighted adjacency from (src, tgt, weight) edges.

    STRING combined_score >= 400 => weight in [0.4, 1.0] after /1000.
    """
    nodes: Dict[str, int] = {}
    kept: List[Edge] = []
    for a, b, w in edges:
        if w < min_score or a == b:
            continue
        for name in (a, b):
            if name not in nodes:
                nodes[name] = len(nodes)
        kept.append((a, b, float(w)))
    names = [name for name, _ in sorted(nodes.items(), key=lambda kv: kv[1])]
    mat = _adjacency_from_nodes(names, kept)
    if mat.shape[0] == 0:
        return [], sparse.csr_matrix((0, 0))
    return names, mat


def _adjacency_from_nodes(
    nodes: Sequence[str],
    edges: Iterable[Edge],
    *,
    min_score: float = 0.0,
) -> sparse.csr_matrix:
    """Symmetric weighted adjacency over a fixed, ordered node list."""
    n = len(nodes)
    if n == 0:
        return sparse.csr_matrix((0, 0))
    index = {g: i for i, g in enumerate(nodes)}
    acc: Dict[Tuple[int, int], float] = {}
    for a, b, w in edges:
        if a == b or w < min_score:
            continue
        i = index.get(a)
        j = index.get(b)
        if i is None or j is None:
            continue
        w = float(w)
        for src, tgt in ((i, j), (j, i)):
            key = (src, tgt)
            acc[key] = w if key not in acc else max(acc[key], w)
    if not acc:
        return sparse.csr_matrix((n, n))
    rows, cols, data = zip(*[(i, j, w) for (i, j), w in acc.items()])
    return sparse.csr_matrix((data, (rows, cols)), shape=(n, n))


def _column_stochastic(adj: sparse.csr_matrix) -> sparse.csr_matrix:
    """W[:, j] = A[:, j] / degree(j) (column-stochastic)."""
    deg = np.asarray(adj.sum(axis=0)).ravel()
    deg[deg == 0.0] = 1.0
    inv = sparse.diags(1.0 / deg)
    return adj @ inv


def _initial_distribution(n: int, weights: Mapping[int, float]) -> np.ndarray:
    """Normalized personalization/restart vector; uniform if no mass."""
    p0 = np.zeros(n, dtype=float)
    if n == 0:
        return p0
    for i, v in weights.items():
        if 0 <= int(i) < n:
            p0[int(i)] = max(0.0, float(v))
    total = p0.sum()
    if total <= 0:
        p0[:] = 1.0 / n
    else:
        p0 /= total
    return p0


def _warn_not_converged(name: str, max_iter: int, tol: float) -> None:
    warnings.warn(
        f"{name} did not converge within {max_iter} iterations (tol={tol:g}); "
        "returning the last iterate.",
        RuntimeWarning,
        stacklevel=3,
    )


def extract_1hop_subgraph(
    seed_genes: Sequence[str],
    edges: Iterable[Edge],
    *,
    max_neighbors_per_seed: int = 30,
    min_score: float = 0.4,
) -> Tuple[List[str], sparse.csr_matrix, Dict[str, float]]:
    """1-hop neighbourhood of seeds; at most N highest-weight neighbors per seed.

    All requested seeds are present in the returned node list, including seeds
    with no surviving edges (they get a zero row/column). Ties in the
    top-N neighbour cut are broken by gene symbol so output is deterministic.
    """
    neigh: Dict[str, Dict[str, float]] = {}
    for a, b, w in edges:
        if w < min_score or a == b:
            continue
        neigh.setdefault(a, {})
        neigh.setdefault(b, {})
        neigh[a][b] = max(neigh[a].get(b, 0.0), float(w))
        neigh[b][a] = max(neigh[b].get(a, 0.0), float(w))

    keep = set(seed_genes)
    for s in sorted(set(seed_genes)):
        if s not in neigh:
            continue
        top = sorted(neigh[s].items(), key=lambda kv: (-kv[1], kv[0]))[
            :max_neighbors_per_seed
        ]
        keep.update(n for n, _ in top)

    nodes = sorted(keep)
    if not nodes:
        return [], sparse.csr_matrix((0, 0)), {}

    sub_edges: List[Edge] = []
    for a in nodes:
        for b, w in neigh.get(a, {}).items():
            if b in keep and a < b:
                sub_edges.append((a, b, w))
    adj = _adjacency_from_nodes(nodes, sub_edges)
    raw = np.asarray(adj.sum(axis=1)).ravel()
    degrees = {nodes[i]: float(raw[i]) for i in range(len(nodes))}
    return nodes, adj, degrees


def personalized_pagerank(
    adj: sparse.csr_matrix,
    personalization: Mapping[int, float],
    *,
    damping: float = 0.85,
    tol: float = 1e-6,
    max_iter: int = 100,
) -> np.ndarray:
    """p_{t+1} = d W p_t + (1-d) p0  (personalized PageRank)."""
    n = adj.shape[0]
    p0 = _initial_distribution(n, personalization)
    if n == 0:
        return p0
    W = _column_stochastic(adj)
    p = p0.copy()
    for _ in range(max_iter):
        new_p = damping * (W @ p) + (1.0 - damping) * p0
        if np.linalg.norm(new_p - p, ord=2) < tol:
            return new_p
        p = new_p
    _warn_not_converged("personalized_pagerank", max_iter, tol)
    return p


def random_walk_with_restart(
    adj: sparse.csr_matrix,
    restart_dist: Mapping[int, float],
    *,
    restart_prob: float = 0.30,
    tol: float = 1e-6,
    max_iter: int = 100,
) -> np.ndarray:
    """p_{t+1} = (1-r) W p_t + r p0  (random walk with restart).

    Study default restart_prob = 0.30 (the reported runs used r = 0.30 with
    PPR damping 0.85; the duality r = 1 - damping is a mathematical identity
    between the two schemes, not the reported parameter pairing).
    """
    n = adj.shape[0]
    p0 = _initial_distribution(n, restart_dist)
    if n == 0:
        return p0
    W = _column_stochastic(adj)
    p = p0.copy()
    r = restart_prob
    for _ in range(max_iter):
        new_p = (1.0 - r) * (W @ p) + r * p0
        if np.linalg.norm(new_p - p, ord=2) < tol:
            return new_p
        p = new_p
    _warn_not_converged("random_walk_with_restart", max_iter, tol)
    return p


def neighbor_voting(
    adj: sparse.csr_matrix,
    seed_scores: Mapping[int, float],
    *,
    iterations: int = 1,
) -> np.ndarray:
    """Weighted neighbor voting: score(g) = sum_j A[g, j] * score(j).

    Each iteration is L1-normalized so the output is comparable with the
    propagation algorithms. ``iterations=1`` is the classical one-step vote.
    """
    n = adj.shape[0]
    if n == 0:
        return np.zeros(0, dtype=float)
    x = _initial_distribution(n, seed_scores)
    for _ in range(max(1, int(iterations))):
        x = adj @ x
        total = x.sum()
        if total > 0:
            x = x / total
    return np.asarray(x, dtype=float).ravel()


def heat_diffusion(
    adj: sparse.csr_matrix,
    seed_scores: Mapping[int, float],
    *,
    steps: int = 10,
    step_size: float = 0.25,
    tol: float = 1e-8,
) -> np.ndarray:
    """Heat-kernel diffusion on the normalized graph Laplacian.

    p_{t+1} = p_t - step_size * L p_t with
    L = I - D^{-1/2} A D^{-1/2}. Since lambda_max(L) <= 2, the explicit scheme
    is stable for ``step_size <= 0.5`` (default 0.25). The heat kernel decays
    toward zero, so a fixed number of steps is used and the result is
    L1-normalized for comparability with PPR/RWR.
    """
    n = adj.shape[0]
    if n == 0:
        return np.zeros(0, dtype=float)
    p = _initial_distribution(n, seed_scores)
    deg = np.asarray(adj.sum(axis=1)).ravel()
    deg[deg == 0.0] = 1.0
    d_inv_sqrt = sparse.diags(1.0 / np.sqrt(deg))
    laplacian = sparse.eye(n, format="csr") - d_inv_sqrt @ adj @ d_inv_sqrt
    for _ in range(max(1, int(steps))):
        new_p = p - step_size * (laplacian @ p)
        np.clip(new_p, 0.0, None, out=new_p)
        if np.linalg.norm(new_p - p, ord=2) < tol:
            p = new_p
            break
        p = new_p
    total = p.sum()
    if total > 0:
        p = p / total
    return np.asarray(p, dtype=float).ravel()


def _l1_normalize(vec: np.ndarray) -> np.ndarray:
    total = float(vec.sum())
    if total > 0:
        return vec / total
    return vec


def ensemble_propagation(
    adj: sparse.csr_matrix,
    seed_scores: Mapping[int, float],
    *,
    damping: float = 0.85,
    restart_prob: float = 0.30,
    steps: int = 10,
    step_size: float = 0.25,
) -> np.ndarray:
    """Mean of L1-normalized PPR, RWR, neighbor voting and heat diffusion."""
    n = adj.shape[0]
    if n == 0:
        return np.zeros(0, dtype=float)
    vectors = [
        personalized_pagerank(adj, seed_scores, damping=damping),
        random_walk_with_restart(adj, seed_scores, restart_prob=restart_prob),
        neighbor_voting(adj, seed_scores),
        heat_diffusion(adj, seed_scores, steps=steps, step_size=step_size),
    ]
    stacked = np.vstack([_l1_normalize(v) for v in vectors])
    return _l1_normalize(stacked.mean(axis=0))


def degree_penalty(degree: float) -> float:
    """penalty(g) = 1 / (1 + ln(degree(g) + 1))."""
    return 1.0 / (1.0 + math.log(degree + 1.0))


def blend_scores(
    direct_scores: Mapping[str, float],
    network_scores: Mapping[str, float],
    degrees: Mapping[str, float],
    *,
    blend: float = 0.7,
    network_discount: float = 0.25,
    max_network_genes: int = 80,
    normalize_network: bool = True,
) -> Dict[str, float]:
    """Combine direct evidence and network-discovered scores (Methods).

    Seeds (direct evidence > 0):
        final = blend * direct + (1-blend) * net
    Non-seeds:
        final = network_discount * net * degree_penalty
        keep top ``max_network_genes`` by this score.

    ``net`` is min-max normalized across the subgraph by default. Propagation
    outputs are probability masses on the order of ``1/|subgraph|`` while
    fused direct scores are in [0, 1]; without normalization the ``blend``
    weight would have no practical effect. Pass ``normalize_network=False`` to
    blend raw propagation values (useful for tests or custom score scales).

    Ties in the top-N cut are broken by gene symbol, so the result does not
    depend on input ordering.

    Study defaults: blend = 0.70, network_discount = 0.25,
    max_network_genes = 80 (as used for the reported results).
    """
    net = min_max_normalize(network_scores) if normalize_network else dict(network_scores)
    final: Dict[str, float] = {}
    seeds = {g for g, s in direct_scores.items() if s > 0}
    for g in sorted(seeds):
        d = float(direct_scores.get(g, 0.0))
        n = float(net.get(g, 0.0))
        final[g] = blend * d + (1.0 - blend) * n

    nonseed_cand: List[Tuple[str, float]] = []
    for g, n in net.items():
        if g in seeds:
            continue
        pen = degree_penalty(float(degrees.get(g, 0.0)))
        nonseed_cand.append((g, network_discount * float(n) * pen))
    nonseed_cand.sort(key=lambda x: (-x[1], x[0]))
    for g, s in nonseed_cand[:max_network_genes]:
        if s > 0:
            final[g] = s
    return final


def propagate(
    direct_scores: Mapping[str, float],
    edges: Iterable[Edge],
    *,
    algorithm: str = "ppr",
    damping: float = 0.85,
    restart_prob: Optional[float] = None,
    max_neighbors_per_seed: int = 30,
    min_score: float = 0.4,
    blend: float = 0.7,
    network_discount: float = 0.25,
    max_network_genes: int = 80,
    heat_steps: int = 10,
    heat_step_size: float = 0.25,
) -> Dict[str, float]:
    """Run a propagation algorithm on the 1-hop seed subgraph and blend.

    ``algorithm`` accepts ``ppr`` (alias ``pagerank``), ``rwr``,
    ``neighbor_voting``, ``hd`` (heat diffusion), ``ensemble``, or
    ``none``/``""`` (no propagation). All are accepted by the LeakageAuditor.

    Study defaults: PPR damping = 0.85; RWR restart_prob = 0.30; blend = 0.70;
    network_discount = 0.25; max_network_genes = 80.
    """
    algo = (algorithm or "").strip().lower()
    if algo == "none":
        algo = ""
    if algo not in SUPPORTED_ALGORITHMS:
        raise ValueError(
            f"Unsupported algorithm={algorithm!r}. "
            f"Expected one of {sorted(SUPPORTED_ALGORITHMS - {''})} or 'none'."
        )

    seeds = sorted(g for g, s in direct_scores.items() if s > 0)
    if not seeds or algo == "":
        return dict(direct_scores)

    names, adj, degrees = extract_1hop_subgraph(
        seeds,
        edges,
        max_neighbors_per_seed=max_neighbors_per_seed,
        min_score=min_score,
    )
    if not names or adj.nnz == 0:
        return dict(direct_scores)

    index = {g: i for i, g in enumerate(names)}
    pers = {index[g]: float(direct_scores[g]) for g in seeds if g in index}
    if algo == "rwr":
        r = 0.30 if restart_prob is None else float(restart_prob)
        vec = random_walk_with_restart(adj, pers, restart_prob=r)
    elif algo in {"ppr", "pagerank"}:
        vec = personalized_pagerank(adj, pers, damping=damping)
    elif algo == "neighbor_voting":
        vec = neighbor_voting(adj, pers)
    elif algo == "hd":
        vec = heat_diffusion(adj, pers, steps=heat_steps, step_size=heat_step_size)
    else:  # ensemble
        vec = ensemble_propagation(
            adj,
            pers,
            damping=damping,
            restart_prob=0.30 if restart_prob is None else float(restart_prob),
            steps=heat_steps,
            step_size=heat_step_size,
        )

    net = {names[i]: float(vec[i]) for i in range(len(names))}
    # Include direct-only genes that fell outside the subgraph.
    direct_full = dict(direct_scores)
    return blend_scores(
        direct_full,
        net,
        degrees,
        blend=blend,
        network_discount=network_discount,
        max_network_genes=max_network_genes,
    )
