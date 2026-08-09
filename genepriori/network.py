"""RWR / PPR network propagation with degree penalty (Methods)."""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
from scipy import sparse


Edge = Tuple[str, str, float]


def build_adjacency(
    edges: Iterable[Edge],
    *,
    min_score: float = 0.4,
) -> Tuple[List[str], sparse.csr_matrix]:
    """Build symmetric weighted adjacency from (src, tgt, weight) edges.

    Manuscript: STRING combined_score >= 400 => weight in [0.4, 1.0] after /1000.
    """
    nodes: Dict[str, int] = {}
    acc: Dict[Tuple[int, int], float] = {}
    for a, b, w in edges:
        if w < min_score or a == b:
            continue
        for name in (a, b):
            if name not in nodes:
                nodes[name] = len(nodes)
        i, j = nodes[a], nodes[b]
        for src, tgt in ((i, j), (j, i)):
            key = (src, tgt)
            acc[key] = float(w) if key not in acc else max(acc[key], float(w))
    n = len(nodes)
    if n == 0 or not acc:
        return [], sparse.csr_matrix((0, 0))
    rows, cols, data = zip(*[(i, j, w) for (i, j), w in acc.items()])
    mat = sparse.csr_matrix((data, (rows, cols)), shape=(n, n))
    names: List[Optional[str]] = [None] * n
    for name, idx in nodes.items():
        names[idx] = name
    return [str(x) for x in names], mat

def _column_stochastic(adj: sparse.csr_matrix) -> sparse.csr_matrix:
    """W[:, j] = A[:, j] / degree(j) (column-stochastic)."""
    deg = np.asarray(adj.sum(axis=0)).ravel()
    deg[deg == 0.0] = 1.0
    inv = sparse.diags(1.0 / deg)
    return adj @ inv


def extract_1hop_subgraph(
    seed_genes: Sequence[str],
    edges: Iterable[Edge],
    *,
    max_neighbors_per_seed: int = 30,
    min_score: float = 0.4,
) -> Tuple[List[str], sparse.csr_matrix, Dict[str, float]]:
    """1-hop neighbourhood of seeds; at most N highest-weight neighbors per seed."""
    # adjacency list
    neigh: Dict[str, Dict[str, float]] = {}
    for a, b, w in edges:
        if w < min_score or a == b:
            continue
        neigh.setdefault(a, {})
        neigh.setdefault(b, {})
        neigh[a][b] = w if b not in neigh[a] else max(neigh[a][b], w)
        neigh[b][a] = w if a not in neigh[b] else max(neigh[b][a], w)

    keep = set(seed_genes)
    for s in seed_genes:
        if s not in neigh:
            continue
        top = sorted(neigh[s].items(), key=lambda x: x[1], reverse=True)[:max_neighbors_per_seed]
        keep.update(n for n, _ in top)

    sub_edges: List[Edge] = []
    degrees: Dict[str, float] = {g: 0.0 for g in keep}
    for a in keep:
        for b, w in neigh.get(a, {}).items():
            if b in keep and a < b:
                sub_edges.append((a, b, w))
                degrees[a] += w
                degrees[b] += w
    names, adj = build_adjacency(sub_edges, min_score=min_score)
    # recompute degrees on subgraph nodes
    deg_out = {n: 0.0 for n in names}
    if names:
        raw = np.asarray(adj.sum(axis=1)).ravel()
        for i, n in enumerate(names):
            deg_out[n] = float(raw[i])
    return names, adj, deg_out


def personalized_pagerank(
    adj: sparse.csr_matrix,
    personalization: Mapping[int, float],
    *,
    damping: float = 0.85,
    tol: float = 1e-6,
    max_iter: int = 100,
) -> np.ndarray:
    """p_{t+1} = d W p_t + (1-d) p0  (manuscript PPR)."""
    n = adj.shape[0]
    p0 = np.zeros(n, dtype=float)
    for i, v in personalization.items():
        p0[i] = float(v)
    s = p0.sum()
    if s <= 0:
        p0[:] = 1.0 / max(n, 1)
    else:
        p0 /= s
    if n == 0:
        return p0
    W = _column_stochastic(adj)
    p = p0.copy()
    for _ in range(max_iter):
        new_p = damping * (W @ p) + (1.0 - damping) * p0
        if np.linalg.norm(new_p - p, ord=2) < tol:
            return new_p
        p = new_p
    return p


def random_walk_with_restart(
    adj: sparse.csr_matrix,
    restart_dist: Mapping[int, float],
    *,
    restart_prob: float = 0.30,
    tol: float = 1e-6,
    max_iter: int = 100,
) -> np.ndarray:
    """p_{t+1} = (1-r) W p_t + r p0  (manuscript RWR).

    Study default restart_prob = 0.30 (the reported runs used r = 0.30 with
    PPR damping 0.85; the duality r = 1 - damping is a mathematical identity
    between the two schemes, not the reported parameter pairing).
    """
    n = adj.shape[0]
    p0 = np.zeros(n, dtype=float)
    for i, v in restart_dist.items():
        p0[i] = float(v)
    s = p0.sum()
    if s <= 0:
        p0[:] = 1.0 / max(n, 1)
    else:
        p0 /= s
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
    return p


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
) -> Dict[str, float]:
    """Combine direct evidence and network-discovered scores (Methods).

    Seeds (direct evidence > 0):
        final = blend * direct + (1-blend) * net
    Non-seeds:
        final = network_discount * net * degree_penalty
        keep top ``max_network_genes`` by this score.

    Study defaults: blend = 0.70, network_discount = 0.25,
    max_network_genes = 80 (as used for the reported results).
    """
    final: Dict[str, float] = {}
    seeds = {g for g, s in direct_scores.items() if s > 0}
    for g in seeds:
        d = float(direct_scores.get(g, 0.0))
        n = float(network_scores.get(g, 0.0))
        final[g] = blend * d + (1.0 - blend) * n

    nonseed_cand: List[Tuple[str, float]] = []
    for g, n in network_scores.items():
        if g in seeds:
            continue
        pen = degree_penalty(float(degrees.get(g, 0.0)))
        nonseed_cand.append((g, network_discount * float(n) * pen))
    nonseed_cand.sort(key=lambda x: x[1], reverse=True)
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
) -> Dict[str, float]:
    """Run PPR or RWR on the 1-hop seed subgraph and blend with direct scores.

    Study defaults: PPR damping = 0.85; RWR restart_prob = 0.30; blend = 0.70;
    network_discount = 0.25; max_network_genes = 80.
    """
    seeds = [g for g, s in direct_scores.items() if s > 0]
    if not seeds:
        return dict(direct_scores)

    names, adj, degrees = extract_1hop_subgraph(
        seeds,
        edges,
        max_neighbors_per_seed=max_neighbors_per_seed,
        min_score=min_score,
    )
    if not names:
        return dict(direct_scores)

    index = {g: i for i, g in enumerate(names)}
    pers = {index[g]: float(direct_scores[g]) for g in seeds if g in index}
    algo = algorithm.strip().lower()
    if algo == "rwr":
        r = 0.30 if restart_prob is None else float(restart_prob)
        vec = random_walk_with_restart(adj, pers, restart_prob=r)
    elif algo in {"ppr", "pagerank"}:
        vec = personalized_pagerank(adj, pers, damping=damping)
    else:
        raise ValueError(f"Unsupported algorithm={algorithm!r} (use 'ppr' or 'rwr').")

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
