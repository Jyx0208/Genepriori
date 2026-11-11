"""STRING protein-protein interaction download and parsing.

Downloads the human (taxid 9606) network in STRING's bulk format, maps STRING
protein identifiers to preferred gene names, filters by ``combined_score`` and
caches the resulting edge list as a compact gzipped TSV so that later runs do
not re-parse a multi-hundred-megabyte file.

STRING v12.5 is the current release; older archives such as v12.0 remain
available and can be selected with ``version="12.0"``.
"""

from __future__ import annotations

import gzip
import os
import tempfile
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import requests

from ..network import Edge
from .cache import Cache, sha256_file, utc_now

STRING_BASE_URL = "https://stringdb-downloads.org/download"
DEFAULT_STRING_VERSION = "12.5"
DEFAULT_TAXID = 9606
DEFAULT_MIN_COMBINED_SCORE = 400

ProgressCallback = Optional[Callable[[int], None]]


class StringError(RuntimeError):
    """Raised when STRING data cannot be downloaded or parsed."""


def links_url(version: str = DEFAULT_STRING_VERSION, taxid: int = DEFAULT_TAXID) -> str:
    return f"{STRING_BASE_URL}/protein.links.v{version}/{taxid}.protein.links.v{version}.txt.gz"


def info_url(version: str = DEFAULT_STRING_VERSION, taxid: int = DEFAULT_TAXID) -> str:
    return f"{STRING_BASE_URL}/protein.info.v{version}/{taxid}.protein.info.v{version}.txt.gz"


def raw_links_key(version: str = DEFAULT_STRING_VERSION, taxid: int = DEFAULT_TAXID) -> str:
    return f"raw/string/v{version}/{taxid}.protein.links.v{version}.txt.gz"


def raw_info_key(version: str = DEFAULT_STRING_VERSION, taxid: int = DEFAULT_TAXID) -> str:
    return f"raw/string/v{version}/{taxid}.protein.info.v{version}.txt.gz"


def edges_cache_key(
    version: str = DEFAULT_STRING_VERSION,
    taxid: int = DEFAULT_TAXID,
    min_combined_score: int = DEFAULT_MIN_COMBINED_SCORE,
) -> str:
    return f"ppi/v{version}/{taxid}.edges.min{min_combined_score}.tsv.gz"


def download_file(
    url: str,
    dest: Path,
    *,
    session: Optional[requests.Session] = None,
    timeout: float = 600.0,
    chunk_size: int = 1 << 20,
    progress: ProgressCallback = None,
) -> Path:
    """Stream ``url`` to ``dest`` atomically, reporting downloaded bytes."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    session = session or requests.Session()
    fd, tmp_name = tempfile.mkstemp(dir=str(dest.parent), prefix=".download-", suffix=".part")
    os.close(fd)
    downloaded = 0
    try:
        with session.get(url, stream=True, timeout=timeout) as response:
            if response.status_code != 200:
                raise StringError(f"HTTP {response.status_code} while downloading {url}")
            total = int(response.headers.get("Content-Length") or 0)
            with open(tmp_name, "wb") as handle:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if not chunk:
                        continue
                    handle.write(chunk)
                    downloaded += len(chunk)
                    if progress is not None:
                        progress(downloaded if total <= 0 else downloaded * 100 // total)
        os.replace(tmp_name, dest)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise
    return dest


def download_string(
    *,
    version: str = DEFAULT_STRING_VERSION,
    taxid: int = DEFAULT_TAXID,
    cache: Optional[Cache] = None,
    refresh: bool = False,
    offline: bool = False,
    session: Optional[requests.Session] = None,
    progress: ProgressCallback = None,
) -> Tuple[Path, Path]:
    """Ensure raw links and info files are cached; return their paths."""
    cache = cache if cache is not None else Cache()
    links_key = raw_links_key(version, taxid)
    info_key = raw_info_key(version, taxid)
    links_path = cache.path(links_key)
    info_path = cache.path(info_key)

    if offline:
        missing = [k for k in (links_key, info_key) if not cache.exists(k)]
        if missing:
            raise FileNotFoundError(
                "Offline mode: missing STRING file(s) "
                + ", ".join(missing)
                + ". Run `genepriori fetch-ppi` once with network access."
            )
        return links_path, info_path

    for key, url, path in (
        (links_key, links_url(version, taxid), links_path),
        (info_key, info_url(version, taxid), info_path),
    ):
        if path.exists() and not refresh:
            continue
        download_file(url, path, session=session, progress=progress)
        cache.record(key, url=url, extra={"version": version, "taxid": taxid})
    return links_path, info_path


def load_id_map(info_path: Path) -> Dict[str, str]:
    """Read ``protein.info`` and map STRING protein ID -> preferred gene name."""
    mapping: Dict[str, str] = {}
    with gzip.open(info_path, "rt", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line or line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            protein_id, preferred = parts[0].strip(), parts[1].strip()
            if protein_id and preferred:
                mapping[protein_id] = preferred
    return mapping


def build_edges(
    links_path: Path,
    id_map: Dict[str, str],
    *,
    min_combined_score: int = DEFAULT_MIN_COMBINED_SCORE,
    progress: ProgressCallback = None,
) -> List[Edge]:
    """Parse ``protein.links`` into gene-symbol edges with weights in [0, 1].

    STRING scores are integers in [0, 1000]; the weight is
    ``combined_score / 1000`` so a threshold of 400 becomes weight 0.4.
    Edges whose proteins have no gene symbol are skipped. The output is
    symmetric and deduplicated: each undirected pair appears once (lexico-
    graphically ordered) with its maximum score, which is the representation
    ``network.build_adjacency`` expects.
    """
    accumulated: Dict[Tuple[str, str], float] = {}
    lines = 0
    with gzip.open(links_path, "rt", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            lines += 1
            if not line:
                continue
            parts = line.rstrip("\n").split()
            if len(parts) < 3:
                continue
            if parts[0] == "protein1":
                continue
            p1, p2, raw_score = parts[0], parts[1], parts[2]
            try:
                score = int(float(raw_score))
            except ValueError:
                continue
            if score < min_combined_score:
                continue
            g1 = id_map.get(p1)
            g2 = id_map.get(p2)
            if not g1 or not g2 or g1 == g2:
                continue
            if g2 < g1:
                g1, g2 = g2, g1
            key = (g1, g2)
            weight = score / 1000.0
            previous = accumulated.get(key)
            if previous is None or weight > previous:
                accumulated[key] = weight
            if progress is not None and lines % 100_000 == 0:
                progress(len(accumulated))
    if progress is not None:
        progress(len(accumulated))
    return [(a, b, w) for (a, b), w in sorted(accumulated.items())]


def write_edges_cache(path: Path, edges: Iterable[Edge]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=".edges-", suffix=".tsv.gz")
    os.close(fd)
    try:
        with gzip.open(tmp_name, "wt", encoding="utf-8", compresslevel=6) as handle:
            handle.write("gene1\tgene2\tweight\n")
            for a, b, w in edges:
                handle.write(f"{a}\t{b}\t{w:.4f}\n")
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def read_edges_cache(path: Path) -> List[Edge]:
    edges: List[Edge] = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        header = handle.readline()
        if not header.startswith("gene1"):
            handle.seek(0)
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            edges.append((parts[0], parts[1], float(parts[2])))
    return edges


def load_string_edges(
    *,
    version: str = DEFAULT_STRING_VERSION,
    taxid: int = DEFAULT_TAXID,
    min_combined_score: int = DEFAULT_MIN_COMBINED_SCORE,
    cache: Optional[Cache] = None,
    refresh: bool = False,
    offline: bool = False,
    session: Optional[requests.Session] = None,
    progress: ProgressCallback = None,
) -> List[Edge]:
    """Return the human PPI edge list, using cached downloads and parsed edges."""
    cache = cache if cache is not None else Cache()
    edges_key = edges_cache_key(version, taxid, min_combined_score)
    if cache.exists(edges_key) and not refresh:
        return read_edges_cache(cache.path(edges_key))

    links_path, info_path = download_string(
        version=version,
        taxid=taxid,
        cache=cache,
        refresh=refresh,
        offline=offline,
        session=session,
        progress=progress,
    )
    id_map = load_id_map(info_path)
    edges = build_edges(
        links_path,
        id_map,
        min_combined_score=min_combined_score,
        progress=progress,
    )
    write_edges_cache(cache.path(edges_key), edges)
    cache.record(
        edges_key,
        url=links_url(version, taxid),
        extra={
            "version": version,
            "taxid": taxid,
            "min_combined_score": min_combined_score,
            "n_edges": len(edges),
            "built_at": utc_now(),
            "source_sha256": sha256_file(links_path) if links_path.exists() else None,
        },
    )
    return edges


def ppi_summary(edges: List[Edge]) -> Dict[str, float]:
    """Basic network statistics for logs and ``doctor`` output."""
    nodes = set()
    for a, b, _ in edges:
        nodes.add(a)
        nodes.add(b)
    weights = [w for _, _, w in edges]
    return {
        "edges": float(len(edges)),
        "nodes": float(len(nodes)),
        "mean_weight": float(sum(weights) / len(weights)) if weights else 0.0,
        "min_weight": float(min(weights)) if weights else 0.0,
        "max_weight": float(max(weights)) if weights else 0.0,
    }
