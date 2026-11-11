"""Open Targets Platform GraphQL client and disease evidence fetching.

The client talks to ``https://api.platform.opentargets.org/api/v4/graphql``
(no API key required) and paginates over ``disease.associatedTargets``.
Datatype scores are mapped to the six standardized evidence types from
Table 3 of the manuscript; datasource scores are ignored by default because
they are finer-grained and already aggregated into datatypes.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import requests

from ..evidence import OT_TO_STANDARD, STANDARD_TYPES, map_ot_type
from .cache import Cache, utc_now
from .registry import Disease

OT_GRAPHQL_URL = "https://api.platform.opentargets.org/api/v4/graphql"
USER_AGENT = "genepriori/0.2 (+https://github.com/Jyx0208/Genepriori)"
DEFAULT_PAGE_SIZE = 500

DISEASE_TARGETS_QUERY = """
query DiseaseTargets($efoId: String!, $index: Int!, $size: Int!) {
  disease(efoId: $efoId) {
    id
    name
    associatedTargets(page: {index: $index, size: $size}) {
      count
      rows {
        score
        target {
          id
          approvedSymbol
        }
        datatypeScores {
          id
          score
        }
        datasourceScores {
          id
          score
        }
      }
    }
  }
}
"""

DISEASE_SEARCH_QUERY = """
query SearchDisease($q: String!, $size: Int!) {
  search(queryString: $q, entityNames: ["disease"], page: {index: 0, size: $size}) {
    hits {
      id
      name
      entity
    }
  }
}
"""


class OpenTargetsError(RuntimeError):
    """Raised when the Open Targets API cannot satisfy a request."""


class _RateLimiter:
    """Simple minimum-interval limiter (default 5 requests/second)."""

    def __init__(self, rate_per_second: float) -> None:
        self.min_interval = 0.0 if rate_per_second <= 0 else 1.0 / float(rate_per_second)
        self._last = 0.0

    def wait(self) -> None:
        if self.min_interval <= 0:
            return
        now = time.monotonic()
        delta = now - self._last
        if delta < self.min_interval:
            time.sleep(self.min_interval - delta)
        self._last = time.monotonic()


@dataclass
class DiseaseEvidence:
    """Gene-disease evidence for one disease, keyed by gene symbol."""

    disease_id: str
    disease_name: str
    gene_type_scores: Dict[str, Dict[str, float]]
    ensembl_ids: Dict[str, str] = field(default_factory=dict)
    n_rows: int = 0
    total_associations: Optional[int] = None
    fetched_at: str = field(default_factory=utc_now)
    source: str = "Open Targets Platform"

    @property
    def genes(self) -> List[str]:
        return sorted(self.gene_type_scores)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": 1,
            "disease_id": self.disease_id,
            "disease_name": self.disease_name,
            "gene_type_scores": self.gene_type_scores,
            "ensembl_ids": self.ensembl_ids,
            "n_rows": self.n_rows,
            "total_associations": self.total_associations,
            "fetched_at": self.fetched_at,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DiseaseEvidence":
        return cls(
            disease_id=str(payload["disease_id"]),
            disease_name=str(payload.get("disease_name") or payload["disease_id"]),
            gene_type_scores={
                str(gene): {str(t): float(v) for t, v in tmap.items()}
                for gene, tmap in (payload.get("gene_type_scores") or {}).items()
            },
            ensembl_ids={str(k): str(v) for k, v in (payload.get("ensembl_ids") or {}).items()},
            n_rows=int(payload.get("n_rows") or 0),
            total_associations=payload.get("total_associations"),
            fetched_at=str(payload.get("fetched_at") or ""),
            source=str(payload.get("source") or "Open Targets Platform"),
        )

    def cache_key(self) -> str:
        return f"diseases/{self.disease_id}.json"


def rows_to_evidence(
    rows: Iterable[Mapping[str, Any]],
    *,
    include_datasources: bool = False,
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, str]]:
    """Convert GraphQL association rows into gene -> {standard type -> score}.

    Fine-grained datatype scores are mapped with ``OT_TO_STANDARD`` and the
    maximum score per standardized type is retained, matching Table 3.
    Genes without any mapped evidence are dropped.
    """
    gene_type_scores: Dict[str, Dict[str, float]] = {}
    ensembl_ids: Dict[str, str] = {}
    for row in rows:
        target = row.get("target") or {}
        symbol = str(target.get("approvedSymbol") or "").strip()
        if not symbol:
            continue
        scores: Dict[str, float] = {}
        blocks: List[Sequence[Mapping[str, Any]]] = [row.get("datatypeScores") or []]
        if include_datasources:
            blocks.append(row.get("datasourceScores") or [])
        for block in blocks:
            for item in block:
                std = map_ot_type(str(item.get("id") or ""))
                if std is None:
                    continue
                score = float(item.get("score") or 0.0)
                if std not in scores or score > scores[std]:
                    scores[std] = score
        if not scores:
            continue
        slot = gene_type_scores.setdefault(symbol, {})
        for std, score in scores.items():
            if std not in slot or score > slot[std]:
                slot[std] = score
        if target.get("id"):
            ensembl_ids[symbol] = str(target["id"])
    return gene_type_scores, ensembl_ids


class OpenTargetsClient:
    """Minimal, dependency-light GraphQL client with rate limiting and retries."""

    def __init__(
        self,
        *,
        url: str = OT_GRAPHQL_URL,
        rate_limit: float = 5.0,
        timeout: float = 120.0,
        max_retries: int = 3,
        backoff: float = 1.0,
        session: Optional[requests.Session] = None,
    ) -> None:
        self.url = url
        self.timeout = timeout
        self.max_retries = max(0, int(max_retries))
        self.backoff = backoff
        self.session = session or requests.Session()
        self._limiter = _RateLimiter(rate_limit)

    # ------------------------------------------------------------------ GraphQL
    def query(self, query: str, variables: Mapping[str, Any]) -> Dict[str, Any]:
        payload = {"query": query, "variables": dict(variables)}
        last_error: Optional[BaseException] = None
        for attempt in range(self.max_retries + 1):
            self._limiter.wait()
            try:
                response = self.session.post(
                    self.url,
                    json=payload,
                    timeout=self.timeout,
                    headers={"User-Agent": USER_AGENT},
                )
                if response.status_code in {429, 500, 502, 503, 504}:
                    raise OpenTargetsError(f"HTTP {response.status_code} from Open Targets")
                response.raise_for_status()
                body = response.json()
            except (requests.RequestException, ValueError, OpenTargetsError) as exc:
                last_error = exc
                if attempt < self.max_retries:
                    time.sleep(self.backoff * (2 ** attempt))
                    continue
                raise OpenTargetsError(
                    f"Open Targets request failed after {attempt + 1} attempt(s): {last_error}"
                ) from exc

            errors = body.get("errors")
            if errors:
                message = "; ".join(str(e.get("message", e)) for e in errors[:3])
                raise OpenTargetsError(f"Open Targets GraphQL error: {message}")
            return body.get("data") or {}
        raise OpenTargetsError("Unreachable")  # pragma: no cover

    # ------------------------------------------------------------------- search
    def search_diseases(self, text: str, *, limit: int = 10) -> List[Dict[str, str]]:
        data = self.query(DISEASE_SEARCH_QUERY, {"q": text, "size": int(limit)})
        hits = ((data.get("search") or {}).get("hits") or [])
        return [
            {"id": str(h["id"]), "name": str(h.get("name") or h["id"]), "entity": str(h.get("entity", ""))}
            for h in hits
            if h.get("id")
        ]

    # -------------------------------------------------------------- association
    def fetch_associations(
        self,
        disease_id: str,
        *,
        max_genes: int = 2000,
        page_size: int = DEFAULT_PAGE_SIZE,
        include_datasources: bool = False,
    ) -> DiseaseEvidence:
        if max_genes <= 0:
            raise ValueError(f"max_genes must be positive, got {max_genes}.")
        page_size = max(1, min(int(page_size), int(max_genes)))
        rows: List[Mapping[str, Any]] = []
        disease_name = disease_id
        total: Optional[int] = None
        index = 0

        while len(rows) < max_genes:
            size = min(page_size, max_genes - len(rows))
            data = self.query(
                DISEASE_TARGETS_QUERY,
                {"efoId": disease_id, "index": index, "size": size},
            )
            disease = data.get("disease")
            if disease is None:
                raise OpenTargetsError(
                    f"Disease {disease_id!r} was not found in the current Open Targets "
                    "release. Check the identifier with `genepriori search-disease`."
                )
            disease_name = str(disease.get("name") or disease_name)
            associated = disease.get("associatedTargets") or {}
            if total is None and associated.get("count") is not None:
                total = int(associated["count"])
            page = associated.get("rows") or []
            if not page:
                break
            rows.extend(page)
            index += 1
            if total is not None and len(rows) >= total:
                break

        gene_type_scores, ensembl_ids = rows_to_evidence(
            rows[:max_genes], include_datasources=include_datasources
        )
        return DiseaseEvidence(
            disease_id=str(data.get("disease", {}).get("id") or disease_id),
            disease_name=disease_name,
            gene_type_scores=gene_type_scores,
            ensembl_ids=ensembl_ids,
            n_rows=len(rows[:max_genes]),
            total_associations=total,
        )


def fetch_disease_evidence(
    disease: Disease | str,
    *,
    client: Optional[OpenTargetsClient] = None,
    cache: Optional[Cache] = None,
    max_genes: int = 2000,
    page_size: int = DEFAULT_PAGE_SIZE,
    refresh: bool = False,
    offline: bool = False,
    max_age_days: Optional[float] = None,
    include_datasources: bool = False,
) -> DiseaseEvidence:
    """Fetch (or load from cache) evidence for one disease.

    ``offline=True`` never touches the network and raises ``FileNotFoundError``
    when the disease has not been fetched before.
    """
    if isinstance(disease, str):
        from .registry import resolve_disease

        disease = resolve_disease(disease)
    cache = cache if cache is not None else Cache()
    key = f"diseases/{disease.disease_id}.json"

    if not refresh and cache.exists(key):
        if not offline or max_age_days is None:
            cached = cache.info(key) or {}
            same_budget = cached.get("max_genes") in (None, max_genes)
            fresh = max_age_days is None or cache.is_fresh(key, max_age_days=max_age_days)
            if same_budget and fresh:
                return DiseaseEvidence.from_dict(cache.read_json(key))
            if offline:
                # Offline and the cached payload is usable under a different
                # budget: return it rather than failing.
                return DiseaseEvidence.from_dict(cache.read_json(key))

    if offline:
        raise FileNotFoundError(
            f"No cached evidence for {disease.disease_id}. "
            f"Run `genepriori fetch --disease {disease.disease_id}` first."
        )

    client = client or OpenTargetsClient()
    evidence = client.fetch_associations(
        disease.disease_id,
        max_genes=max_genes,
        page_size=page_size,
        include_datasources=include_datasources,
    )
    if not evidence.disease_name or evidence.disease_name == disease.disease_id:
        evidence.disease_name = disease.name
    cache.write_json(evidence.to_dict(), key)
    cache.record(
        key,
        url=OT_GRAPHQL_URL,
        extra={
            "disease_id": disease.disease_id,
            "max_genes": max_genes,
            "n_genes": len(evidence.gene_type_scores),
            "n_rows": evidence.n_rows,
            "total_associations": evidence.total_associations,
        },
    )
    return evidence


def evidence_summary(evidence: DiseaseEvidence) -> Dict[str, Any]:
    """Small dict suitable for logs and ``--json`` output."""
    counts = {t: 0 for t in STANDARD_TYPES}
    for tmap in evidence.gene_type_scores.values():
        for t in tmap:
            counts[t] = counts.get(t, 0) + 1
    return {
        "disease_id": evidence.disease_id,
        "disease_name": evidence.disease_name,
        "genes": len(evidence.gene_type_scores),
        "rows": evidence.n_rows,
        "total_associations": evidence.total_associations,
        "genes_with_evidence_per_type": counts,
        "fetched_at": evidence.fetched_at,
    }


__all__ = [
    "OT_GRAPHQL_URL",
    "OpenTargetsError",
    "OpenTargetsClient",
    "DiseaseEvidence",
    "fetch_disease_evidence",
    "rows_to_evidence",
    "evidence_summary",
    "OT_TO_STANDARD",
]
