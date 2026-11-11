"""Built-in disease registry and disease-name resolution.

Disease identifiers follow the current Open Targets release, which uses
MONDO as the primary ontology (legacy EFO identifiers are increasingly
retired). Any EFO/MONDO identifier can be used directly on the command line;
the registry is a convenience list of 19 diseases spanning oncology,
neurodegeneration, psychiatry, autoimmunity, respiratory, cardiovascular,
metabolic and renal categories.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

_ID_PATTERN = re.compile(r"^(EFO|MONDO|ORPHANET|HP)_\d+$", re.IGNORECASE)


@dataclass(frozen=True)
class Disease:
    """A disease entry usable by the pipeline."""

    key: str
    name: str
    disease_id: str
    category: str = "other"
    aliases: Tuple[str, ...] = field(default_factory=tuple)

    @property
    def label(self) -> str:
        return f"{self.name} ({self.disease_id})"


DEFAULT_DISEASES: Tuple[Disease, ...] = (
    # Oncology: the seven solid tumours anchoring the translational focus.
    Disease("breast_carcinoma", "breast carcinoma", "MONDO_0004989", "oncology", ("breast cancer",)),
    Disease("colorectal_cancer", "colorectal cancer", "MONDO_0005575", "oncology", ("crc", "colon cancer")),
    Disease("lung_carcinoma", "lung carcinoma", "MONDO_0005138", "oncology", ("lung cancer",)),
    Disease("prostate_carcinoma", "prostate carcinoma", "MONDO_0005159", "oncology", ("prostate cancer",)),
    Disease(
        "pancreatic_carcinoma",
        "pancreatic carcinoma",
        "MONDO_0005192",
        "oncology",
        ("pancreatic cancer", "exocrine pancreatic carcinoma"),
    ),
    Disease("gastric_carcinoma", "gastric carcinoma", "MONDO_0004950", "oncology", ("stomach cancer",)),
    Disease(
        "hepatocellular_carcinoma",
        "hepatocellular carcinoma",
        "MONDO_0007256",
        "oncology",
        ("hcc", "liver cancer"),
    ),
    # Neurodegenerative
    Disease("alzheimer_disease", "Alzheimer disease", "MONDO_0004975", "neurodegenerative", ("ad",)),
    Disease("parkinson_disease", "Parkinson disease", "MONDO_0005180", "neurodegenerative", ("pd",)),
    # Psychiatric
    Disease("schizophrenia", "schizophrenia", "MONDO_0005090", "psychiatric", ("scz",)),
    # Autoimmune / inflammatory
    Disease("rheumatoid_arthritis", "rheumatoid arthritis", "MONDO_0008383", "autoimmune", ("ra",)),
    Disease(
        "systemic_lupus_erythematosus",
        "systemic lupus erythematosus",
        "MONDO_0007915",
        "autoimmune",
        ("sle", "lupus"),
    ),
    Disease(
        "inflammatory_bowel_disease",
        "inflammatory bowel disease",
        "MONDO_0005265",
        "autoimmune",
        ("ibd",),
    ),
    Disease(
        "multiple_sclerosis",
        "multiple sclerosis",
        "MONDO_0005314",
        "autoimmune",
        # The generic multiple sclerosis node is not indexed in the current
        # Open Targets release; relapsing-remitting MS is used instead.
        ("ms", "relapsing-remitting multiple sclerosis"),
    ),
    # Respiratory
    Disease("asthma", "asthma", "MONDO_0004979", "respiratory"),
    Disease(
        "copd",
        "chronic obstructive pulmonary disease",
        "MONDO_0005002",
        "respiratory",
        ("copd",),
    ),
    # Cardiovascular
    Disease(
        "coronary_artery_disease",
        "coronary artery disease",
        "MONDO_0005010",
        "cardiovascular",
        ("cad", "coronary artery disorder"),
    ),
    # Metabolic
    Disease(
        "type_2_diabetes",
        "type 2 diabetes mellitus",
        "MONDO_0005148",
        "metabolic",
        ("t2dm", "type 2 diabetes"),
    ),
    # Renal
    Disease(
        "chronic_kidney_disease",
        "chronic kidney disease",
        "MONDO_0005300",
        "renal",
        ("ckd",),
    ),
)


def is_disease_id(value: str) -> bool:
    """Whether ``value`` looks like an ontology identifier."""
    return bool(_ID_PATTERN.match((value or "").strip()))


def list_diseases() -> List[Disease]:
    """All built-in registry entries."""
    return list(DEFAULT_DISEASES)


def diseases_by_category() -> Dict[str, List[Disease]]:
    grouped: Dict[str, List[Disease]] = {}
    for disease in DEFAULT_DISEASES:
        grouped.setdefault(disease.category, []).append(disease)
    return grouped


def _normalize(text: str) -> str:
    return re.sub(r"[\s_\-]+", " ", (text or "").strip().lower())


def _match(disease: Disease, needle: str) -> bool:
    candidates = {_normalize(disease.key), _normalize(disease.name), _normalize(disease.disease_id)}
    candidates.update(_normalize(alias) for alias in disease.aliases)
    return needle in candidates


def find_disease(query: str) -> Optional[Disease]:
    """Look up a registry entry by key, name, alias or exact identifier."""
    needle = _normalize(query)
    if not needle:
        return None
    for disease in DEFAULT_DISEASES:
        if _match(disease, needle):
            return disease
    # Substring fallback, but only for reasonably specific queries.
    if len(needle) >= 4:
        partial = [
            d
            for d in DEFAULT_DISEASES
            if needle in _normalize(d.name) or needle in _normalize(d.key)
        ]
        if len(partial) == 1:
            return partial[0]
    return None


def resolve_disease(
    query: str,
    *,
    client=None,
    allow_network: bool = False,
) -> Disease:
    """Resolve a registry name, an ontology ID, or (optionally) a free-text name.

    Raises ``ValueError`` when the query cannot be resolved. When
    ``allow_network`` is true and ``client`` is an
    :class:`~genepriori.data.opentargets.OpenTargetsClient`, the Open Targets
    search API is consulted as a last resort.
    """
    text = (query or "").strip()
    if not text:
        raise ValueError("Empty disease query.")

    if is_disease_id(text):
        for disease in DEFAULT_DISEASES:
            if disease.disease_id.lower() == text.lower():
                return disease
        return Disease(key=text, name=text, disease_id=text.upper())

    match = find_disease(text)
    if match is not None:
        return match

    if allow_network and client is not None:
        hits = client.search_diseases(text, limit=5)
        if hits:
            best = hits[0]
            return Disease(
                key=str(best.get("id", "")),
                name=str(best.get("name", text)),
                disease_id=str(best.get("id", "")),
                category="resolved",
            )

    known = ", ".join(d.name for d in DEFAULT_DISEASES[:6])
    raise ValueError(
        f"Unknown disease {query!r}. Use an ontology ID (e.g. MONDO_0004989) "
        f"or one of the built-in names ({known}, ...). "
        "Run `genepriori list-diseases` for the full list."
    )


def iter_registry(entries: Optional[Iterable[str]] = None) -> List[Disease]:
    """Resolve a list of registry keys/IDs; ``None`` means the full registry."""
    if entries is None:
        return list(DEFAULT_DISEASES)
    return [resolve_disease(entry) for entry in entries]
