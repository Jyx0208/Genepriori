"""Tests for the data layer: cache, registry, Open Targets, STRING."""

from __future__ import annotations

import gzip
import json
import math
from pathlib import Path

import pytest
import requests

from genepriori.data.cache import Cache, sha256_file
from genepriori.data.opentargets import (
    DiseaseEvidence,
    OpenTargetsClient,
    OpenTargetsError,
    evidence_summary,
    fetch_disease_evidence,
    rows_to_evidence,
)
from genepriori.data.registry import (
    DEFAULT_DISEASES,
    find_disease,
    is_disease_id,
    resolve_disease,
)
from genepriori.data.string import (
    build_edges,
    download_string,
    edges_cache_key,
    load_id_map,
    read_edges_cache,
    write_edges_cache,
)


# --------------------------------------------------------------------- cache
def test_cache_json_roundtrip(tmp_path: Path) -> None:
    cache = Cache(tmp_path / "cache")
    cache.write_json({"a": 1}, "x", "y.json")
    assert cache.read_json("x", "y.json") == {"a": 1}


def test_cache_manifest_records_sha(tmp_path: Path) -> None:
    cache = Cache(tmp_path / "cache")
    cache.write_json({"a": 1}, "x.json")
    entry = cache.record("x.json", url="https://example.org/x", extra={"k": "v"})
    assert entry["url"] == "https://example.org/x"
    assert entry["k"] == "v"
    assert entry["sha256"] == sha256_file(cache.path("x.json"))
    assert cache.info("x.json")["sha256"] == entry["sha256"]


def test_cache_missing_manifest_is_empty(tmp_path: Path) -> None:
    cache = Cache(tmp_path / "cache")
    assert cache.manifest() == {}


# ------------------------------------------------------------------ registry
def test_registry_has_nineteen_diseases() -> None:
    assert len(DEFAULT_DISEASES) == 19


def test_registry_categories_cover_paper() -> None:
    categories = {d.category for d in DEFAULT_DISEASES}
    assert {"oncology", "neurodegenerative", "psychiatric", "autoimmune", "respiratory",
            "cardiovascular", "metabolic", "renal"} <= categories
    assert sum(1 for d in DEFAULT_DISEASES if d.category == "oncology") == 7


def test_resolve_by_key_name_and_alias() -> None:
    assert resolve_disease("breast_carcinoma").disease_id == "MONDO_0004989"
    assert resolve_disease("breast carcinoma").disease_id == "MONDO_0004989"
    assert resolve_disease("breast cancer").disease_id == "MONDO_0004989"
    assert resolve_disease("CRC").disease_id == "MONDO_0005575"


def test_resolve_ontology_id_passthrough() -> None:
    disease = resolve_disease("MONDO_1234567")
    assert disease.disease_id == "MONDO_1234567"
    assert disease.name == "MONDO_1234567"


def test_resolve_known_id_returns_registry_entry() -> None:
    assert resolve_disease("MONDO_0004989").name == "breast carcinoma"


def test_resolve_unknown_without_network_raises() -> None:
    with pytest.raises(ValueError, match="Unknown disease"):
        resolve_disease("totally made up disease")


def test_resolve_unknown_with_network_uses_client() -> None:
    class FakeClient:
        def search_diseases(self, text, *, limit=10):
            return [{"id": "MONDO_9999999", "name": text}]

    disease = resolve_disease("made up disease", client=FakeClient(), allow_network=True)
    assert disease.disease_id == "MONDO_9999999"


def test_find_disease_partial_unique_match() -> None:
    assert find_disease("alzheimer").disease_id == "MONDO_0004975"


def test_is_disease_id() -> None:
    assert is_disease_id("MONDO_0004989")
    assert is_disease_id("EFO_0000305")
    assert not is_disease_id("breast carcinoma")


# ----------------------------------------------------------- Open Targets
def _row(symbol: str, ensembl: str, datatypes: dict, datasources: dict | None = None) -> dict:
    return {
        "score": max(datatypes.values()) if datatypes else 0.0,
        "target": {"id": ensembl, "approvedSymbol": symbol},
        "datatypeScores": [{"id": k, "score": v} for k, v in datatypes.items()],
        "datasourceScores": [{"id": k, "score": v} for k, v in (datasources or {}).items()],
    }


def test_rows_to_evidence_maps_and_takes_max() -> None:
    rows = [
        _row("BRCA1", "ENSG1", {"known_drug": 0.4, "drugs": 0.5, "literature": 0.7}),
        _row("TP53", "ENSG2", {"genetic_association": 0.6, "somatic_mutation": 0.8}),
    ]
    scores, ensembl = rows_to_evidence(rows)
    assert scores["BRCA1"]["drug"] == 0.5
    assert math.isclose(scores["BRCA1"]["literature"], 0.7)
    assert scores["TP53"]["genetics"] == 0.8
    assert ensembl == {"BRCA1": "ENSG1", "TP53": "ENSG2"}


def test_rows_to_evidence_ignores_datasources_by_default() -> None:
    rows = [_row("G1", "E1", {}, {"drugs": 0.9})]
    scores, _ = rows_to_evidence(rows)
    assert scores == {}


def test_rows_to_evidence_can_include_datasources() -> None:
    rows = [_row("G1", "E1", {}, {"drugs": 0.9})]
    scores, _ = rows_to_evidence(rows, include_datasources=True)
    assert scores["G1"]["drug"] == 0.9


def test_rows_to_evidence_skips_unknown_types_and_missing_symbols() -> None:
    rows = [
        _row("G1", "E1", {"mystery": 0.9}),
        {"score": 0.5, "target": {"id": "E2", "approvedSymbol": ""}, "datatypeScores": []},
    ]
    scores, ensembl = rows_to_evidence(rows)
    assert scores == {}
    assert ensembl == {}


def test_rows_to_evidence_merges_duplicate_symbols() -> None:
    rows = [
        _row("G1", "E1", {"literature": 0.2}),
        _row("G1", "E2", {"literature": 0.9}),
    ]
    scores, _ = rows_to_evidence(rows)
    assert scores["G1"]["literature"] == 0.9


class _FakeResponse:
    def __init__(self, payload: dict, status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code

    def json(self) -> dict:
        return self._payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")


class _FakeSession:
    def __init__(self, responses) -> None:
        self._responses = list(responses)
        self.calls = []

    def post(self, url, json=None, timeout=None, headers=None):
        self.calls.append(json)
        if not self._responses:
            raise AssertionError("unexpected extra request")
        return self._responses.pop(0)


def _disease_payload(name: str, rows, count: int) -> dict:
    return {
        "data": {
            "disease": {
                "id": "MONDO_0004989",
                "name": name,
                "associatedTargets": {"count": count, "rows": rows},
            }
        }
    }


def test_client_paginates_until_max_genes() -> None:
    session = _FakeSession(
        [
            _FakeResponse(_disease_payload("breast carcinoma", [
                _row("A", "E1", {"literature": 0.9}),
                _row("B", "E2", {"literature": 0.8}),
            ], 3)),
            _FakeResponse(_disease_payload("breast carcinoma", [
                _row("C", "E3", {"literature": 0.7}),
            ], 3)),
        ]
    )
    client = OpenTargetsClient(session=session, rate_limit=0, max_retries=0)
    evidence = client.fetch_associations("MONDO_0004989", max_genes=3, page_size=2)
    assert evidence.disease_name == "breast carcinoma"
    assert set(evidence.gene_type_scores) == {"A", "B", "C"}
    assert len(session.calls) == 2
    assert session.calls[0]["variables"]["size"] == 2
    assert session.calls[1]["variables"]["size"] == 1


def test_client_raises_for_unknown_disease() -> None:
    session = _FakeSession([_FakeResponse({"data": {"disease": None}})])
    client = OpenTargetsClient(session=session, rate_limit=0, max_retries=0)
    with pytest.raises(OpenTargetsError, match="was not found"):
        client.fetch_associations("MONDO_0")


def test_client_raises_on_graphql_errors() -> None:
    session = _FakeSession([_FakeResponse({"errors": [{"message": "boom"}]})])
    client = OpenTargetsClient(session=session, rate_limit=0, max_retries=0)
    with pytest.raises(OpenTargetsError, match="boom"):
        client.query("query { x }", {})


def test_client_retries_on_server_error() -> None:
    session = _FakeSession(
        [
            _FakeResponse({}, status_code=503),
            _FakeResponse(_disease_payload("d", [_row("A", "E", {"literature": 0.9})], 1)),
        ]
    )
    client = OpenTargetsClient(session=session, rate_limit=0, max_retries=1, backoff=0)
    evidence = client.fetch_associations("MONDO_0004989", max_genes=1)
    assert set(evidence.gene_type_scores) == {"A"}


def test_fetch_disease_evidence_uses_cache(tmp_path: Path) -> None:
    cache = Cache(tmp_path / "cache")
    calls = {"n": 0}

    class FakeClient:
        def fetch_associations(self, disease_id, *, max_genes=2000, page_size=500, include_datasources=False):
            calls["n"] += 1
            return DiseaseEvidence(
                disease_id=disease_id,
                disease_name="breast carcinoma",
                gene_type_scores={"A": {"literature": 0.9}},
                ensembl_ids={"A": "E1"},
                n_rows=1,
                total_associations=1,
            )

    first = fetch_disease_evidence("breast carcinoma", client=FakeClient(), cache=cache)
    second = fetch_disease_evidence("breast carcinoma", cache=cache, offline=True)
    assert calls["n"] == 1
    assert first.gene_type_scores == second.gene_type_scores
    assert cache.info("diseases/MONDO_0004989.json")["disease_id"] == "MONDO_0004989"


def test_fetch_disease_evidence_offline_miss_raises(tmp_path: Path) -> None:
    cache = Cache(tmp_path / "cache")
    with pytest.raises(FileNotFoundError, match="genepriori fetch"):
        fetch_disease_evidence("MONDO_0004989", cache=cache, offline=True)


def test_evidence_summary_counts_types() -> None:
    evidence = DiseaseEvidence(
        disease_id="MONDO_1",
        disease_name="x",
        gene_type_scores={"A": {"drug": 0.1, "literature": 0.2}, "B": {"drug": 0.3}},
    )
    summary = evidence_summary(evidence)
    assert summary["genes"] == 2
    assert summary["genes_with_evidence_per_type"]["drug"] == 2
    assert summary["genes_with_evidence_per_type"]["literature"] == 1


# ------------------------------------------------------------------ STRING
def _write_string_fixture(tmp_path: Path) -> tuple[Path, Path]:
    links = tmp_path / "links.txt.gz"
    info = tmp_path / "info.txt.gz"
    with gzip.open(links, "wt", encoding="utf-8") as handle:
        handle.write("protein1 protein2 combined_score\n")
        handle.write("9606.P1 9606.P2 900\n")
        handle.write("9606.P2 9606.P1 900\n")  # duplicate direction
        handle.write("9606.P1 9606.P2 950\n")  # higher duplicate
        handle.write("9606.P1 9606.P3 300\n")  # below threshold
        handle.write("9606.P1 9606.P4 800\n")  # P4 has no symbol
    with gzip.open(info, "wt", encoding="utf-8") as handle:
        handle.write("#string_protein_id\tpreferred_name\tprotein_size\tannotation\n")
        handle.write("9606.P1\tGENEA\t100\tx\n")
        handle.write("9606.P2\tGENEB\t100\tx\n")
        handle.write("9606.P3\tGENEC\t100\tx\n")
    return links, info


def test_load_id_map(tmp_path: Path) -> None:
    _, info = _write_string_fixture(tmp_path)
    mapping = load_id_map(info)
    assert mapping["9606.P1"] == "GENEA"
    assert "#string_protein_id" not in mapping


def test_build_edges_filters_maps_and_dedupes(tmp_path: Path) -> None:
    links, info = _write_string_fixture(tmp_path)
    edges = build_edges(links, load_id_map(info), min_combined_score=400)
    assert edges == [("GENEA", "GENEB", 0.95)]


def test_edges_cache_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "edges.tsv.gz"
    write_edges_cache(path, [("A", "B", 0.9), ("B", "C", 0.5)])
    assert read_edges_cache(path) == [("A", "B", 0.9), ("B", "C", 0.5)]


def test_edges_cache_key_includes_threshold() -> None:
    assert "min400" in edges_cache_key("12.5", 9606, 400)
    assert "min700" in edges_cache_key("12.5", 9606, 700)


def test_download_string_offline_missing_raises(tmp_path: Path) -> None:
    cache = Cache(tmp_path / "cache")
    with pytest.raises(FileNotFoundError, match="genepriori fetch-ppi"):
        download_string(cache=cache, offline=True)
