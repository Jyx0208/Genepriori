"""End-to-end CLI tests (all offline, using the synthetic cache)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from genepriori.cli import main
from genepriori.data.toy import build_toy_cache


@pytest.fixture()
def cache_dir(tmp_path: Path) -> Path:
    return build_toy_cache(tmp_path / "cache")


def test_list_diseases(capsys) -> None:
    assert main(["list-diseases"]) == 0
    out = capsys.readouterr().out
    assert "breast carcinoma" in out
    assert "MONDO_0004989" in out


def test_list_diseases_category(capsys) -> None:
    assert main(["list-diseases", "--category", "oncology"]) == 0
    out = capsys.readouterr().out
    assert "colorectal cancer" in out
    assert "asthma" not in out


def test_list_diseases_json(capsys) -> None:
    assert main(["list-diseases", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert len(payload) == 19


def test_doctor_offline(capsys) -> None:
    assert main(["doctor", "--offline"]) == 0
    out = capsys.readouterr().out
    assert "python" in out
    assert "cache_writable" in out


def test_demo_writes_outputs(tmp_path: Path, capsys) -> None:
    outdir = tmp_path / "demo"
    code = main(["demo", "-o", str(outdir), "--cache-dir", str(tmp_path / "cache"), "--top", "3"])
    assert code == 0
    assert (outdir / "rank" / "ranked_genes.csv").exists()
    assert (outdir / "evaluation").exists()
    assert (outdir / "run_manifest.json").exists()
    assert "Leakage-controlled evaluation" in capsys.readouterr().out


def test_rank_command(cache_dir: Path, tmp_path: Path, capsys) -> None:
    outdir = tmp_path / "rank"
    code = main(
        [
            "rank",
            "--disease",
            "MONDO_9000001",
            "--cache-dir",
            str(cache_dir),
            "--offline",
            "--network",
            "ppr",
            "--top",
            "5",
            "-o",
            str(outdir),
        ]
    )
    assert code == 0
    assert (outdir / "ranked_genes.csv").exists()
    text = capsys.readouterr().out
    assert "synthetic breast carcinoma" in text


def test_rank_json_output(cache_dir: Path, capsys) -> None:
    code = main(
        [
            "rank",
            "--disease",
            "MONDO_9000001",
            "--cache-dir",
            str(cache_dir),
            "--offline",
            "--network",
            "none",
            "--json",
            "--top",
            "2",
        ]
    )
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["disease_id"] == "MONDO_9000001"
    assert len(payload["ranked"]) == 2


def test_evaluate_command(cache_dir: Path, tmp_path: Path, capsys) -> None:
    outdir = tmp_path / "eval"
    code = main(
        [
            "evaluate",
            "--disease",
            "MONDO_9000001",
            "--cache-dir",
            str(cache_dir),
            "--offline",
            "--holdout",
            "both",
            "-o",
            str(outdir),
        ]
    )
    assert code == 0
    files = {p.name for p in outdir.glob("*.json")}
    assert "MONDO_9000001_drug_metrics.json" in files
    assert "MONDO_9000001_genetics_metrics.json" in files


def test_benchmark_command(cache_dir: Path, tmp_path: Path, capsys) -> None:
    outdir = tmp_path / "bench"
    code = main(
        [
            "benchmark",
            "--diseases",
            "MONDO_9000001,MONDO_9000002",
            "--cache-dir",
            str(cache_dir),
            "--offline",
            "--holdout",
            "drug",
            "--variants",
            "baseline,ppr",
            "--n-bootstrap",
            "100",
            "-o",
            str(outdir),
        ]
    )
    assert code == 0
    assert (outdir / "benchmark.csv").exists()
    comparison = json.loads((outdir / "comparison.json").read_text(encoding="utf-8"))
    assert "drug" in comparison


def test_debate_command(cache_dir: Path, tmp_path: Path, capsys) -> None:
    outdir = tmp_path / "debate"
    code = main(
        [
            "debate",
            "--diseases",
            "MONDO_9000001,MONDO_9000002",
            "--cache-dir",
            str(cache_dir),
            "--offline",
            "--rounds",
            "1",
            "--budget",
            "2",
            "--holdout",
            "drug",
            "-o",
            str(outdir),
        ]
    )
    assert code == 0
    payload = json.loads((outdir / "debate.json").read_text(encoding="utf-8"))
    assert "drug" in payload


def test_unknown_disease_returns_error_code(capsys) -> None:
    code = main(["rank", "--disease", "not a real disease", "--offline"])
    assert code == 2
    assert "Unknown disease" in capsys.readouterr().err


def test_offline_miss_hints_fetch(cache_dir: Path, tmp_path: Path, capsys) -> None:
    code = main(
        [
            "rank",
            "--disease",
            "MONDO_0004989",
            "--cache-dir",
            str(tmp_path / "empty"),
            "--offline",
        ]
    )
    assert code == 2
    err = capsys.readouterr().err
    assert "No cached evidence" in err


def test_config_template(tmp_path: Path, capsys) -> None:
    assert main(["config"]) == 0
    assert "weights" in capsys.readouterr().out
    path = tmp_path / "run.yaml"
    assert main(["config", "--write", str(path)]) == 0
    assert path.exists()
