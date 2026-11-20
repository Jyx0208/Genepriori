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
    manifest = json.loads((outdir / "run_manifest.json").read_text(encoding="utf-8"))
    for disease_id in ("MONDO_9000001", "MONDO_9000002"):
        ranking = f"rankings/{disease_id}/ranked_genes.csv"
        assert ranking in manifest["outputs"]
        assert (outdir / ranking).read_text(encoding="utf-8").startswith("rank,gene,score,")
        result = json.loads((outdir / f"rankings/{disease_id}/result.json").read_text(encoding="utf-8"))
        assert result["disease_id"] == disease_id and result["ranked"]


def test_debate_shows_model_reply_and_assessment_as_they_happen(
    cache_dir, tmp_path, capsys, monkeypatch
):
    private_key = "test-private-key"
    monkeypatch.setenv("GENEPRIORI_LLM_API_KEY", private_key)
    monkeypatch.setenv("GENEPRIORI_LLM_MODEL", "mock-flash")
    monkeypatch.setenv("GENEPRIORI_LLM_BASE_URL", "https://example.com/v1")
    seen = []

    class Response:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            claim = {
                "title": "Evaluate a new recipe",
                "hypothesis": f"Expression may help. \x1b[31m{private_key}",
                "evidence_type_weights": {"genetics": 0.6, "expression": 0.4},
                "network_algorithm": "rwr",
                "network_params": {"restart_prob": 0.3, "blend": 0.9},
                "falsification_test": "AUPRC does not improve",
            }
            return {"choices": [{"message": {"content": json.dumps({"claims": [claim]})}}]}

    def model_post(session, url, **kwargs):
        seen.append(capsys.readouterr().err)
        return Response()

    monkeypatch.setattr("requests.Session.post", model_post)
    outdir = tmp_path / "debate"
    assert main([
        "debate", "--diseases", "MONDO_9000001", "--cache-dir", str(cache_dir),
        "--offline", "--mode", "llm", "--holdout", "drug", "--rounds", "1",
        "--budget", "2", "-o", str(outdir),
    ]) == 0
    assert "Loading STRING" in seen[0] and "Baseline AUPRC" in seen[0]
    assert "CausalGenetics: waiting for model response" in seen[0] and "mock-flash" in seen[0]
    assert "Hypothesis: Expression may help" in seen[1]
    assert "NetworkMechanism: waiting for model response" in seen[1]
    output = capsys.readouterr()
    assert "not selected: duplicate proposal" in output.err
    assert "Evaluating" in output.err and "measured AUPRC" in output.err
    assert "round" in output.out and "auprc" in output.out
    transcript = "".join(seen) + output.err
    assert "AUPRC does not improve" in transcript and '"restart_prob": 0.3' in transcript
    assert private_key not in transcript and "\x1b" not in transcript
    report = json.loads((outdir / "debate.json").read_text(encoding="utf-8"))["drug"]
    assert len(report["leaderboard"]) == 1 and report["llm_errors"] == []
    assert report["llm"] == {
        "enabled": True, "model": "mock-flash", "base_url": "https://example.com/v1",
    }
    assert len(seen) == 4
    selection = json.loads((outdir / "run_manifest.json").read_text(encoding="utf-8"))["ranking_selection"]
    assert selection["selected"] == "drug_winner"
    assert selection["candidates"]["drug_winner"]["claim_id"] == report["best"]["claim_id"]
    result = json.loads((outdir / "result.json").read_text(encoding="utf-8"))
    assert result["config"]["weights"]["drug"] == pytest.approx(1 / 6)
    assert result["config"]["network"]["algorithm"] == "rwr"
    from genepriori import prioritize

    baseline = prioritize("MONDO_9000001", cache_dir=cache_dir, offline=True)
    positions = {row.gene: row.rank for row in baseline.ranked}
    assert any(row["rank"] != positions.get(row["gene"]) for row in result["ranked"])
    for row in result["ranked"]:
        before = positions.get(row["gene"])
        assert row["baseline_rank"] == before
        assert row["rank_change"] == (before - row["rank"] if before is not None else None)
    csv_text = (outdir / "ranked_genes.csv").read_text(encoding="utf-8")
    assert csv_text.splitlines()[0].endswith("direct_score,network_score,baseline_rank,rank_change")


def test_debate_reports_llm_fallback(cache_dir, tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("GENEPRIORI_LLM_API_KEY", "test-private-key")

    def unavailable(*args, **kwargs):
        raise TimeoutError("test timeout")

    monkeypatch.setattr("requests.Session.post", unavailable)
    monkeypatch.setattr("genepriori.agents.llm.time.sleep", lambda seconds: None)
    outdir = tmp_path / "debate"
    assert main([
        "debate", "--diseases", "MONDO_9000001", "--cache-dir", str(cache_dir),
        "--offline", "--mode", "llm", "--holdout", "drug", "--rounds", "1",
        "--budget", "2", "-o", str(outdir),
    ]) == 0
    output = capsys.readouterr()
    assert "Model request attempt 3/3" in output.err
    assert "falling back to heuristic proposals" in output.err
    assert "from heuristic agent" in output.err
    report = json.loads((outdir / "debate.json").read_text(encoding="utf-8"))["drug"]
    assert len(report["llm_errors"]) == 4
    assert all(not entry["claim_id"].startswith("llm-") for entry in report["leaderboard"])


@pytest.mark.parametrize(
    ("flags", "visible"),
    [(["--progress"], True), (["--no-progress"], False), (["--progress", "--quiet"], False)],
)
def test_debate_progress_keeps_json_stdout_clean(cache_dir, tmp_path, capsys, flags, visible):
    assert main([
        "debate", "--diseases", "MONDO_9000001", "--cache-dir", str(cache_dir),
        "--offline", "--mode", "heuristic", "--holdout", "both", "--rounds", "1",
        "--budget", "2", "--json", "-o", str(tmp_path / "debate"), *flags,
    ]) == 0
    output = capsys.readouterr()
    assert set(json.loads(output.out)) == {"drug", "genetics"}
    assert ("Holdout 1/2" in output.err) is visible
    assert ("Holdout 2/2" in output.err) is visible
    assert ("Saved report to" in output.err) is visible
    assert ("Selected scoring configuration" in output.err) is visible
    if visible:
        assert output.err.index("Holdout 1/2") < output.err.index("Holdout 2/2")
        assert output.err.index("Holdout 2/2") < output.err.index("Selected scoring configuration")
    manifest = json.loads((tmp_path / "debate" / "run_manifest.json").read_text(encoding="utf-8"))
    selection = manifest["ranking_selection"]
    chosen = selection["candidates"][selection["selected"]]["auprc"]
    baseline = selection["candidates"]["baseline"]["auprc"]
    assert set(chosen) == {"drug", "genetics"}
    assert all(chosen[key] >= baseline[key] for key in baseline)


def test_debate_can_skip_final_ranking(cache_dir, tmp_path):
    outdir = tmp_path / "debate"
    assert main([
        "debate", "--diseases", "MONDO_9000001", "--cache-dir", str(cache_dir),
        "--offline", "--mode", "heuristic", "--holdout", "drug", "--rounds", "1",
        "--budget", "2", "--no-ranking", "-o", str(outdir),
    ]) == 0
    manifest = json.loads((outdir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["outputs"] == ["debate.json"]
    assert "ranking_selection" not in manifest
    assert not (outdir / "ranked_genes.csv").exists()


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
