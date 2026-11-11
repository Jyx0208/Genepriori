"""Command line interface for GenePriori.

Examples
--------
    genepriori demo
    genepriori list-diseases
    genepriori fetch --disease breast carcinoma --max-genes 2000
    genepriori fetch-ppi
    genepriori rank --disease breast carcinoma --network ppr -o results/
    genepriori evaluate --disease breast carcinoma --holdout both
    genepriori benchmark --diseases oncology --holdout both -o results/benchmark
    genepriori doctor
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from . import __version__
from .config import RunConfig
from .data.cache import Cache, default_home
from .data.opentargets import (
    OpenTargetsClient,
    OpenTargetsError,
    evidence_summary,
    fetch_disease_evidence,
)
from .data.registry import (
    DEFAULT_DISEASES,
    Disease,
    diseases_by_category,
    list_diseases,
    resolve_disease,
)
from .data.string import StringError, load_string_edges, ppi_summary
from .data.toy import TOY_DISEASES, build_toy_cache
from .pipeline.evaluation import (
    EvaluationResult,
    adjust_families,
    compare_evaluations,
    evaluate_evidence,
)
from .pipeline.fusion import prioritize
from .pipeline.report import (
    evaluation_table,
    ranked_table,
    write_evaluation,
    write_json,
    write_manifest,
    write_ranking,
)

LOG = logging.getLogger("genepriori")

BENCHMARK_VARIANTS: Dict[str, Dict[str, str]] = {
    "baseline": {"weights": "balanced", "network": "none"},
    "ppr": {"weights": "balanced", "network": "ppr"},
    "rwr": {"weights": "balanced", "network": "rwr"},
    "genetics_ppr": {"weights": "genetics", "network": "ppr"},
    "translational_ppr": {"weights": "translational", "network": "ppr"},
}

SUPPORTED_HOLDOUTS = ("drug", "genetics")


# --------------------------------------------------------------------- helpers
def _make_cache(args: argparse.Namespace) -> Cache:
    return Cache(args.cache_dir if getattr(args, "cache_dir", None) else None)


def _args_payload(args: argparse.Namespace) -> Dict[str, Any]:
    """JSON-safe copy of parsed arguments (drops the command callable)."""
    payload: Dict[str, Any] = {}
    for key, value in vars(args).items():
        if key == "func":
            continue
        if key in {"llm_api_key"}:
            payload[key] = "***" if value else None
            continue
        if isinstance(value, (str, int, float, bool)) or value is None:
            payload[key] = value
        elif isinstance(value, (list, tuple)):
            payload[key] = [
                item for item in value if isinstance(item, (str, int, float, bool)) or item is None
            ]
    return payload


def _parse_weights(value: Optional[str]) -> Optional[Any]:
    if value is None:
        return None
    candidate = Path(value)
    if candidate.suffix.lower() in {".yaml", ".yml", ".json"} and candidate.exists():
        import yaml

        text = candidate.read_text(encoding="utf-8")
        payload = yaml.safe_load(text) if candidate.suffix.lower() != ".json" else json.loads(text)
        if not isinstance(payload, dict):
            raise ValueError(f"{candidate} must contain a mapping of evidence type to weight.")
        return payload
    return value


def _config_from_args(args: argparse.Namespace) -> RunConfig:
    config = RunConfig.load(args.config) if getattr(args, "config", None) else RunConfig()
    weights = _parse_weights(getattr(args, "weights", None))
    if weights is not None:
        config = config.with_weights(weights)
    if getattr(args, "network", None) is not None:
        payload = config.to_dict()
        payload["network"]["algorithm"] = args.network
        config = RunConfig.from_dict(payload)
    if getattr(args, "folds", None) is not None:
        config.evaluation.folds = args.folds
    if getattr(args, "protocol", None) is not None:
        config.evaluation.protocol = args.protocol
    if getattr(args, "k", None) is not None:
        config.evaluation.k = args.k
    if getattr(args, "max_genes", None) is not None:
        config.max_genes = args.max_genes
    if getattr(args, "positive_threshold", None) is not None:
        config.evaluation.positive_threshold = args.positive_threshold
    if getattr(args, "string_version", None) is not None:
        config.string_version = args.string_version
    if getattr(args, "min_combined_score", None) is not None:
        config.min_combined_score = args.min_combined_score
    if getattr(args, "llm_model", None) or getattr(args, "llm_base_url", None):
        payload = config.to_dict()
        if getattr(args, "llm_model", None):
            payload["llm"]["model"] = args.llm_model
        if getattr(args, "llm_base_url", None):
            payload["llm"]["base_url"] = args.llm_base_url
        config = RunConfig.from_dict(payload)
    # Re-validate after attribute assignment.
    return RunConfig.from_dict(config.to_dict())


def _collect_diseases(values: Optional[Sequence[str]]) -> List[Disease]:
    if not values:
        raise ValueError("At least one --disease/--diseases value is required.")
    collected: List[Disease] = []
    seen = set()
    for value in values:
        text = (value or "").strip()
        if not text:
            continue
        lowered = text.lower()
        if lowered in {"registry", "all", "default"}:
            candidates = list(DEFAULT_DISEASES)
        elif lowered in {c.lower() for c in diseases_by_category()}:
            category = next(c for c in diseases_by_category() if c.lower() == lowered)
            candidates = diseases_by_category()[category]
        else:
            candidates = [resolve_disease(part) for part in text.split(",") if part.strip()]
        for disease in candidates:
            if disease.disease_id not in seen:
                seen.add(disease.disease_id)
                collected.append(disease)
    if not collected:
        raise ValueError("No diseases resolved from the provided values.")
    return collected


def _emit(args: argparse.Namespace, payload: Any, text: str) -> None:
    if getattr(args, "json", False):
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(text)


# -------------------------------------------------------------------- commands
def cmd_list_diseases(args: argparse.Namespace) -> int:
    grouped = diseases_by_category()
    entries = list_diseases()
    if args.category:
        entries = [d for d in entries if d.category == args.category.lower()]
    rows = [[d.category, d.key, d.name, d.disease_id] for d in entries]
    payload = [
        {"category": d.category, "key": d.key, "name": d.name, "disease_id": d.disease_id}
        for d in entries
    ]
    from .pipeline.report import format_table

    text = format_table(["category", "key", "name", "disease_id"], rows)
    if not args.category:
        text += f"\n\n{len(DEFAULT_DISEASES)} diseases across {len(grouped)} categories."
    _emit(args, payload, text)
    return 0


def cmd_search_disease(args: argparse.Namespace) -> int:
    if args.offline:
        raise ValueError("search-disease requires network access; drop --offline.")
    client = OpenTargetsClient()
    hits = client.search_diseases(args.query, limit=args.limit)
    if not hits:
        print(f"No disease matched {args.query!r}.", file=sys.stderr)
        return 1
    rows = [[h["id"], h["name"]] for h in hits]
    from .pipeline.report import format_table

    _emit(args, hits, format_table(["disease_id", "name"], rows))
    return 0


def cmd_fetch(args: argparse.Namespace) -> int:
    diseases = _collect_diseases(args.disease if not args.all else ["registry"])
    cache = _make_cache(args)
    client = None if args.offline else OpenTargetsClient()
    summaries = []
    for disease in diseases:
        evidence = fetch_disease_evidence(
            disease,
            client=client,
            cache=cache,
            max_genes=args.max_genes,
            refresh=args.refresh,
            offline=args.offline,
        )
        summary = evidence_summary(evidence)
        summaries.append(summary)
        LOG.info("fetched %s: %d genes", disease.disease_id, summary["genes"])

    from .pipeline.report import format_table

    rows = [
        [s["disease_id"], s["disease_name"], s["genes"], s["rows"], s["total_associations"]]
        for s in summaries
    ]
    _emit(
        args,
        summaries,
        format_table(["disease_id", "name", "genes", "rows", "total_associations"], rows),
    )
    return 0


def cmd_fetch_ppi(args: argparse.Namespace) -> int:
    cache = _make_cache(args)
    edges = load_string_edges(
        version=args.string_version,
        min_combined_score=args.min_combined_score,
        cache=cache,
        refresh=args.refresh,
        offline=args.offline,
    )
    summary = ppi_summary(edges)
    summary.update(
        {
            "version": args.string_version,
            "min_combined_score": args.min_combined_score,
            "cache_file": str(
                cache.path(
                    f"ppi/v{args.string_version}/9606.edges.min{args.min_combined_score}.tsv.gz"
                )
            ),
        }
    )
    from .pipeline.report import format_table

    text = format_table(
        ["metric", "value"],
        [[k, v] for k, v in summary.items()],
    )
    _emit(args, summary, text)
    return 0


def cmd_rank(args: argparse.Namespace) -> int:
    config = _config_from_args(args)
    cache = _make_cache(args)
    result = prioritize(
        args.disease,
        config=config,
        cache=cache,
        offline=args.offline,
        refresh=args.refresh,
    )
    outdir = Path(args.out or f"results/rank_{result.disease_id}")
    write_ranking(result, outdir)
    write_manifest(
        outdir,
        command="rank",
        arguments=_args_payload(args),
        config=result.config,
        inputs={
            "disease_id": result.disease_id,
            "evidence_summary": result.evidence_summary,
            "network_summary": result.network_summary,
        },
        outputs=["ranked_genes.csv", "result.json"],
    )
    if args.json:
        payload = result.to_dict()
        payload["ranked"] = payload["ranked"][: args.top]
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(f"{result.disease_name} ({result.disease_id}) - {len(result.ranked)} genes")
        print(ranked_table(result, top=args.top))
        print(f"\nwritten to {outdir}")
    return 0


def _evaluate_one(
    disease: Disease,
    config: RunConfig,
    cache: Cache,
    edges,
    args: argparse.Namespace,
) -> List[EvaluationResult]:
    results: List[EvaluationResult] = []
    holdouts = SUPPORTED_HOLDOUTS if args.holdout == "both" else (args.holdout,)
    for holdout in holdouts:
        holdout_config = config.with_holdout(holdout)
        evidence = fetch_disease_evidence(
            disease,
            client=None if args.offline else OpenTargetsClient(),
            cache=cache,
            max_genes=holdout_config.max_genes,
            refresh=args.refresh,
            offline=args.offline,
        )
        results.append(evaluate_evidence(evidence, config=holdout_config, edges=edges))
    return results


def cmd_evaluate(args: argparse.Namespace) -> int:
    config = _config_from_args(args)
    cache = _make_cache(args)
    diseases = _collect_diseases([args.disease] + (args.diseases or []))
    edges = (
        load_string_edges(
            version=config.string_version,
            min_combined_score=config.min_combined_score,
            cache=cache,
            refresh=args.refresh,
            offline=args.offline,
        )
        if config.network.algorithm != "none"
        else None
    )

    results: List[EvaluationResult] = []
    for disease in diseases:
        results.extend(_evaluate_one(disease, config, cache, edges, args))

    outdir = Path(args.out or "results/evaluation")
    for result in results:
        prefix = f"{result.disease_id}_{result.holdout}"
        write_evaluation(result, outdir, prefix=prefix)
    write_manifest(
        outdir,
        command="evaluate",
        arguments=_args_payload(args),
        config=config.to_dict(),
        inputs={"diseases": [d.disease_id for d in diseases]},
        outputs=sorted(p.name for p in outdir.glob("*") if p.is_file()),
    )

    if args.json:
        print(json.dumps([r.to_dict() for r in results], indent=2))
    else:
        print(evaluation_table(results, metric="auprc", k=config.evaluation.k))
        print(f"\nwritten to {outdir}")
    return 0


def cmd_benchmark(args: argparse.Namespace) -> int:
    config = _config_from_args(args)
    cache = _make_cache(args)
    diseases = _collect_diseases(args.diseases or ["oncology"])
    if config.network.algorithm == "none":
        config = RunConfig.from_dict({**config.to_dict(), "network": {**config.to_dict()["network"], "algorithm": "ppr"}})
    edges = load_string_edges(
        version=config.string_version,
        min_combined_score=config.min_combined_score,
        cache=cache,
        refresh=args.refresh,
        offline=args.offline,
    )

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    unknown = [v for v in variants if v not in BENCHMARK_VARIANTS]
    if unknown:
        raise ValueError(
            f"Unknown benchmark variant(s) {unknown}. Expected a subset of {sorted(BENCHMARK_VARIANTS)}."
        )
    if "baseline" not in variants:
        variants.insert(0, "baseline")

    holdouts = SUPPORTED_HOLDOUTS if args.holdout == "both" else (args.holdout,)
    evidence_by_disease = {}
    for disease in diseases:
        evidence_by_disease[disease.disease_id] = fetch_disease_evidence(
            disease,
            client=None if args.offline else OpenTargetsClient(),
            cache=cache,
            max_genes=config.max_genes,
            refresh=args.refresh,
            offline=args.offline,
        )

    results: Dict[str, Dict[str, Dict[str, EvaluationResult]]] = {}
    rows: List[List[Any]] = []
    for holdout in holdouts:
        for variant in variants:
            spec = BENCHMARK_VARIANTS[variant]
            variant_config = config.with_holdout(holdout).with_weights(spec["weights"])
            payload = variant_config.to_dict()
            payload["network"]["algorithm"] = spec["network"]
            variant_config = RunConfig.from_dict(payload)
            for disease in diseases:
                try:
                    result = evaluate_evidence(
                        evidence_by_disease[disease.disease_id],
                        config=variant_config,
                        edges=edges,
                    )
                except ValueError as exc:
                    LOG.warning("%s / %s / %s skipped: %s", disease.disease_id, holdout, variant, exc)
                    continue
                results.setdefault(holdout, {}).setdefault(variant, {})[
                    disease.disease_id
                ] = result
                summary = result.summary["auprc"]
                rows.append(
                    [
                        holdout,
                        variant,
                        disease.disease_id,
                        evidence_by_disease[disease.disease_id].disease_name,
                        f"{summary['mean']:.4f}",
                        f"{summary['sd']:.4f}",
                    ]
                )

    comparisons: Dict[str, Dict[str, Any]] = {}
    for holdout in holdouts:
        by_variant = results.get(holdout, {})
        baseline = by_variant.get("baseline", {})
        if not baseline:
            continue
        family = {}
        for variant, runs in by_variant.items():
            if variant == "baseline":
                continue
            candidate = list(runs.values())
            base = list(baseline.values())
            if not candidate:
                continue
            family[variant] = compare_evaluations(
                base,
                candidate,
                metric="auprc",
                n_bootstrap=args.n_bootstrap,
            )
        comparisons[holdout] = adjust_families(family, alpha=args.alpha) if family else {}

    outdir = Path(args.out or "results/benchmark")
    outdir.mkdir(parents=True, exist_ok=True)
    import csv

    with open(outdir / "benchmark.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["holdout", "variant", "disease_id", "disease_name", "auprc_mean", "auprc_sd"])
        writer.writerows(rows)
    write_json(comparisons, outdir / "comparison.json")
    write_manifest(
        outdir,
        command="benchmark",
        arguments=_args_payload(args),
        config=config.to_dict(),
        inputs={"diseases": [d.disease_id for d in diseases], "variants": variants},
        outputs=["benchmark.csv", "comparison.json"],
    )

    if args.json:
        print(json.dumps(comparisons, indent=2))
    else:
        from .pipeline.report import format_table

        print(format_table(["holdout", "variant", "disease", "name", "auprc", "sd"], rows[:50]))
        for holdout, family in comparisons.items():
            if not family:
                continue
            print(f"\n[{holdout}] vs baseline")
            summary_rows = [
                [
                    variant,
                    f"{c['delta_mean']:+.4f}",
                    f"{c['wins']}/{c['n_diseases']}",
                    f"{c['sign_test_p']:.3g}",
                    f"{c.get('sign_test_p_adj', float('nan')):.3g}",
                ]
                for variant, c in family.items()
            ]
            print(format_table(["variant", "delta_auprc", "wins", "p_sign", "p_adj"], summary_rows))
        print(f"\nwritten to {outdir}")
    return 0


def cmd_debate(args: argparse.Namespace) -> int:
    from .agents.debate import run_debate

    config = _config_from_args(args)
    cache = _make_cache(args)
    diseases = _collect_diseases(args.diseases or ["oncology"])
    holdouts = SUPPORTED_HOLDOUTS if args.holdout == "both" else (args.holdout,)
    reports: Dict[str, Any] = {}
    for holdout in holdouts:
        reports[holdout] = run_debate(
            diseases,
            config=config,
            cache=cache,
            rounds=args.rounds,
            budget=args.budget,
            mode=args.mode,
            holdout=holdout,
            offline=args.offline,
            refresh=args.refresh,
            llm_api_key=getattr(args, "llm_api_key", None),
        )
    outdir = Path(args.out or "results/debate")
    outdir.mkdir(parents=True, exist_ok=True)
    write_json(reports, outdir / "debate.json")
    write_manifest(
        outdir,
        command="debate",
        arguments=_args_payload(args),
        config=config.to_dict(),
        inputs={"diseases": [d.disease_id for d in diseases]},
        outputs=["debate.json"],
    )
    if args.json:
        print(json.dumps(reports, indent=2))
    else:
        from .pipeline.report import format_table

        rows = []
        for holdout, report in reports.items():
            for entry in report["decisions"]:
                rows.append(
                    [
                        holdout,
                        entry["round"],
                        entry["claim_id"],
                        entry["agent"],
                        f"{entry['score']:.4f}",
                        "accepted" if entry["accepted"] else "rejected",
                        entry.get("reason", ""),
                    ]
                )
        print(
            format_table(
                ["holdout", "round", "claim", "agent", "auprc", "status", "reason"], rows
            )
        )
        best_lines = [
            f"  {holdout}: {report['best']['claim_id']} ({report['best']['agent']}) "
            f"AUPRC={report['best']['auprc']:.4f}"
            for holdout, report in reports.items()
            if report.get("best")
        ]
        if best_lines:
            print("\nbest accepted claims:")
            print("\n".join(best_lines))
        mode_used = next(iter(reports.values())).get("mode")
        print(f"\nmode: {mode_used}")
        llm_errors = [err for report in reports.values() for err in report.get("llm_errors", [])]
        if llm_errors:
            print(
                f"warning: {len(llm_errors)} LLM call(s) failed; heuristic fallback was used. "
                "See llm_errors in debate.json.",
                file=sys.stderr,
            )
        print(f"\nwritten to {outdir}")
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    cache_root = Path(args.cache_dir) if args.cache_dir else default_home() / "demo-cache"
    build_toy_cache(cache_root)
    cache = Cache(cache_root)
    outdir = Path(args.out or "demo_results")

    result = prioritize(
        TOY_DISEASES[0][0], cache=cache, offline=True, network="ppr"
    )
    write_ranking(result, outdir / "rank")

    config = RunConfig()
    edges = load_string_edges(cache=cache, offline=True)
    evaluations = []
    for holdout in SUPPORTED_HOLDOUTS:
        evaluations.append(
            evaluate_evidence(
                fetch_disease_evidence(
                    TOY_DISEASES[0][0], cache=cache, offline=True
                ),
                config=config.with_holdout(holdout),
                edges=edges,
            )
        )
    for evaluation in evaluations:
        write_evaluation(
            evaluation, outdir / "evaluation", prefix=f"{evaluation.disease_id}_{evaluation.holdout}"
        )
    write_manifest(
        outdir,
        command="demo",
        arguments=_args_payload(args),
        config=config.to_dict(),
        inputs={"diseases": [TOY_DISEASES[0][0]], "cache": str(cache_root)},
        outputs=["rank/", "evaluation/"],
        extra={"note": "synthetic offline demo data"},
    )

    if args.json:
        print(
            json.dumps(
                {
                    "ranking": result.to_dict()["ranked"][: args.top],
                    "disease": result.disease_name,
                    "evaluations": [e.to_dict()["summary"] for e in evaluations],
                },
                indent=2,
            )
        )
    else:
        print("GenePriori offline demo (synthetic data)\n")
        print(f"{result.disease_name}: {len(result.ranked)} genes ranked")
        print(ranked_table(result, top=args.top))
        print("\nLeakage-controlled evaluation (5-fold):")
        print(evaluation_table(evaluations))
        print(f"\nwritten to {outdir}")
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    checks: List[Tuple[str, str, str]] = []

    def add(name: str, status: str, detail: str) -> None:
        checks.append((name, status, detail))

    version = sys.version_info
    add(
        "python",
        "OK" if version >= (3, 9) else "FAIL",
        f"{version.major}.{version.minor}.{version.micro} (requires >= 3.9)",
    )
    for module in ("numpy", "scipy", "requests", "yaml"):
        try:
            __import__(module)
            add(module, "OK", "importable")
        except Exception as exc:  # pragma: no cover - defensive
            add(module, "FAIL", f"{type(exc).__name__}: {exc}")

    cache = _make_cache(args)
    add("cache_dir", "OK", str(cache.root))
    try:
        probe = cache.root / ".write-test"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        add("cache_writable", "OK", "writable")
    except OSError as exc:
        add("cache_writable", "FAIL", str(exc))

    manifest = cache.manifest()
    cached_diseases = sorted(k for k in manifest if k.startswith("diseases/"))
    cached_ppi = sorted(k for k in manifest if k.startswith("ppi/"))
    add("cached_diseases", "OK" if cached_diseases else "WARN", f"{len(cached_diseases)} file(s)")
    add("cached_ppi", "OK" if cached_ppi else "WARN", f"{len(cached_ppi)} file(s)")

    from .agents.llm import resolve_api_key

    llm_config = RunConfig().llm
    api_key = resolve_api_key(llm_config)
    if api_key:
        masked = f"{api_key[:4]}...{api_key[-4:]}" if len(api_key) > 12 else "***"
        add("llm", "OK", f"{llm_config.model} | key {masked}")
    else:
        add("llm", "WARN", f"{llm_config.model} | no key")

    if args.offline:
        add("opentargets", "SKIP", "offline mode")
    else:
        try:
            client = OpenTargetsClient(timeout=15, max_retries=0)
            client.search_diseases("breast", limit=1)
            add("opentargets", "OK", "reachable")
        except Exception as exc:
            add("opentargets", "WARN", f"{type(exc).__name__}: {exc}")

    hints = []
    if not cached_diseases:
        hints.append("run `genepriori fetch --disease <name>` or `genepriori demo`")
    if not cached_ppi:
        hints.append("run `genepriori fetch-ppi` for network propagation")
    if hints:
        checks.append(("next", "INFO", "; ".join(hints)))

    if args.json:
        print(json.dumps([{"check": c, "status": s, "detail": d} for c, s, d in checks], indent=2))
    else:
        from .pipeline.report import format_table

        print(format_table(["check", "status", "detail"], checks))
    failed = any(status == "FAIL" for _, status, _ in checks)
    return 1 if failed else 0


def cmd_config(args: argparse.Namespace) -> int:
    config = RunConfig()
    if args.write:
        path = config.save(args.write)
        print(f"wrote template configuration to {path}")
        return 0
    import yaml

    text = yaml.safe_dump(config.to_dict(), sort_keys=False)
    if args.json:
        print(json.dumps(config.to_dict(), indent=2))
    else:
        print(text)
    return 0


# ----------------------------------------------------------------------- parser
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="genepriori",
        description="Leakage-aware gene prioritization with evidence fusion and network propagation.",
    )
    parser.add_argument("--version", action="version", version=f"genepriori {__version__}")

    # Shared options, accepted both before and after the subcommand.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--cache-dir", default=argparse.SUPPRESS, help="override the cache directory")
    common.add_argument("--offline", action="store_true", default=argparse.SUPPRESS,
                        help="never touch the network")
    common.add_argument("--json", action="store_true", default=argparse.SUPPRESS,
                        help="machine-readable output")
    common.add_argument("-v", "--verbose", action="count", default=argparse.SUPPRESS,
                        help="increase log verbosity")
    common.add_argument("-q", "--quiet", action="store_true", default=argparse.SUPPRESS,
                        help="suppress informational logs")

    parser.add_argument("--cache-dir", default=None, help="override the cache directory")
    parser.add_argument("--offline", action="store_true", help="never touch the network")
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    parser.add_argument("-v", "--verbose", action="count", default=0, help="increase log verbosity")
    parser.add_argument("-q", "--quiet", action="store_true", help="suppress informational logs")
    sub = parser.add_subparsers(dest="command", required=True)

    def add(name: str, func, help_text: str) -> argparse.ArgumentParser:
        sp = sub.add_parser(name, parents=[common], help=help_text)
        sp.set_defaults(func=func)
        return sp

    sp = add("list-diseases", cmd_list_diseases, "list the built-in disease registry")
    sp.add_argument("--category", default=None, help="filter by category (e.g. oncology)")

    sp = add("search-disease", cmd_search_disease, "search Open Targets for a disease identifier")
    sp.add_argument("query")
    sp.add_argument("--limit", type=int, default=10)

    sp = add("fetch", cmd_fetch, "download Open Targets evidence for one or more diseases")
    sp.add_argument("--disease", action="append", default=[], help="registry name or ontology ID")
    sp.add_argument("--all", action="store_true", help="fetch the whole built-in registry")
    sp.add_argument("--max-genes", type=int, default=2000)
    sp.add_argument("--refresh", action="store_true", help="ignore cached data")

    sp = add("fetch-ppi", cmd_fetch_ppi, "download and cache the STRING human PPI network")
    sp.add_argument("--string-version", default="12.5")
    sp.add_argument("--min-combined-score", type=int, default=400)
    sp.add_argument("--refresh", action="store_true")

    sp = add("rank", cmd_rank, "rank genes for one disease")
    sp.add_argument("--disease", required=True)
    sp.add_argument("-o", "--out", default=None)
    sp.add_argument("--weights", default=None, help="preset name or path to a YAML/JSON mapping")
    sp.add_argument("--network", default=None, help="ppr|rwr|neighbor_voting|hd|ensemble|none")
    sp.add_argument("--top", type=int, default=20)
    sp.add_argument("--config", default=None, help="path to a run configuration")
    sp.add_argument("--max-genes", type=int, default=None)
    sp.add_argument("--string-version", default=None)
    sp.add_argument("--min-combined-score", type=int, default=None)
    sp.add_argument("--refresh", action="store_true")

    sp = add("evaluate", cmd_evaluate, "dual-holdout cross-validation for one or more diseases")
    sp.add_argument("--disease", required=True)
    sp.add_argument("--diseases", action="append", default=[])
    sp.add_argument("--holdout", choices=["drug", "genetics", "both"], default="both")
    sp.add_argument("--folds", type=int, default=None)
    sp.add_argument("--protocol", choices=["mask", "restore"], default=None)
    sp.add_argument("--k", type=int, default=None)
    sp.add_argument("--positive-threshold", type=float, default=None)
    sp.add_argument("--weights", default=None)
    sp.add_argument("--network", default=None)
    sp.add_argument("--config", default=None)
    sp.add_argument("--max-genes", type=int, default=None)
    sp.add_argument("--string-version", default=None)
    sp.add_argument("--min-combined-score", type=int, default=None)
    sp.add_argument("-o", "--out", default=None)
    sp.add_argument("--refresh", action="store_true")

    sp = add("benchmark", cmd_benchmark, "compare method variants across diseases")
    sp.add_argument("--diseases", action="append", default=None,
                    help="registry|oncology|<category>|comma-separated names or IDs")
    sp.add_argument("--holdout", choices=["drug", "genetics", "both"], default="both")
    sp.add_argument("--variants", default="baseline,ppr,rwr")
    sp.add_argument("--folds", type=int, default=None)
    sp.add_argument("--protocol", choices=["mask", "restore"], default=None)
    sp.add_argument("--weights", default=None)
    sp.add_argument("--network", default=None)
    sp.add_argument("--config", default=None)
    sp.add_argument("--max-genes", type=int, default=None)
    sp.add_argument("--string-version", default=None)
    sp.add_argument("--min-combined-score", type=int, default=None)
    sp.add_argument("--n-bootstrap", type=int, default=2000)
    sp.add_argument("--alpha", type=float, default=0.05)
    sp.add_argument("-o", "--out", default=None)
    sp.add_argument("--refresh", action="store_true")

    sp = add("debate", cmd_debate, "run the multi-agent claim debate")
    sp.add_argument("--diseases", action="append", default=None)
    sp.add_argument("--rounds", type=int, default=3)
    sp.add_argument("--budget", type=int, default=12)
    sp.add_argument(
        "--mode",
        choices=["auto", "heuristic", "llm"],
        default="auto",
        help="auto uses the LLM when an API key is configured",
    )
    sp.add_argument("--holdout", choices=["drug", "genetics", "both"], default="both")
    sp.add_argument("--llm-api-key", default=None,
                    help="API key value; prefer DEEPSEEK_API_KEY or a .env file")
    sp.add_argument("--llm-model", default=None, help="e.g. deepseek-chat")
    sp.add_argument("--llm-base-url", default=None,
                    help="OpenAI-compatible endpoint, e.g. https://api.deepseek.com/v1")
    sp.add_argument("--weights", default=None)
    sp.add_argument("--network", default=None)
    sp.add_argument("--config", default=None)
    sp.add_argument("--max-genes", type=int, default=None)
    sp.add_argument("-o", "--out", default=None)
    sp.add_argument("--refresh", action="store_true")

    sp = add("demo", cmd_demo, "run the offline synthetic demo")
    sp.add_argument("-o", "--out", default=None)
    sp.add_argument("--top", type=int, default=10)

    sp = add("doctor", cmd_doctor, "check the environment, cache and network access")

    sp = add("config", cmd_config, "print or write a run configuration template")
    sp.add_argument("--write", default=None, help="write a template to this path")
    return parser


def _setup_logging(args: argparse.Namespace) -> None:
    if args.quiet:
        level = logging.ERROR
    elif args.verbose >= 2:
        level = logging.DEBUG
    elif args.verbose == 1:
        level = logging.INFO
    else:
        level = logging.WARNING
    logging.basicConfig(level=level, format="%(levelname)s %(message)s")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _setup_logging(args)
    try:
        from .agents.llm import load_dotenv

        load_dotenv()
    except Exception:  # pragma: no cover - .env support must never block the CLI
        pass
    try:
        return int(args.func(args))
    except (ValueError, FileNotFoundError, OpenTargetsError, StringError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        if "cached" in str(exc).lower() or "offline" in str(exc).lower():
            print("hint: run without --offline, or fetch the data first.", file=sys.stderr)
        if "api key" in str(exc).lower():
            print(
                "hint: export DEEPSEEK_API_KEY=..., or put it in .env, "
                "then rerun. Use --mode heuristic to run without an LLM.",
                file=sys.stderr,
            )
        return 2
    except KeyboardInterrupt:  # pragma: no cover
        print("interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
