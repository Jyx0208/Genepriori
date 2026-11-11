"""Output writers and console tables for ranking and evaluation results."""

from __future__ import annotations

import json
import platform
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Union

PathLike = Union[str, Path]


def _versions() -> Dict[str, str]:
    from .. import __version__

    try:
        import numpy
        import scipy

        numpy_version, scipy_version = numpy.__version__, scipy.__version__
    except Exception:  # pragma: no cover - numpy/scipy are hard dependencies
        numpy_version = scipy_version = "unknown"
    return {
        "genepriori": __version__,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": numpy_version,
        "scipy": scipy_version,
    }


def write_json(payload: Any, path: PathLike) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return target


def write_manifest(
    outdir: PathLike,
    *,
    command: str,
    arguments: Mapping[str, Any],
    config: Optional[Mapping[str, Any]] = None,
    inputs: Optional[Mapping[str, Any]] = None,
    outputs: Optional[Sequence[str]] = None,
    extra: Optional[Mapping[str, Any]] = None,
) -> Path:
    """Record versions, arguments and input provenance for reproducibility."""
    from ..data.cache import utc_now

    payload: Dict[str, Any] = {
        "command": command,
        "created_at": utc_now(),
        "versions": _versions(),
        "arguments": dict(arguments or {}),
    }
    if config is not None:
        payload["config"] = dict(config)
    if inputs is not None:
        payload["inputs"] = dict(inputs)
    if outputs is not None:
        payload["outputs"] = list(outputs)
    if extra:
        payload.update(dict(extra))
    return write_json(payload, Path(outdir) / "run_manifest.json")


def write_ranking(result, outdir: PathLike) -> List[Path]:
    """Write ``ranked_genes.csv`` and ``result.json`` for a RankedResult."""
    target = Path(outdir)
    target.mkdir(parents=True, exist_ok=True)
    csv_path = result.to_csv(target / "ranked_genes.csv")
    json_path = target / "result.json"
    write_json(result.to_dict(), json_path)
    return [csv_path, json_path]


def write_evaluation(result, outdir: PathLike, *, prefix: str = "") -> List[Path]:
    """Write ``metrics.json`` and ``folds.csv`` for an EvaluationResult."""
    target = Path(outdir)
    target.mkdir(parents=True, exist_ok=True)
    stem = f"{prefix}" if not prefix else f"{prefix}_"
    json_path = result.to_json(target / f"{stem}metrics.json")
    csv_path = result.to_csv(target / f"{stem}folds.csv")
    return [json_path, csv_path]


def format_table(
    headers: Sequence[str],
    rows: Sequence[Sequence[Any]],
    *,
    max_col_width: int = 32,
) -> str:
    """Render a simple fixed-width table."""
    def cell(value: Any) -> str:
        text = "" if value is None else str(value)
        return text if len(text) <= max_col_width else text[: max_col_width - 1] + "~"

    body = [[cell(v) for v in row] for row in rows]
    widths = [len(h) for h in headers]
    for row in body:
        for i, value in enumerate(row):
            widths[i] = max(widths[i], len(value))
    lines = ["  ".join(h.ljust(widths[i]) for i, h in enumerate(headers)).rstrip()]
    lines.append("  ".join("-" * widths[i] for i in range(len(headers))))
    for row in body:
        lines.append("  ".join(value.ljust(widths[i]) for i, value in enumerate(row)).rstrip())
    return "\n".join(lines)


def ranked_table(result, *, top: int = 20) -> str:
    rows = [
        [
            row.rank,
            row.gene,
            f"{row.score:.6g}",
            f"{row.confidence:.2f}",
            row.n_evidence_types,
            "yes" if row.network_discovered else "",
        ]
        for row in result.top(top)
    ]
    return format_table(
        ["rank", "gene", "score", "confidence", "types", "network"],
        rows,
    )


def evaluation_table(
    results: Iterable,
    *,
    metric: str = "auprc",
    k: int = 20,
) -> str:
    rows = []
    for result in results:
        summary = result.summary.get(metric, {})
        recall_key = f"recall_at_{k}"
        recall = result.summary.get(recall_key, {})
        rows.append(
            [
                result.disease_name,
                result.holdout,
                result.protocol,
                f"{summary.get('mean', 0.0):.4f}",
                f"{summary.get('sd', 0.0):.4f}",
                f"{summary.get('ci_low', 0.0):.4f}",
                f"{summary.get('ci_high', 0.0):.4f}",
                f"{recall.get('mean', 0.0):.4f}",
            ]
        )
    return format_table(
        [metric, "holdout", "protocol", "mean", "sd", "ci_low", "ci_high", f"R@{k}"],
        rows,
    )
