#!/usr/bin/env python3
"""
Compute pooled clone-detection performance for the direct/algo_based/agentic pipelines.

"Pooled" performance means: instead of averaging per-(model, dataset) precision/
recall/F1 values, first sum the raw confusion counts (TP, TN, FP, FN) across every
run in the group, then derive precision/recall/F1 once from those combined counts.
This is the same methodology already described in the thesis chapter (see the
pooled-performance equations in chap4_agentic_clcd.tex) and matches how the
"Overall pooled performance" and "Model-level pooled F1-score" figures were built.

This script reproduces both of those figures directly from output/:

1. Overall pooled performance per pipeline - precision, recall, F1 pooled across
   every model and every selected dataset (matches Figure panel (a)).
2. Model-level pooled F1-score per pipeline - F1 pooled across every selected
   dataset, broken out per model (matches Figure panel (b)).

By default this pools across ALL languages (ll2) present in the selected result
files, since new experiments now include C++ and C# pairs alongside Python. Pass
--languages Python to restrict to Java-Python pairs only (e.g., to reproduce or
sanity-check earlier Java-Python-only figures against a fresh run).

Usage:
    python scripts/analyze_pooled_performance.py
    python scripts/analyze_pooled_performance.py --languages Python
    python scripts/analyze_pooled_performance.py --datasets xlcost codenet gptclonebench
"""

from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.constants import CLONE, ERROR, NOT_CLONE, PIPELINE_CHOICES  # noqa: E402

OUTPUT_DIR = PROJECT_ROOT / "output"
REPORTS_DIR = PROJECT_ROOT / "reports"

# The two datasets every model/pipeline combination was originally evaluated on;
# gptclonebench/clcdsa are supplementary and only partially populated for some
# models, so they are excluded by default (pass --datasets to widen this).
DEFAULT_DATASETS = ["xlcost", "codenet"]


def _model_dataset_from_filename(path: Path) -> Tuple[str, str]:
    """Recover (model, dataset) from a ``results_<model>_<dataset>.csv`` filename."""
    stem = path.stem[len("results_") :]
    model, dataset = stem.rsplit("_", 1)
    return model, dataset


def _discover_result_files(pipeline: str, datasets: List[str]) -> List[Tuple[str, str, Path]]:
    """
    Find result files for one pipeline restricted to the given datasets.

    Args:
        pipeline: Pipeline name (must be a key under output/).
        datasets: Dataset names to include.

    Returns:
        List of (model, dataset, path) tuples, sorted by (dataset, model).
    """
    pipeline_dir = OUTPUT_DIR / pipeline
    found = []
    for path in sorted(pipeline_dir.glob("results_*.csv")):
        model, dataset = _model_dataset_from_filename(path)
        if dataset in datasets:
            found.append((model, dataset, path))

    return sorted(found, key=lambda t: (t[1], t[0]))


def _bump_confusion(counts: List[int], ground_truth: int, predicted_label: str) -> None:
    """
    Accumulate one row into a [TP, TN, FP, FN] list in place.

    A prediction that is neither CLONE nor NOT_CLONE (i.e., ERROR, or any other
    unrecognized value) is always counted as incorrect: a false negative if the
    true label is a clone, a false positive otherwise.
    """
    if predicted_label not in (CLONE, NOT_CLONE):
        if ground_truth == 1:
            counts[3] += 1
        else:
            counts[2] += 1
        return

    if ground_truth == 1 and predicted_label == CLONE:
        counts[0] += 1
    elif ground_truth == 1 and predicted_label == NOT_CLONE:
        counts[3] += 1
    elif ground_truth == 0 and predicted_label == CLONE:
        counts[2] += 1
    elif ground_truth == 0 and predicted_label == NOT_CLONE:
        counts[1] += 1


def _metrics_from_confusion(counts: List[int]) -> Dict[str, Any]:
    """Derive total/accuracy/precision/recall/F1 from a [TP, TN, FP, FN] list."""
    tp, tn, fp, fn = counts
    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    return {
        "total": total,
        "true_positive": tp,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
    }


def _pool_pipeline_and_model(
    pipeline: str, datasets: List[str], languages: Optional[List[str]]
) -> Tuple[List[int], Dict[str, List[int]]]:
    """
    Pool confusion counts for one pipeline, overall and broken out per model.

    Args:
        pipeline: Pipeline name.
        datasets: Dataset names to include.
        languages: If given, only rows whose ``ll2`` is in this list are counted;
            if None, every language present is included.

    Returns:
        A (overall_counts, per_model_counts) tuple, where overall_counts is a
        [TP, TN, FP, FN] list pooled across every model and dataset, and
        per_model_counts maps model alias to its own pooled [TP, TN, FP, FN]
        list (pooled across the selected datasets only).
    """
    overall = [0, 0, 0, 0]
    per_model: Dict[str, List[int]] = {}

    for model, _dataset, path in _discover_result_files(pipeline, datasets):
        per_model.setdefault(model, [0, 0, 0, 0])
        with path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if languages is not None and row.get("ll2") not in languages:
                    continue

                gt = int(row["ground_truth"])
                pred = row["predicted_label"].strip().upper()
                _bump_confusion(overall, gt, pred)
                _bump_confusion(per_model[model], gt, pred)

    return overall, per_model


def _pct(value: float) -> str:
    """Format a 0-1 fraction as a percentage string with one decimal place."""
    return f"{value * 100:.1f}%"


def _pct_int(value: float) -> str:
    """Format a 0-1 fraction as a rounded whole-number percentage (matches the tikz charts)."""
    return f"{round(value * 100)}"


def _format_table(headers: List[str], rows: List[List[str]]) -> str:
    """Render a left-aligned, fixed-width monospace table."""
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))

    def fmt(cells: List[str]) -> str:
        return "  ".join(str(c).ljust(widths[i]) for i, c in enumerate(cells))

    lines = [fmt(headers), "  ".join("-" * w for w in widths)]
    lines.extend(fmt(r) for r in rows)
    return "\n".join(lines)


def _build_report_text(
    datasets: List[str],
    languages: Optional[List[str]],
    overall_by_pipeline: Dict[str, List[int]],
    per_model_by_pipeline: Dict[str, Dict[str, List[int]]],
) -> str:
    """Assemble the full report text from the pooled confusion counts."""
    pipelines = [p for p in PIPELINE_CHOICES if p in overall_by_pipeline]
    models = sorted({m for pm in per_model_by_pipeline.values() for m in pm})

    lang_desc = ", ".join(languages) if languages else "all languages present"

    lines = [
        "POOLED CLONE-DETECTION PERFORMANCE ANALYSIS",
        "=" * 60,
        f"Generated:  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Datasets:   {', '.join(datasets)}",
        f"Languages:  {lang_desc}",
        f"Pipelines:  {', '.join(pipelines)}",
        f"Models:     {', '.join(models)}",
        "",
        "Methodology: pooled metrics are computed by first summing raw",
        "TP/TN/FP/FN confusion counts across every run in the group, then",
        "deriving precision/recall/F1 once from those combined counts - not by",
        "averaging per-run metrics. This matches the pooling formulas in",
        "chap4_agentic_clcd.tex (TP_pooled = sum(TP_i), etc.). A prediction that",
        "is not CLONE or NOT_CLONE (e.g. ERROR) always counts as incorrect: a",
        "false negative if the pair is a true clone, a false positive otherwise.",
        "",
    ]

    # --- Section 1: Overall pooled performance (Figure panel a) ---
    lines.append("1. OVERALL POOLED PERFORMANCE (pooled across all models and datasets)")
    lines.append("=" * 72)
    lines.append("")
    header = ["Pipeline", "Total", "Precision", "Recall", "F1", "TP", "TN", "FP", "FN"]
    rows = []
    for p in pipelines:
        m = _metrics_from_confusion(overall_by_pipeline[p])
        rows.append(
            [
                p,
                str(m["total"]),
                _pct(m["precision"]),
                _pct(m["recall"]),
                _pct(m["f1_score"]),
                str(m["true_positive"]),
                str(m["true_negative"]),
                str(m["false_positive"]),
                str(m["false_negative"]),
            ]
        )
    lines.append(_format_table(header, rows))
    lines.append("")

    # --- Section 2: Model-level pooled F1 (Figure panel b) ---
    lines.append("2. MODEL-LEVEL POOLED F1-SCORE (pooled across selected datasets, per model)")
    lines.append("=" * 76)
    lines.append("")
    header2 = ["Model", *[p for p in pipelines]]
    rows2 = []
    for model in models:
        row = [model]
        for p in pipelines:
            counts = per_model_by_pipeline.get(p, {}).get(model, [0, 0, 0, 0])
            m = _metrics_from_confusion(counts)
            row.append(_pct(m["f1_score"]))
        rows2.append(row)
    lines.append(_format_table(header2, rows2))
    lines.append("")

    # --- Section 3: LaTeX-ready tikz coordinates ---
    lines.append("3. LATEX-READY VALUES (rounded to nearest whole percent, matches tikz charts)")
    lines.append("=" * 78)
    lines.append("")
    lines.append("-- Panel (a): Overall pooled performance --")
    for metric_name, metric_key in [("Precision", "precision"), ("Recall", "recall"), ("F1", "f1_score")]:
        coords = []
        for p in pipelines:
            m = _metrics_from_confusion(overall_by_pipeline[p])
            coords.append(f"({p},{_pct_int(m[metric_key])})")
        lines.append(f"% --- {metric_name}")
        lines.append("\\addplot+[...] coordinates {")
        lines.append("    " + "\n    ".join(coords))
        lines.append("};")
        lines.append("")

    lines.append("-- Panel (b): Model-level pooled F1-score --")
    for p in pipelines:
        coords = []
        for model in models:
            counts = per_model_by_pipeline.get(p, {}).get(model, [0, 0, 0, 0])
            m = _metrics_from_confusion(counts)
            coords.append(f"({model},{_pct_int(m['f1_score'])})")
        lines.append(f"% --- {p}")
        lines.append("\\addplot+[...] coordinates {")
        lines.append("    " + "\n    ".join(coords))
        lines.append("};")
        lines.append("")

    # --- Section 4: Notes ---
    lines.append("4. NOTES")
    lines.append("=" * 8)
    lines.append("")
    lines.append(
        "- Rerun this script any time after new experiment results land; it always\n"
        "  reflects the current contents of output/ for the selected datasets and\n"
        "  languages, so the numbers above will shift as more runs complete.\n"
        "- Pass --languages Python to restrict to Java-Python pairs only (useful for\n"
        "  comparing against earlier, pre-multi-language figures).\n"
        "- Pass --datasets to widen or narrow which datasets are pooled together;\n"
        "  the default (xlcost, codenet) matches the two datasets every model was\n"
        "  originally evaluated on across all three pipelines."
    )

    return "\n".join(lines) + "\n"


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=DEFAULT_DATASETS,
        help=f"Datasets to pool together (default: {DEFAULT_DATASETS}).",
    )
    parser.add_argument(
        "--languages",
        nargs="+",
        default=None,
        help="Restrict to these ll2 values only (e.g. --languages Python). "
        "Default: include every language present.",
    )
    args = parser.parse_args()

    overall_by_pipeline: Dict[str, List[int]] = {}
    per_model_by_pipeline: Dict[str, Dict[str, List[int]]] = {}

    for pipeline in PIPELINE_CHOICES:
        overall, per_model = _pool_pipeline_and_model(pipeline, args.datasets, args.languages)
        if overall == [0, 0, 0, 0]:
            print(f"No result rows found for pipeline={pipeline!r} with the selected filters; skipping.")
            continue
        overall_by_pipeline[pipeline] = overall
        per_model_by_pipeline[pipeline] = per_model

    if not overall_by_pipeline:
        print("No matching result rows found for any pipeline.")
        sys.exit(1)

    report_text = _build_report_text(args.datasets, args.languages, overall_by_pipeline, per_model_by_pipeline)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = REPORTS_DIR / f"pooled_performance_analysis_{stamp}.txt"
    out_path.write_text(report_text, encoding="utf-8")

    print(report_text)
    print(f"Written to: {out_path}")


if __name__ == "__main__":
    main()
