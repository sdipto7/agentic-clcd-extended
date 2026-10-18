#!/usr/bin/env python3
"""
Compute pooled (micro-averaged) precision/recall/F1 for each baseline across
all evaluated datasets.

Each baseline (nicad_source, nicad_algorithms, codebert, clcdsa) is run
separately per dataset (xlcost, codenet, gptclonebench), producing one
results_*.csv per (baseline, dataset) pair - see evaluate_baselines.py for the
canonical list of these files. This script pools each baseline's per-dataset
confusion counts (true_positive/true_negative/false_positive/false_negative)
by summing them across datasets, then computes precision/recall/F1 once from
the pooled counts. This is the exact same micro-averaging methodology already
used and cross-validated (against scikit-learn) for the LLM pipelines in
scripts/analyze_pooled_performance.py: pool raw confusion counts first, then
derive metrics from the pooled counts - never average the per-dataset
percentages directly, since that would implicitly (and incorrectly) weight
every dataset equally regardless of how many pairs it contributes.

graphcodebert is intentionally excluded: no run script or result files exist
for it anywhere in this repository.

Usage:
    python scripts/analyze_pooled_baseline_performance.py
    python scripts/analyze_pooled_baseline_performance.py --datasets xlcost codenet
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluate import _evaluate_file  # noqa: E402
from evaluate_baselines import BASELINE_FILES  # noqa: E402

REPORTS_DIR = PROJECT_ROOT / "reports"

# Every dataset any baseline has been run on; pass --datasets to narrow this.
DEFAULT_DATASETS = ["xlcost", "codenet", "gptclonebench"]

CONFUSION_FIELDS = ["true_positive", "true_negative", "false_positive", "false_negative"]


def _empty_counts() -> Dict[str, int]:
    return {field: 0 for field in CONFUSION_FIELDS} | {"total": 0}


def _metrics_from_confusion(counts: Dict[str, int]) -> Dict[str, float]:
    """Derive accuracy/precision/recall/F1 from pooled confusion counts."""
    tp, tn = counts["true_positive"], counts["true_negative"]
    fp, fn = counts["false_positive"], counts["false_negative"]
    total = counts["total"]

    accuracy = (tp + tn) / total if total else 0.0
    precision_denom = tp + fp
    recall_denom = tp + fn
    precision = tp / precision_denom if precision_denom else 0.0
    recall = tp / recall_denom if recall_denom else 0.0
    f1_denom = precision + recall
    f1_score = (2 * precision * recall / f1_denom) if f1_denom else 0.0

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1_score": f1_score}


def _baseline_dataset_paths(datasets: List[str]) -> Dict[str, Dict[str, Path]]:
    """Build {baseline_name: {dataset: path}} from evaluate_baselines.BASELINE_FILES."""
    by_baseline: Dict[str, Dict[str, Path]] = {}
    for baseline, _model, dataset, path in BASELINE_FILES:
        if dataset not in datasets:
            continue
        by_baseline.setdefault(baseline, {})[dataset] = path
    return by_baseline


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


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=DEFAULT_DATASETS,
        help=f"Datasets to pool together (default: {DEFAULT_DATASETS}).",
    )
    args = parser.parse_args()

    baseline_paths = _baseline_dataset_paths(args.datasets)
    if not baseline_paths:
        print("No baselines found for the selected datasets.")
        sys.exit(1)

    per_dataset_rows: List[List[str]] = []
    pooled_rows: List[List[str]] = []
    missing: List[Path] = []

    for baseline in sorted(baseline_paths):
        pooled_counts = _empty_counts()
        for dataset in args.datasets:
            path = baseline_paths[baseline].get(dataset)
            if path is None:
                continue
            if not path.is_file():
                missing.append(path)
                continue

            report = _evaluate_file(str(path))
            for field in CONFUSION_FIELDS:
                pooled_counts[field] += int(report[field])
            pooled_counts["total"] += int(report["total"])

            per_dataset_rows.append(
                [
                    baseline,
                    dataset,
                    report["total"],
                    report["precision"],
                    report["recall"],
                    report["f1_score"],
                ]
            )

        if pooled_counts["total"] == 0:
            continue

        pooled_metrics = _metrics_from_confusion(pooled_counts)
        pooled_rows.append(
            [
                baseline,
                str(pooled_counts["total"]),
                str(pooled_counts["true_positive"]),
                str(pooled_counts["true_negative"]),
                str(pooled_counts["false_positive"]),
                str(pooled_counts["false_negative"]),
                f"{pooled_metrics['precision']:.4f}",
                f"{pooled_metrics['recall']:.4f}",
                f"{pooled_metrics['f1_score']:.4f}",
            ]
        )

    if missing:
        print("Missing result files (skipped):")
        for path in missing:
            print(f"  - {path}")
        print()

    if not pooled_rows:
        print("No baseline result files found for the selected datasets.")
        sys.exit(1)

    header_lines = [
        "POOLED BASELINE PERFORMANCE ANALYSIS",
        "=" * 60,
        f"Generated:  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Datasets:   {', '.join(args.datasets)}",
        "",
        "Methodology: for each baseline, true_positive/true_negative/",
        "false_positive/false_negative are summed across every selected",
        "dataset's results CSV, and precision/recall/F1 are computed once from",
        "the pooled counts (micro-averaging) - not by averaging each dataset's",
        "own precision/recall/F1 percentages. This is the same pooling method",
        "already used and validated against scikit-learn for the LLM pipelines",
        "in scripts/analyze_pooled_performance.py. All baseline result files",
        "are restricted to Java-Python pairs only (see baselines/*/run_*.py),",
        "so this pools across datasets, not across languages.",
        "",
    ]

    per_dataset_table = _format_table(
        ["Baseline", "Dataset", "Pairs", "Prec.", "Rec.", "F1"], per_dataset_rows
    )

    pooled_table = _format_table(
        ["Baseline", "Pairs", "TP", "TN", "FP", "FN", "Prec.", "Rec.", "F1"], pooled_rows
    )

    latex_lines = ["-- Copy-ready table rows (pooled across datasets) --"]
    for row in pooled_rows:
        baseline, pairs, tp, tn, fp, fn, prec, rec, f1 = row
        latex_lines.append(f"{baseline} & {prec} & {rec} & {f1} \\\\")

    sections = [
        "1. PER-DATASET BREAKDOWN (input to pooling)",
        "=" * 45,
        "",
        per_dataset_table,
        "",
        "2. POOLED PERFORMANCE PER BASELINE (across all selected datasets)",
        "=" * 66,
        "",
        pooled_table,
        "",
        "3. LATEX-READY VALUES",
        "=" * 22,
        "",
        "\n".join(latex_lines),
        "",
        "4. NOTES",
        "=" * 8,
        "",
        "- Rerun this script any time after new baseline results land; it\n"
        "  always reflects the current contents of output/{nicad_source,\n"
        "  nicad_algorithms,codebert,clcdsa}/ for the selected datasets.\n"
        "- Pass --datasets to pool a subset (e.g. only xlcost codenet, to\n"
        "  reproduce the original two-dataset RQ1 comparison).\n"
        "- graphcodebert is excluded: no run script or results exist for it.\n"
        "- Pooling combines confusion counts before computing metrics, so\n"
        "  larger datasets (e.g. gptclonebench and xlcost at 1,200 pairs each)\n"
        "  naturally carry more weight than smaller ones (codenet at 600\n"
        "  pairs) in the pooled figure - this matches how the LLM pipelines'\n"
        "  pooled performance is computed elsewhere in this project.",
    ]

    report_text = "\n".join(header_lines) + "\n" + "\n".join(sections) + "\n"

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = REPORTS_DIR / f"pooled_baseline_performance_{stamp}.txt"
    out_path.write_text(report_text, encoding="utf-8")

    print(report_text)
    print(f"Written to: {out_path}")


if __name__ == "__main__":
    main()
