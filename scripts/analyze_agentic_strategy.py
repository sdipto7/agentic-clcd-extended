#!/usr/bin/env python3
"""
Analyze which reasoning route the agentic pipeline selected for each pair.

The agentic pipeline never records which route it chose as an explicit field -
instead, route selection is inferred from a side effect: the agent only calls
the record_algorithms tool (which writes an entry to
output/agentic/algorithms_<model>_<dataset>.json) when it selects the
algorithm-based route. So for each pair_id in a results_<model>_<dataset>.csv
file:

  - pair_id appears in the matching algorithms_<model>_<dataset>.json
    -> algorithm-based comparison was used.
  - pair_id does not appear there
    -> direct comparison was used (the agent never extracted/recorded
       algorithms for this pair).

This reproduces the "Agentic strategy selection" table (Direct vs
Algorithm-based counts and percentages) reported in the chapter.

By default this pools across ALL languages (ll2) present in the selected
result files. Pass --languages Python to restrict to Java-Python pairs only
(e.g., to reproduce or sanity-check the original Java-Python-only figures).

Usage:
    python scripts/analyze_agentic_strategy.py
    python scripts/analyze_agentic_strategy.py --languages Python
    python scripts/analyze_agentic_strategy.py --datasets xlcost codenet gptclonebench
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.constants import MODEL_CHOICES, PIPELINE_AGENTIC  # noqa: E402

AGENTIC_DIR = PROJECT_ROOT / "output" / PIPELINE_AGENTIC
REPORTS_DIR = PROJECT_ROOT / "reports"

# The two datasets every model was originally evaluated on; pass --datasets to widen this.
DEFAULT_DATASETS = ["xlcost", "codenet"]


def _load_algorithm_pair_ids(model: str, dataset: str) -> set[str]:
    """Return the set of pair_ids recorded in algorithms_<model>_<dataset>.json, if it exists."""
    path = AGENTIC_DIR / f"algorithms_{model}_{dataset}.json"
    if not path.is_file():
        return set()

    with path.open(encoding="utf-8") as f:
        data = json.load(f)

    return set(data.keys())


def _classify_stratum(model: str, dataset: str, languages: Optional[List[str]]) -> Optional[Dict[str, int]]:
    """
    Classify every pair in one (model, dataset) results file as direct or algorithm-based.

    Args:
        model: Model alias.
        dataset: Dataset name.
        languages: If given, only rows whose ``ll2`` is in this list are counted;
            if None, every row is included regardless of language.

    Returns:
        {"direct": n, "algorithm_based": n, "total": n}, or None if the results
        file for this (model, dataset) doesn't exist.
    """
    results_path = AGENTIC_DIR / f"results_{model}_{dataset}.csv"
    if not results_path.is_file():
        return None

    algorithm_pair_ids = _load_algorithm_pair_ids(model, dataset)

    direct = 0
    algorithm_based = 0
    with results_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if languages is not None and row.get("ll2") not in languages:
                continue

            if row["pair_id"] in algorithm_pair_ids:
                algorithm_based += 1
            else:
                direct += 1

    return {"direct": direct, "algorithm_based": algorithm_based, "total": direct + algorithm_based}


def _add(total: Dict[str, int], counts: Dict[str, int]) -> None:
    """Accumulate one stratum's counts into a running total, in place."""
    for key in ("direct", "algorithm_based", "total"):
        total[key] += counts[key]


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


def _pct(count: int, total: int) -> str:
    return f"{(count / total * 100):.2f}%" if total else "0.00%"


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

    overall: Dict[str, int] = {"direct": 0, "algorithm_based": 0, "total": 0}
    per_model_dataset_rows: List[List[str]] = []

    for dataset in args.datasets:
        for model in MODEL_CHOICES:
            counts = _classify_stratum(model, dataset, args.languages)
            if counts is None:
                continue

            _add(overall, counts)
            per_model_dataset_rows.append(
                [
                    model,
                    dataset,
                    str(counts["total"]),
                    f"{counts['direct']} ({_pct(counts['direct'], counts['total'])})",
                    f"{counts['algorithm_based']} ({_pct(counts['algorithm_based'], counts['total'])})",
                ]
            )

    if overall["total"] == 0:
        print("No matching agentic result files found for the selected datasets/languages.")
        sys.exit(1)

    lang_desc = ", ".join(args.languages) if args.languages else "all languages present"

    header_lines = [
        "AGENTIC STRATEGY SELECTION ANALYSIS",
        "=" * 60,
        f"Generated:  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Datasets:   {', '.join(args.datasets)}",
        f"Languages:  {lang_desc}",
        f"Models:     {', '.join(MODEL_CHOICES)}",
        "",
        "Methodology: a pair_id is classified as 'algorithm-based' if it appears",
        "in the matching output/agentic/algorithms_<model>_<dataset>.json file",
        "(written only when the agent called record_algorithms), and 'direct'",
        "otherwise. This reflects an observed side effect of route selection,",
        "not an explicit logged field.",
        "",
    ]

    overall_table = _format_table(
        ["Selected Strategy", "Usage"],
        [
            ["Direct comparison", f"{overall['direct']} ({_pct(overall['direct'], overall['total'])})"],
            [
                "Algorithm-based comparison",
                f"{overall['algorithm_based']} ({_pct(overall['algorithm_based'], overall['total'])})",
            ],
            ["Total", f"{overall['total']} (100.00%)"],
        ],
    )

    breakdown_table = _format_table(
        ["Model", "Dataset", "Total", "Direct", "Algorithm-based"], per_model_dataset_rows
    )

    latex_lines = [
        "-- Copy-ready table rows --",
        f"Direct comparison & {overall['direct']} ({_pct(overall['direct'], overall['total'])}) \\\\",
        f"Algorithm-based comparison & {overall['algorithm_based']} "
        f"({_pct(overall['algorithm_based'], overall['total'])}) \\\\",
        f"Total & {overall['total']} (100.00\\%) \\\\",
    ]

    sections = [
        "1. OVERALL (pooled across all models and selected datasets)",
        "=" * 60,
        "",
        overall_table,
        "",
        "2. PER-MODEL, PER-DATASET BREAKDOWN",
        "=" * 36,
        "",
        breakdown_table,
        "",
        "3. LATEX-READY VALUES",
        "=" * 22,
        "",
        "\n".join(latex_lines),
        "",
        "4. NOTES",
        "=" * 8,
        "",
        "- Rerun this script any time after new experiment results land; it\n"
        "  always reflects the current contents of output/agentic/ for the\n"
        "  selected datasets and languages.\n"
        "- Pass --languages Python to restrict to Java-Python pairs only.\n"
        "- A pair recorded as ERROR (no confirmed verdict) is still classified\n"
        "  by whichever route left a trace: if record_algorithms was called\n"
        "  before the failure, it counts as algorithm-based; otherwise direct.",
    ]

    report_text = "\n".join(header_lines) + "\n" + "\n".join(sections) + "\n"

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = REPORTS_DIR / f"agentic_strategy_analysis_{stamp}.txt"
    out_path.write_text(report_text, encoding="utf-8")

    print(report_text)
    print(f"Written to: {out_path}")


if __name__ == "__main__":
    main()
