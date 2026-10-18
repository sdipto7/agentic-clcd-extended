#!/usr/bin/env python3
"""
Pair-level complementarity between the direct and algorithm-based pipelines.

For each (dataset, model) combination with completed results in both
output/direct/ and output/algo_based/, this joins the two pipelines'
predictions by pair_id and classifies every shared pair into one of four
mutually-exclusive outcomes:

  - both_correct:            both pipelines matched the ground truth
  - direct_correct_algo_wrong: direct matched, algorithm-based did not
  - algo_correct_direct_wrong: algorithm-based matched, direct did not
  - both_wrong:               neither pipeline matched the ground truth

This reproduces the "Pair-level complementarity" table (tab:direct-algo-
complementarity), pooled across every model and the selected datasets. It
also reports the direct_correct_algo_wrong outcome broken down by
(dataset, language) - the case where the extra algorithm-extraction step
actively hurt rather than helped - since that is usually the outcome of
most interest when checking whether algorithmic abstraction is safe to
apply broadly.

By default this pools across ALL languages (ll2) present in the selected
result files. Pass --languages Python to restrict to Java-Python pairs
only (e.g. to reproduce the original two-dataset, Java-Python-only figures).

Only pair_ids present in BOTH files for a given (dataset, model) are
compared, so partially-completed runs are handled gracefully.

Usage:
    python scripts/compare_direct_vs_algo_based.py
    python scripts/compare_direct_vs_algo_based.py --languages Python
    python scripts/compare_direct_vs_algo_based.py --datasets codenet xlcost gptclonebench
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.constants import ERROR, LABEL_TO_VERDICT, MODEL_CHOICES, NOT_CLONE  # noqa: E402

DIRECT_DIR = PROJECT_ROOT / "output" / "direct"
ALGO_DIR = PROJECT_ROOT / "output" / "algo_based"
REPORTS_DIR = PROJECT_ROOT / "reports"

# The two datasets the original ad-hoc report covered; pass --datasets to widen this.
DEFAULT_DATASETS = ["codenet", "xlcost"]

_PAIR_ID_TRAILING_NUMBER_RE = re.compile(r"(\d+)$")

# [both_correct, direct_correct_algo_wrong, algo_correct_direct_wrong, both_wrong]
Cells = List[int]


def _empty_cells() -> Cells:
    return [0, 0, 0, 0]


def _add_cells(total: Cells, cells: Cells) -> None:
    for i in range(4):
        total[i] += cells[i]


def _model_from_filename(path: Path, dataset: str) -> str:
    """Recover the model alias from a ``results_<model>_<dataset>.csv`` filename."""
    stem = path.stem  # results_<model>_<dataset>
    prefix = "results_"
    suffix = f"_{dataset}"
    return stem[len(prefix) : -len(suffix)]


def _discover_models(dataset: str) -> List[str]:
    """
    Find model aliases with a completed results file for this dataset in both pipelines.

    Args:
        dataset: Dataset name (e.g., "codenet").

    Returns:
        Sorted list of model aliases present under both output/direct/ and
        output/algo_based/ for this dataset.
    """
    direct_models = {
        _model_from_filename(p, dataset) for p in DIRECT_DIR.glob(f"results_*_{dataset}.csv")
    }
    algo_models = {
        _model_from_filename(p, dataset) for p in ALGO_DIR.glob(f"results_*_{dataset}.csv")
    }

    return sorted(direct_models & algo_models)


def _load_predictions(path: Path) -> Dict[str, Tuple[int, str, str]]:
    """
    Load a results CSV into {pair_id: (ground_truth, predicted_label, ll2)}.

    Args:
        path: Path to a ``results_<model>_<dataset>.csv`` file.

    Returns:
        Mapping of pair_id to (ground_truth as int, predicted_label upper-cased, ll2).
    """
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return {
            row["pair_id"]: (
                int(row["ground_truth"]),
                row["predicted_label"].strip().upper(),
                row.get("ll2", "UNKNOWN"),
            )
            for row in reader
        }


def _natural_sort_key(pair_id: str) -> Tuple[str, int]:
    """Sort pair_ids like codenet_6 before codenet_12 (numeric suffix, not lexicographic)."""
    match = _PAIR_ID_TRAILING_NUMBER_RE.search(pair_id)
    if not match:
        return (pair_id, 0)

    prefix = pair_id[: match.start()]
    return (prefix, int(match.group(1)))


def _compare(
    model: str, dataset: str, languages: Optional[List[str]]
) -> Tuple[Cells, Dict[str, Cells], List[Dict[str, Any]]]:
    """
    Compare direct vs algo_based predictions for one (model, dataset) pair.

    Args:
        model: Model alias.
        dataset: Dataset name.
        languages: If given, only rows whose ``ll2`` is in this list are
            included; if None, every language present is included.

    Returns:
        A (cells, by_language, detail_rows) tuple:
        - cells: [both_correct, direct_correct_algo_wrong,
          algo_correct_direct_wrong, both_wrong] for this stratum.
        - by_language: the same four-cell breakdown, keyed by ll2.
        - detail_rows: one row per pair where direct was correct and
          algo_based was not (kept for qualitative pair-level inspection).

    Raises:
        ValueError: If the two files disagree on a shared pair_id's ground
            truth (which would indicate the two runs used different dataset
            states).
    """
    direct_path = DIRECT_DIR / f"results_{model}_{dataset}.csv"
    algo_path = ALGO_DIR / f"results_{model}_{dataset}.csv"

    direct_rows = _load_predictions(direct_path)
    algo_rows = _load_predictions(algo_path)

    shared_pair_ids = sorted(set(direct_rows) & set(algo_rows), key=_natural_sort_key)

    cells = _empty_cells()
    by_language: Dict[str, Cells] = {}
    detail_rows: List[Dict[str, Any]] = []

    for pair_id in shared_pair_ids:
        direct_gt, direct_pred, direct_ll2 = direct_rows[pair_id]
        algo_gt, algo_pred, _algo_ll2 = algo_rows[pair_id]

        if languages is not None and direct_ll2 not in languages:
            continue

        if direct_gt != algo_gt:
            raise ValueError(
                f"ground_truth mismatch for {pair_id} ({model}/{dataset}): "
                f"direct={direct_gt} algo_based={algo_gt}. The two result files "
                "appear to disagree about this pair's label."
            )

        gt_label = LABEL_TO_VERDICT.get(direct_gt, NOT_CLONE)

        # ERROR (or any unrecognized label) never counts as correct.
        direct_is_correct = direct_pred == gt_label and direct_pred != ERROR
        algo_is_correct = algo_pred == gt_label and algo_pred != ERROR

        if direct_is_correct and algo_is_correct:
            idx = 0
        elif direct_is_correct and not algo_is_correct:
            idx = 1
        elif not direct_is_correct and algo_is_correct:
            idx = 2
        else:
            idx = 3

        cells[idx] += 1
        by_language.setdefault(direct_ll2, _empty_cells())[idx] += 1

        if idx == 1:
            detail_rows.append(
                {
                    "model": model,
                    "dataset": dataset,
                    "ll2": direct_ll2,
                    "pair_id": pair_id,
                    "ground_truth": gt_label,
                    "direct_prediction": direct_pred,
                    "algo_based_prediction": algo_pred,
                    "direct_file": os.path.relpath(direct_path, PROJECT_ROOT),
                    "algo_based_file": os.path.relpath(algo_path, PROJECT_ROOT),
                }
            )

    return cells, by_language, detail_rows


def _write_detail_csv(path: Path, detail_rows: List[Dict[str, Any]]) -> None:
    """Write the per-pair detail CSV (one row per direct-correct/algo-wrong pair)."""
    fieldnames = [
        "model",
        "dataset",
        "ll2",
        "pair_id",
        "ground_truth",
        "direct_prediction",
        "algo_based_prediction",
        "direct_file",
        "algo_based_file",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(detail_rows)


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
        help=f"Datasets to compare (default: {DEFAULT_DATASETS}).",
    )
    parser.add_argument(
        "--languages",
        nargs="+",
        default=None,
        help="Restrict to these ll2 values only (e.g. --languages Python). "
        "Default: include every language present.",
    )
    args = parser.parse_args()

    overall_cells = _empty_cells()
    cells_by_dataset: Dict[str, Cells] = {d: _empty_cells() for d in args.datasets}
    cells_by_dataset_language: Dict[Tuple[str, str], Cells] = {}
    all_detail_rows: List[Dict[str, Any]] = []
    strata_used = 0

    for dataset in args.datasets:
        models = _discover_models(dataset)
        if not models:
            print(f"No matching direct/algo_based file pairs found for dataset={dataset!r}; skipping.")
            continue

        for model in models:
            if model not in MODEL_CHOICES:
                continue
            cells, by_language, detail_rows = _compare(model, dataset, args.languages)
            if sum(cells) == 0:
                continue

            strata_used += 1
            _add_cells(overall_cells, cells)
            _add_cells(cells_by_dataset[dataset], cells)
            for ll2, lang_cells in by_language.items():
                key = (dataset, ll2)
                cells_by_dataset_language.setdefault(key, _empty_cells())
                _add_cells(cells_by_dataset_language[key], lang_cells)
            all_detail_rows.extend(detail_rows)

    if strata_used == 0:
        print("No (model, dataset) combinations had completed results in both pipelines.")
        sys.exit(1)

    lang_desc = ", ".join(args.languages) if args.languages else "all languages present"

    header_lines = [
        "DIRECT VS ALGORITHM-BASED PAIR-LEVEL COMPLEMENTARITY",
        "=" * 60,
        f"Generated:  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Datasets:   {', '.join(args.datasets)}",
        f"Languages:  {lang_desc}",
        f"Models:     {', '.join(MODEL_CHOICES)}",
        "",
        "Methodology: for each (model, dataset) stratum, every pair_id present in",
        "both output/direct/ and output/algo_based/ results is classified into one",
        "of four mutually-exclusive outcomes by comparing each pipeline's",
        "predicted_label against the dataset's ground_truth label (an ERROR or",
        "unrecognized predicted_label is always treated as incorrect). The four",
        "outcomes are pooled across every model by summing counts (the same",
        "sum-then-report approach used for pooled F1 elsewhere in this project).",
        "",
    ]

    # 1. Overall complementarity table (matches tab:direct-algo-complementarity)
    outcome_labels = [
        "Both pipelines correct",
        "Direct correct, algorithm wrong",
        "Algorithm correct, direct wrong",
        "Both pipelines wrong",
    ]
    dataset_list = [d for d in args.datasets if sum(cells_by_dataset[d]) > 0]
    main_headers = ["Prediction Outcome"] + [d for d in dataset_list] + ["Total"]
    main_rows = []
    for i, label in enumerate(outcome_labels):
        row = [label]
        for d in dataset_list:
            row.append(str(cells_by_dataset[d][i]))
        row.append(str(overall_cells[i]))
        main_rows.append(row)
    main_rows.append(
        ["Total"] + [str(sum(cells_by_dataset[d])) for d in dataset_list] + [str(sum(overall_cells))]
    )
    main_table = _format_table(main_headers, main_rows)

    # 2. Focus: direct-correct/algo-wrong broken down by (dataset, language)
    focus_headers = ["Dataset", "Language", "Total Pairs", "Direct-right/Algo-wrong", "% of pairs", "% of direct-correct"]
    focus_rows = []
    for (dataset, ll2), cells in sorted(cells_by_dataset_language.items()):
        total = sum(cells)
        direct_correct_total = cells[0] + cells[1]
        focus_rows.append(
            [
                dataset,
                ll2,
                str(total),
                str(cells[1]),
                _pct(cells[1], total),
                _pct(cells[1], direct_correct_total),
            ]
        )
    focus_table = _format_table(focus_headers, focus_rows)

    latex_lines = ["-- Copy-ready table rows (overall complementarity) --"]
    for i, label in enumerate(outcome_labels):
        cells_per_dataset = " & ".join(f"{cells_by_dataset[d][i]:,}" for d in dataset_list)
        latex_lines.append(f"{label} & {cells_per_dataset} & {overall_cells[i]:,} \\\\")
    latex_lines.append(
        "\\textbf{Total} & "
        + " & ".join(f"\\textbf{{{sum(cells_by_dataset[d]):,}}}" for d in dataset_list)
        + f" & \\textbf{{{sum(overall_cells):,}}} \\\\"
    )

    sections = [
        "1. OVERALL COMPLEMENTARITY (pooled across all models)",
        "=" * 54,
        "",
        main_table,
        "",
        "2. FOCUS: DIRECT-CORRECT / ALGORITHM-WRONG, BY DATASET AND LANGUAGE",
        "=" * 68,
        "",
        "This is the case where the extra algorithm-extraction step actively hurt\n"
        "rather than helped - direct comparison already had the right answer, and\n"
        "converting to an algorithm representation changed it to the wrong one.\n"
        "'% of pairs' is relative to all pairs in that (dataset, language) stratum;\n"
        "'% of direct-correct' is relative only to pairs direct got right (i.e. of\n"
        "the pairs direct answered correctly, what fraction did algo then get wrong).",
        "",
        focus_table,
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
        "  always reflects the current contents of output/direct/ and\n"
        "  output/algo_based/ for the selected datasets and languages.\n"
        "- Pass --languages Python to reproduce the original Java-Python-only,\n"
        "  two-dataset figures.\n"
        "- A per-pair detail CSV listing every direct-correct/algo-wrong pair_id\n"
        "  is written alongside this report for qualitative inspection.",
    ]

    report_text = "\n".join(header_lines) + "\n" + "\n".join(sections) + "\n"

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    txt_path = REPORTS_DIR / f"direct_algo_complementarity_{stamp}.txt"
    csv_path = REPORTS_DIR / f"direct_correct_algo_wrong_pairs_{stamp}.csv"

    txt_path.write_text(report_text, encoding="utf-8")
    _write_detail_csv(csv_path, all_detail_rows)

    print(report_text)
    print(f"Written to: {txt_path}")
    print(f"Per-pair detail CSV: {csv_path}")


if __name__ == "__main__":
    main()
