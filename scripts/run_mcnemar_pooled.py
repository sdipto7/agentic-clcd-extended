#!/usr/bin/env python3
"""
Run McNemar's exact test on pipeline comparisons, pooled across all LLM models.

For each pair of pipelines (Direct vs Algorithm, Direct vs Agentic, Algorithm vs
Agentic), this compares per-pair correctness on shared pair_ids and classifies
each pair into one of four cells: both correct, only pipeline A correct, only
pipeline B correct, or both wrong. McNemar's exact (binomial) test then checks
whether the two discordant cells (A-correct/B-wrong vs A-wrong/B-correct) are
asymmetric enough to call one pipeline significantly better than the other.

Unlike the original Claude-only version of this script, each comparison is
pooled across every model AND every selected dataset before the test is run -
the same scope used for the pooled F1/precision/recall numbers reported for
RQ1/RQ2 - so the output is a single overall result per comparison, not a
per-model or per-dataset breakdown.

Pooling here means summing the four correctness-cell counts across every
(model, dataset) stratum before running a single exact McNemar test on the
combined discordant counts - the same "sum raw counts, then compute once"
approach already used for pooled F1/precision/recall elsewhere in this project.
One caveat worth noting in the chapter text: this treats each (pair_id, model)
combination as an independent trial, which is a reasonable simplification but
ignores that a given pair_id may be inherently harder for every model at once.

By default this pools across ALL languages (ll2) present in the selected
result files. Pass --languages Python to restrict to Java-Python pairs only.

Usage:
    python scripts/run_mcnemar_pooled.py
    python scripts/run_mcnemar_pooled.py --languages Python
    python scripts/run_mcnemar_pooled.py --datasets xlcost codenet gptclonebench
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.constants import CLONE, MODEL_CHOICES, NOT_CLONE  # noqa: E402

OUTPUT_PATH = PROJECT_ROOT / "reports" / "mcnemar_pooled_all_models.csv"

# The two datasets every model/pipeline combination was originally evaluated on;
# pass --datasets to widen this (e.g. to include gptclonebench).
DEFAULT_DATASETS = ["xlcost", "codenet"]

PIPELINES = {
    "Direct": PROJECT_ROOT / "output" / "direct",
    "Algorithm": PROJECT_ROOT / "output" / "algo_based",
    "Agentic": PROJECT_ROOT / "output" / "agentic",
}
COMPARISONS = (
    ("Direct", "Algorithm"),
    ("Direct", "Agentic"),
    ("Algorithm", "Agentic"),
)

# [both_correct, a_correct_b_wrong, a_wrong_b_correct, both_wrong]
Cells = List[int]


def normalize_prediction(value: str) -> int:
    """Map a predicted_label string to 1 (clone), 0 (not clone), or -1 (unrecognized/ERROR)."""
    pred = value.strip().upper()
    if pred == CLONE:
        return 1
    if pred == NOT_CLONE:
        return 0
    return -1


def load_correctness(path: Path, languages: Optional[List[str]]) -> Dict[str, bool]:
    """
    Load per-pair correctness from one results CSV, optionally filtered by language.

    Args:
        path: Path to a ``results_<model>_<dataset>.csv`` file.
        languages: If given, only rows whose ``ll2`` is in this list are included;
            if None, every row is included regardless of language.

    Returns:
        Mapping of pair_id to whether the prediction matched ground truth.
        An unrecognized/ERROR prediction is always treated as incorrect.
    """
    rows: Dict[str, bool] = {}
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if languages is not None and row.get("ll2") not in languages:
                continue
            ground_truth = int(row["ground_truth"])
            predicted = normalize_prediction(row["predicted_label"])
            rows[row["pair_id"]] = predicted == ground_truth

    return rows


def exact_mcnemar_p_value(a_correct_b_wrong: int, a_wrong_b_correct: int) -> float:
    """
    Compute McNemar's exact two-sided p-value from the two discordant cell counts.

    Uses the exact binomial form (appropriate for any sample size, including the
    small discordant counts typical here) rather than the chi-squared
    approximation: under the null hypothesis that the two pipelines are equally
    likely to be the one that's "right" on a discordant pair, each discordant
    pair is an independent Bernoulli(0.5) trial.

    Args:
        a_correct_b_wrong: Count of pairs where A was correct and B was not.
        a_wrong_b_correct: Count of pairs where B was correct and A was not.

    Returns:
        Two-sided p-value in [0, 1]. Returns 1.0 when there are no discordant
        pairs (the test is uninformative in that case).
    """
    disagreements = a_correct_b_wrong + a_wrong_b_correct
    if disagreements == 0:
        return 1.0

    smaller = min(a_correct_b_wrong, a_wrong_b_correct)
    tail_probability = sum(math.comb(disagreements, i) for i in range(smaller + 1)) / (2**disagreements)

    return min(1.0, 2 * tail_probability)


def result_path(pipeline: str, model: str, dataset: str) -> Path:
    """Build the results CSV path for one (pipeline, model, dataset) combination."""
    return PIPELINES[pipeline] / f"results_{model}_{dataset}.csv"


def compare_stratum(
    pipeline_a: str, pipeline_b: str, model: str, dataset: str, languages: Optional[List[str]]
) -> Optional[Cells]:
    """
    Compute the four correctness-cell counts for one (model, dataset) stratum.

    Args:
        pipeline_a: First pipeline name (key into PIPELINES).
        pipeline_b: Second pipeline name.
        model: Model alias.
        dataset: Dataset name.
        languages: Optional ll2 filter (see ``load_correctness``).

    Returns:
        [both_correct, a_correct_b_wrong, a_wrong_b_correct, both_wrong], or
        None if either pipeline is missing a results file for this
        (model, dataset) combination.
    """
    path_a = result_path(pipeline_a, model, dataset)
    path_b = result_path(pipeline_b, model, dataset)
    if not path_a.is_file() or not path_b.is_file():
        return None

    correctness_a = load_correctness(path_a, languages)
    correctness_b = load_correctness(path_b, languages)
    shared_ids = sorted(set(correctness_a) & set(correctness_b))

    cells: Cells = [0, 0, 0, 0]
    for pair_id in shared_ids:
        a_correct = correctness_a[pair_id]
        b_correct = correctness_b[pair_id]
        if a_correct and b_correct:
            cells[0] += 1
        elif a_correct and not b_correct:
            cells[1] += 1
        elif not a_correct and b_correct:
            cells[2] += 1
        else:
            cells[3] += 1

    return cells


def _add(total: Cells, cells: Cells) -> None:
    """Accumulate one stratum's cell counts into a running total, in place."""
    for i in range(4):
        total[i] += cells[i]


def _report_row(comparison: str, cells: Cells) -> Dict[str, object]:
    """Build one output CSV row (the overall pooled result) from a comparison label and cell counts."""
    both_correct, a_correct_b_wrong, a_wrong_b_correct, both_wrong = cells
    return {
        "comparison": comparison,
        "shared_pairs": sum(cells),
        "both_correct": both_correct,
        "a_correct_b_wrong": a_correct_b_wrong,
        "a_wrong_b_correct": a_wrong_b_correct,
        "both_wrong": both_wrong,
        "p_value": f"{exact_mcnemar_p_value(a_correct_b_wrong, a_wrong_b_correct):.12g}",
    }


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

    report_rows: List[Dict[str, object]] = []

    for pipeline_a, pipeline_b in COMPARISONS:
        comparison = f"{pipeline_a} vs {pipeline_b}"
        overall_cells: Cells = [0, 0, 0, 0]

        for dataset in args.datasets:
            for model in MODEL_CHOICES:
                cells = compare_stratum(pipeline_a, pipeline_b, model, dataset, args.languages)
                if cells is None:
                    continue

                _add(overall_cells, cells)

        if sum(overall_cells) > 0:
            report_rows.append(_report_row(comparison, overall_cells))

    if not report_rows:
        print("No matching result files found for any (pipeline, model, dataset) combination.")
        sys.exit(1)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(report_rows[0]))
        writer.writeheader()
        writer.writerows(report_rows)

    print(f"Wrote {OUTPUT_PATH}")
    print()
    print("Overall pooled results (all models, all selected datasets):")
    for row in report_rows:
        print(
            f"  {row['comparison']:22s} n={row['shared_pairs']:5d} "
            f"discordant=({row['a_correct_b_wrong']},{row['a_wrong_b_correct']}) "
            f"p={row['p_value']}"
        )


if __name__ == "__main__":
    main()
