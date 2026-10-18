#!/usr/bin/env python3
"""Evaluate the final baseline result files into one consolidated report.

This script intentionally evaluates only the baseline CSVs selected for the
study, not every `results_*.csv` under output/. It keeps LLM pipeline outputs out
of the baseline-only report and writes one stable report file.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluate import _evaluate_file  # noqa: E402


BASELINE_FILES = [
    ("clcdsa", "clcdsa_onthefly_t0.5", "xlcost", PROJECT_ROOT / "output" / "clcdsa" / "results_clcdsa_onthefly_xlcost_full.csv"),
    ("clcdsa", "clcdsa_onthefly_t0.5", "codenet", PROJECT_ROOT / "output" / "clcdsa" / "results_clcdsa_onthefly_codenet_full.csv"),
    ("clcdsa", "clcdsa_onthefly_t0.5", "gptclonebench", PROJECT_ROOT / "output" / "clcdsa" / "results_clcdsa_onthefly_gptclonebench_full.csv"),
    ("codebert", "codebert_rawcal_cosine", "xlcost", PROJECT_ROOT / "output" / "codebert" / "results_codebert_xlcost_full.csv"),
    ("codebert", "codebert_rawcal_cosine", "codenet", PROJECT_ROOT / "output" / "codebert" / "results_codebert_codenet_full.csv"),
    ("codebert", "codebert_rawcal_cosine", "gptclonebench", PROJECT_ROOT / "output" / "codebert" / "results_codebert_gptclonebench_full.csv"),
    ("nicad_algorithms", "nicad_algorithm_summaries", "xlcost", PROJECT_ROOT / "output" / "nicad_algorithms" / "results_nicad_algorithms_xlcost_full.csv"),
    ("nicad_algorithms", "nicad_algorithm_summaries", "codenet", PROJECT_ROOT / "output" / "nicad_algorithms" / "results_nicad_algorithms_codenet_full.csv"),
    ("nicad_algorithms", "nicad_algorithm_summaries", "gptclonebench", PROJECT_ROOT / "output" / "nicad_algorithms" / "results_nicad_algorithms_gptclonebench_full.csv"),
    ("nicad_source", "nicad_direct_source_java", "xlcost", PROJECT_ROOT / "output" / "nicad_source" / "results_nicad_source_xlcost_java_full.csv"),
    ("nicad_source", "nicad_direct_source_java", "codenet", PROJECT_ROOT / "output" / "nicad_source" / "results_nicad_source_codenet_java_full.csv"),
    ("nicad_source", "nicad_direct_source_java", "gptclonebench", PROJECT_ROOT / "output" / "nicad_source" / "results_nicad_source_gptclonebench_java_full.csv"),
]


FIELDNAMES = [
    "pipeline",
    "model",
    "dataset",
    "accuracy",
    "precision",
    "recall",
    "f1_score",
    "total",
    "clones",
    "non_clones",
    "true_positive",
    "true_negative",
    "false_positive",
    "false_negative",
    "file",
]


def _baseline_report_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    missing: list[Path] = []
    for pipeline, model, dataset, path in BASELINE_FILES:
        if not path.is_file():
            missing.append(path)
            continue
        row = _evaluate_file(str(path))
        row["pipeline"] = pipeline
        row["model"] = model
        row["dataset"] = dataset
        row["file"] = os.path.relpath(path, PROJECT_ROOT)
        rows.append(row)
    if missing:
        missing_text = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(f"Missing baseline result files:\n{missing_text}")
    return rows


def write_report(path: Path) -> None:
    rows = _baseline_report_rows()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row[name] for name in FIELDNAMES})
    print(f"Baseline report: {path}")
    print(f"Rows: {len(rows)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "reports" / "baseline_evaluation_report.csv",
        help="Path for the consolidated baseline report.",
    )
    args = parser.parse_args()
    write_report(args.output)


if __name__ == "__main__":
    main()
