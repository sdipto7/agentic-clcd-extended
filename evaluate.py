#!/usr/bin/env python3
"""
Offline evaluation of pipeline CSV outputs under output/ (or a custom directory).
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from datetime import datetime
from typing import Dict, Iterable, List, Tuple

_PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.core.constants import CLONE, NOT_CLONE, PIPELINE_CHOICES, PROJECT_ROOT  # noqa: E402


def _iter_csv_files(results_dir: str) -> List[str]:
    """
    Recursively find output CSV files under a directory.

    This scanner is intentionally strict: it only return CSV files whose filename
    matches `results_*.csv` (case-insensitive). This avoids accidentally
    including non-result artifacts such as `token_usage.csv`.

    Args:
        results_dir: Root directory to walk recursively (e.g., `output/`).

    Returns:
        A sorted list of absolute file paths to matching `results_*.csv` files.
    """
    csv_files: List[str] = []

    for dirpath, _dirnames, filenames in os.walk(results_dir):
        for name in filenames:
            if name.lower().startswith("results_") and name.lower().endswith(".csv"):
                csv_files.append(os.path.join(dirpath, name))

    return sorted(csv_files)


def _filter_llm_pipeline_files(paths: List[str]) -> List[str]:
    """
    Keep only result files that belong to an LLM pipeline (direct/algo_based/agentic).

    This excludes baseline result files (nicad_source, nicad_algorithms, codebert,
    clcdsa, etc.), which are evaluated separately via
    ``evaluate_baselines.py`` so the two experiment families never share one report.

    Args:
        paths: Candidate result CSV paths (e.g., from ``_iter_csv_files``).

    Returns:
        The subset of paths whose inferred pipeline is in ``PIPELINE_CHOICES``.
    """
    return [p for p in paths if _extract_pipeline_model_dataset_from_path(p)[0] in PIPELINE_CHOICES]


def _extract_pipeline_model_dataset_from_path(path: str) -> Tuple[str, str, str]:
    """
    Extract (pipeline, model, dataset) from a results CSV file path.

    Parsing rules:
    - `pipeline` is the name of the parent directory containing the CSV.
      Example: `.../output/agentic/results_x_y.csv` -> pipeline = `agentic`.
    - `model` and `dataset` are extracted from the filename if it follows the pattern
      `results_<model_alias>_<dataset>.csv`.

    If the filename does not match the expected pattern (missing prefix/suffix or
    missing underscore in the stem), `model` and `dataset` default to `"unknown"`.

    Args:
        path: Path to a CSV file.

    Returns:
        A tuple (pipeline, model, dataset) where each element is a string.
    """
    basename = os.path.basename(path)
    parent = os.path.basename(os.path.dirname(path))
    pipeline = parent
    model = "unknown"
    dataset = "unknown"

    if basename.startswith("results_") and basename.endswith(".csv"):
        model_dataset_part = basename[len("results_") : -len(".csv")]
        if "_" in model_dataset_part:
            model, dataset = model_dataset_part.rsplit("_", 1)

    return pipeline, model, dataset


def _compute_classification_metrics(rows: Iterable[dict[str, str]]) -> Dict[str, Any]:
    """
    Compute classification counts and standard metrics from CSV rows.

    Each row is expected to contain:
    - ground_truth: integer label (1 for clone, 0 for not-clone). Missing/invalid
      values are treated as -1 (unknown) and won't match TP/TN/FP/FN conditions.
    - predicted_label: a string label. It is normalized by uppercasing and mapping
      common variants:
        - CLONE constant -> positive class
        - NOT_CLONE constant -> negative class

    Confusion counting behavior:
    - true_positive, true_negative, false_positive, and false_negative are
      accumulated for recognized predictions (clone / not-clone).

    Metrics:
    - accuracy = (true_positive + true_negative) / total
    - precision = true_positive / (true_positive + false_positive)
    - recall = true_positive / (true_positive + false_negative)
    - f1_score = 2 * precision * recall / (precision + recall)

    Args:
        rows: Iterable of dict-like CSV rows.

    Returns:
        A dict containing confusion counts and derived metrics:
        total, clones, non_clones, true_positive, true_negative,
        false_positive, false_negative, accuracy, precision, recall,
        f1_score.
    """
    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0
    total = 0
    clones = 0
    non_clones = 0
    
    for row in rows:
        total += 1

        try:
            ground_truth = int(row.get("ground_truth", -1))
        except ValueError:
            ground_truth = -1

        pred_raw = str(row.get("predicted_label", "")).strip().upper()
        if pred_raw in ("NOT CLONE", "NON_CLONE"):
            pred_raw = NOT_CLONE

        if pred_raw == CLONE:
            predicted_label = 1
        elif pred_raw == NOT_CLONE:
            predicted_label = 0

        else:
            predicted_label = -1  # ERROR or unknown

        if ground_truth == 1:
            clones += 1
        elif ground_truth == 0:
            non_clones += 1

        if predicted_label == -1:
            if ground_truth == 1:
                false_negative += 1
            elif ground_truth == 0:
                false_positive += 1
            continue

        if ground_truth == 1 and predicted_label == 1:
            true_positive += 1
        elif ground_truth == 1 and predicted_label == 0:
            false_negative += 1
        elif ground_truth == 0 and predicted_label == 1:
            false_positive += 1
        elif ground_truth == 0 and predicted_label == 0:
            true_negative += 1

    correct = true_positive + true_negative
    accuracy = correct / total if total else 0.0
    precision_denom = true_positive + false_positive
    recall_denom = true_positive + false_negative
    precision = true_positive / precision_denom if precision_denom else 0.0
    recall = true_positive / recall_denom if recall_denom else 0.0
    f1_denom = precision + recall
    f1_score = (2 * precision * recall / f1_denom) if f1_denom else 0.0

    return {
        "total": total,
        "clones": clones,
        "non_clones": non_clones,
        "true_positive": true_positive,
        "true_negative": true_negative,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "accuracy": f"{accuracy:.2f}",
        "precision": f"{precision:.2f}",
        "recall": f"{recall:.2f}",
        "f1_score": f"{f1_score:.2f}",
    }


def _evaluate_file(path: str) -> Dict[str, Any]:
    """
    Evaluate one results CSV file and return a single aggregated report row.

    This function:
    - reads the CSV at path using csv.DictReader
    - infers pipeline, model, and dataset from the file path
    - computes classification counts and metrics across all rows
    - returns a dict that merges the inferred metadata with the computed metrics

    Expected CSV columns:
    - ground_truth: 1 for clone, 0 for not-clone
    - predicted_label: label string that will be normalized by the metrics function

    Args:
        path: Path to a single results CSV file.

    Returns:
        A dictionary with keys:
        file, pipeline, model, dataset,
        total, clones, non_clones,
        true_positive, true_negative, false_positive, false_negative,
        accuracy, precision, recall, f1_score.
    """
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    pipeline, model, dataset = _extract_pipeline_model_dataset_from_path(path)

    metrics = _compute_classification_metrics(rows)

    return {
        "file": path,
        "pipeline": pipeline,
        "model": model,
        "dataset": dataset,
        **metrics,
    }


def _get_report_path(prefix: str, extension: str) -> str:
    """
    Build a timestamped report file path under the reports directory.

    Args:
        prefix: Filename prefix to use before the timestamp.
        extension: File extension without a leading dot (e.g. "csv", "txt").

    Returns:
        Full path to a file under PROJECT_ROOT/reports named like
        prefix_YYYYMMDD_HHMMSS.<extension>.
    """
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    reports_dir = os.path.join(PROJECT_ROOT, "reports")

    return os.path.join(reports_dir, f"{prefix}_{stamp}.{extension}")


def _write_report_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    """
    Write an aggregated evaluation report CSV to disk.

    The output CSV contains one row per evaluated input file (or per evaluated
    result set), with metadata fields (pipeline, model, dataset) and the computed
    classification metrics and confusion counts.

    The destination directory is created if it does not already exist. The file is
    written with a header row followed by one row per element in rows.

    Args:
        path: Output CSV file path to write (typically under the reports directory).
        rows: List of dictionaries containing the report fields to write. Each dict
            must provide values for the predefined report columns written by this
            function.

    Returns:
        None.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = [
        "pipeline",
        "model",
        "dataset",
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "total",
        "true_positive",
        "true_negative",
        "false_positive",
        "false_negative",
    ]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r[k] for k in fieldnames})


def _load_rows_with_metadata(paths: List[str]) -> List[Dict[str, Any]]:
    """
    Read every row from every target file, tagged with its pipeline/model/dataset/ll2.

    Each returned row keeps the original CSV fields (pair_id, ll1, ll2, ground_truth,
    predicted_label) and adds ``_pipeline``, ``_model``, ``_dataset`` (inferred from
    the file path) and ``_ll2`` (the row's own ``ll2`` value, or ``"UNKNOWN"`` for
    older files that predate the ll1/ll2 columns), so all rows across every file can
    be pooled and re-grouped along any of these dimensions in one pass.

    Args:
        paths: Result CSV file paths to load.

    Returns:
        A flat list of enriched row dicts, one per data row across all files.
    """
    all_rows: List[Dict[str, Any]] = []
    for path in paths:
        pipeline, model, dataset = _extract_pipeline_model_dataset_from_path(path)
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                enriched = dict(row)
                enriched["_pipeline"] = pipeline
                enriched["_model"] = model
                enriched["_dataset"] = dataset
                enriched["_ll2"] = str(row.get("ll2", "")).strip() or "UNKNOWN"
                all_rows.append(enriched)

    return all_rows


def _pool_by(
    rows: List[Dict[str, Any]], key_fn
) -> Dict[Any, Dict[str, Any]]:
    """
    Group rows by a key function and compute pooled classification metrics per group.

    Pooling means every group's metrics are derived from that group's *combined*
    true_positive/true_negative/false_positive/false_negative counts, not from
    averaging per-file accuracy/precision/recall/F1 values. This is the statistically
    correct way to combine groups of unequal size (e.g., xlcost has 1200 pairs per
    model, codenet has 600) and matches the pooling convention already used
    elsewhere in this project's analysis.

    Args:
        rows: Enriched rows (see ``_load_rows_with_metadata``).
        key_fn: Function mapping a row to its group key (e.g., a pipeline name,
            or a ``(pipeline, ll2)`` tuple for a cross-tab).

    Returns:
        Mapping of group key to that group's metrics dict (see
        ``_compute_classification_metrics``).
    """
    groups: Dict[Any, List[Dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(key_fn(row), []).append(row)

    return {key: _compute_classification_metrics(group_rows) for key, group_rows in groups.items()}


def _format_table(headers: List[str], data_rows: List[List[str]]) -> str:
    """
    Render a left-aligned, fixed-width monospace table as a single string.

    Column widths are derived from the longest value (including the header) in
    each column, with a 2-space gutter between columns.

    Args:
        headers: Column header labels.
        data_rows: Table body rows; each inner list must have the same length
            as ``headers``.

    Returns:
        The rendered table (header row, a divider line, then data rows),
        newline-joined, with no trailing newline.
    """
    widths = [len(h) for h in headers]
    for row in data_rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))

    def _fmt_row(cells: List[str]) -> str:
        return "  ".join(str(c).ljust(widths[i]) for i, c in enumerate(cells))

    lines = [_fmt_row(headers), "  ".join("-" * w for w in widths)]
    lines.extend(_fmt_row(row) for row in data_rows)

    return "\n".join(lines)


def _metrics_row(label_cols: List[str], metrics: Dict[str, Any]) -> List[str]:
    """Build one table row: label column(s) followed by the standard metric columns."""
    return [
        *label_cols,
        str(metrics["total"]),
        metrics["accuracy"],
        metrics["precision"],
        metrics["recall"],
        metrics["f1_score"],
        str(metrics["true_positive"]),
        str(metrics["true_negative"]),
        str(metrics["false_positive"]),
        str(metrics["false_negative"]),
    ]


_METRIC_HEADERS = ["Total", "Accuracy", "Precision", "Recall", "F1", "TP", "TN", "FP", "FN"]


def _render_single_key_section(
    title: str,
    label_header: str,
    grouped: Dict[str, Dict[str, Any]],
) -> str:
    """
    Render one report section for a single-dimension grouping (e.g., by pipeline).

    Rows are sorted by F1 descending, and a one-line best/worst summary is appended
    (skipped when there is only one group, since a ranking is meaningless there).

    Args:
        title: Section heading text.
        label_header: Header label for the group-key column.
        grouped: Mapping of group key to its pooled metrics (see ``_pool_by``).

    Returns:
        The rendered section as a string.
    """
    ordered = sorted(grouped.items(), key=lambda kv: float(kv[1]["f1_score"]), reverse=True)
    table = _format_table(
        [label_header, *_METRIC_HEADERS],
        [_metrics_row([key], metrics) for key, metrics in ordered],
    )

    lines = [title, "=" * len(title), "", table]
    if len(ordered) > 1:
        best_key, best_metrics = ordered[0]
        worst_key, worst_metrics = ordered[-1]
        lines.append("")
        lines.append(
            f"Highest F1: {best_key} (F1={best_metrics['f1_score']}, "
            f"accuracy={best_metrics['accuracy']}) | "
            f"Lowest F1: {worst_key} (F1={worst_metrics['f1_score']}, "
            f"accuracy={worst_metrics['accuracy']})"
        )

    return "\n".join(lines)


def _render_cross_tab_section(
    title: str,
    row_label_header: str,
    col_label_header: str,
    grouped: Dict[Tuple[str, str], Dict[str, Any]],
) -> str:
    """
    Render one report section for a two-dimension grouping (e.g., pipeline x language).

    Args:
        title: Section heading text.
        row_label_header: Header label for the first key element's column.
        col_label_header: Header label for the second key element's column.
        grouped: Mapping of (row_key, col_key) tuples to pooled metrics.

    Returns:
        The rendered section as a string, with sub-tables grouped by row_key
        (each sorted by F1 descending within that row_key), so e.g. every
        pipeline's per-language breakdown is shown together.
    """
    row_keys = sorted({k[0] for k in grouped})
    lines = [title, "=" * len(title), ""]

    for row_key in row_keys:
        entries = sorted(
            ((k[1], v) for k, v in grouped.items() if k[0] == row_key),
            key=lambda kv: float(kv[1]["f1_score"]),
            reverse=True,
        )
        table = _format_table(
            [col_label_header, *_METRIC_HEADERS],
            [_metrics_row([col_key], metrics) for col_key, metrics in entries],
        )
        lines.append(f"{row_label_header}: {row_key}")
        lines.append(table)
        lines.append("")

    return "\n".join(lines).rstrip()


def _write_analysis_txt(path: str, targets: List[str]) -> None:
    """
    Write the detailed, human-readable analysis report for the LLM pipeline runs.

    Unlike the CSV report (one row per file, exactly as evaluated), this report
    pools raw rows across files along several dimensions - pipeline, model,
    dataset, and language (ll2) - both individually and cross-tabulated, so
    questions like "was Python detection stronger than C++ overall" or "how does
    claude-3-haiku compare across languages" can be read directly without manual
    recomputation.

    Args:
        path: Output .txt file path to write (typically under the reports directory).
        targets: Result CSV file paths that were included in this evaluation run.

    Returns:
        None.
    """
    rows = _load_rows_with_metadata(targets)

    by_pipeline = _pool_by(rows, lambda r: r["_pipeline"])
    by_model = _pool_by(rows, lambda r: r["_model"])
    by_dataset = _pool_by(rows, lambda r: r["_dataset"])
    by_language = _pool_by(rows, lambda r: r["_ll2"])
    by_pipeline_language = _pool_by(rows, lambda r: (r["_pipeline"], r["_ll2"]))
    by_model_language = _pool_by(rows, lambda r: (r["_model"], r["_ll2"]))
    by_full = _pool_by(
        rows, lambda r: (r["_pipeline"], r["_model"], r["_dataset"], r["_ll2"])
    )

    overall = _compute_classification_metrics(rows)
    languages_present = sorted(by_language)

    header = [
        "LLM PIPELINE - DETAILED ANALYSIS REPORT",
        "=" * 60,
        f"Generated:       {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Files analyzed:  {len(targets)}",
        f"Total pairs:     {overall['total']}",
        f"Pipelines:       {', '.join(sorted(by_pipeline))}",
        f"Models:          {', '.join(sorted(by_model))}",
        f"Datasets:        {', '.join(sorted(by_dataset))}",
        f"Languages (ll2): {', '.join(languages_present)}",
        "",
        "Methodology: every metric below is computed by pooling raw",
        "true_positive/true_negative/false_positive/false_negative counts across",
        "all rows in the group, then deriving accuracy/precision/recall/F1 from",
        "those pooled counts - not by averaging per-file metrics. A prediction of",
        "ERROR (the model failed to produce a usable verdict) counts as incorrect:",
        "a false negative if the pair is a true clone, a false positive otherwise.",
        "Some datasets do not include every language (e.g., clcdsa is Python-only;",
        "gptclonebench has Python and C# but no C++), so language totals reflect",
        "only the datasets that actually contain that language. If any experiment",
        "is still in progress, totals reflect whatever has completed so far.",
        "",
    ]

    sections = [
        _render_single_key_section(
            "1. PIPELINE-LEVEL SUMMARY (pooled across all models, datasets, languages)",
            "Pipeline",
            by_pipeline,
        ),
        _render_single_key_section(
            "2. MODEL-LEVEL SUMMARY (pooled across all pipelines, datasets, languages)",
            "Model",
            by_model,
        ),
        _render_single_key_section(
            "3. DATASET-LEVEL SUMMARY (pooled across all pipelines, models, languages)",
            "Dataset",
            by_dataset,
        ),
        _render_single_key_section(
            "4. LANGUAGE-LEVEL SUMMARY (pooled across all pipelines, models, datasets)",
            "Language",
            by_language,
        ),
        _render_cross_tab_section(
            "5. PIPELINE x LANGUAGE BREAKDOWN (pooled across models and datasets)",
            "Pipeline",
            "Language",
            by_pipeline_language,
        ),
        _render_cross_tab_section(
            "6. MODEL x LANGUAGE BREAKDOWN (pooled across pipelines and datasets)",
            "Model",
            "Language",
            by_model_language,
        ),
    ]

    full_detail_rows = sorted(
        by_full.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2], kv[0][3])
    )
    full_table = _format_table(
        ["Pipeline", "Model", "Dataset", "Language", *_METRIC_HEADERS],
        [_metrics_row(list(key), metrics) for key, metrics in full_detail_rows],
    )
    sections.append(
        "7. FULL DETAIL (pipeline x model x dataset x language, no pooling beyond "
        "this row)\n" + "=" * 76 + "\n\n" + full_table
    )

    sections.append(
        "8. NOTES\n"
        + "=" * 8
        + "\n\n"
        "- A group with 0 total pairs cannot occur here: every group above is\n"
        "  derived only from rows that actually exist in the scanned result files.\n"
        "- 'UNKNOWN' under Language would mean a result file predates the ll1/ll2\n"
        "  columns; none of the currently scanned files are expected to hit this.\n"
        "- This report is regenerated from scratch on every run of evaluate.py and\n"
        "  always reflects the current contents of the output/ directory at that\n"
        "  moment - rerun it any time after new experiment results land."
    )

    full_text = "\n".join(header) + "\n" + ("\n\n".join(sections)) + "\n"

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(full_text)


def main() -> None:
    """CLI for scanning CSV outputs and writing reports."""
    parser = argparse.ArgumentParser(description="Evaluate experiment CSV outputs.")
    parser.add_argument(
        "--results_dir",
        required=False,
        default=os.path.join(PROJECT_ROOT, "output"),
        help="Directory to scan recursively for result CSV files.",
    )
    parser.add_argument(
        "--file",
        required=False,
        default=None,
        help="Evaluate a single CSV instead of scanning a directory.",
    )
    args = parser.parse_args()

    if args.file:
        targets = [args.file]
    else:
        all_files = _iter_csv_files(args.results_dir)
        targets = _filter_llm_pipeline_files(all_files)
        skipped = len(all_files) - len(targets)
        if skipped:
            print(
                f"Skipping {skipped} baseline result file(s); "
                "use evaluate_baselines.py for the baseline report."
            )

    if not targets:
        print(f"No CSV files found under {args.results_dir}")
        sys.exit(1)

    reports: List[Dict[str, Any]] = [_evaluate_file(path) for path in targets]

    csv_path = _get_report_path("llm_evaluation_report", "csv")
    _write_report_csv(csv_path, reports)
    print(f"LLM pipeline CSV report: {csv_path}")

    txt_path = _get_report_path("llm_detailed_analysis", "txt")
    _write_analysis_txt(txt_path, targets)
    print(f"LLM pipeline detailed analysis: {txt_path}")


if __name__ == "__main__":
    main()
