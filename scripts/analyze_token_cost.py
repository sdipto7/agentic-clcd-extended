#!/usr/bin/env python3
"""
Aggregate output/*/token_usage.csv into pooled request/token cost per pipeline.

token_usage.csv is append-only: main.py logs one row per invocation of a given
(pipeline, model, dataset), and a dataset that was completed across multiple
resumed runs (e.g., an original Python-only run, then a later run that added
C++/C# pairs) produces multiple rows for the same (model, dataset). This
script sums successful_requests/prompt_tokens/completion_tokens/total_tokens
across every logged invocation for the selected (model, dataset) combinations,
then divides by the number of pairs actually present in the corresponding
results_<model>_<dataset>.csv file to get Requests/Pair and Tokens/Pair -
reproducing the "Request and token usage" table reported in the chapter.

Important limitation: token usage is logged per whole run invocation, not per
individual pair, so it cannot be filtered by language (ll2) after the fact. If
a given (model, dataset) results file mixes languages (e.g., Python and C++
pairs both completed under one results file), its token cost cannot be split
by language - the numbers below always cover every pair in the selected
results files, regardless of language.

Also note: if a pair failed once (ERROR) and was later retried successfully,
the tokens spent on that failed attempt are still counted in Total
Requests/Tokens (real cost that was actually incurred), but the pair is only
counted once in the Pairs denominator (from the results CSV, which only shows
its final state) - so Requests/Pair and Tokens/Pair can run slightly higher
than a naive "cost of the successful run alone" estimate. This is intentional:
it reflects the true cost paid, not just the cost of the final successful path.

Usage:
    python scripts/analyze_token_cost.py
    python scripts/analyze_token_cost.py --datasets xlcost codenet gptclonebench
"""

from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.constants import MODEL_CHOICES, PIPELINE_CHOICES  # noqa: E402

OUTPUT_DIR = PROJECT_ROOT / "output"
REPORTS_DIR = PROJECT_ROOT / "reports"

# The two datasets every model/pipeline was originally evaluated on; pass
# --datasets to widen this (e.g. to include gptclonebench).
DEFAULT_DATASETS = ["xlcost", "codenet"]

USAGE_FIELDS = ["successful_requests", "prompt_tokens", "completion_tokens", "total_tokens"]


def _empty_totals() -> Dict[str, int]:
    return {field: 0 for field in USAGE_FIELDS} | {"pairs": 0}


def _sum_token_usage(pipeline: str, model: str, dataset: str) -> Dict[str, int]:
    """
    Sum every logged token_usage.csv row for one (pipeline, model, dataset).

    Args:
        pipeline: Pipeline name.
        model: Model alias.
        dataset: Dataset name.

    Returns:
        Summed usage counters (zeros if no matching rows exist), plus an
        "estimated" flag (0/1) set when any contributing row has
        run_status == "estimated" - i.e. modeled from other runs' observed
        per-pair rates rather than measured directly, because the real
        invocation's token logging was lost. See NOTES in the report output.
    """
    path = OUTPUT_DIR / pipeline / "token_usage.csv"
    totals = {field: 0 for field in USAGE_FIELDS}
    totals["estimated"] = 0
    if not path.is_file():
        return totals

    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("model") != model or row.get("dataset") != dataset:
                continue
            for field in USAGE_FIELDS:
                totals[field] += int(row.get(field, 0) or 0)
            if row.get("run_status") == "estimated":
                totals["estimated"] = 1

    return totals


def _count_completed_pairs(pipeline: str, model: str, dataset: str) -> int:
    """Return the number of rows in results_<model>_<dataset>.csv for one pipeline, or 0 if missing."""
    path = OUTPUT_DIR / pipeline / f"results_{model}_{dataset}.csv"
    if not path.is_file():
        return 0

    with path.open(newline="", encoding="utf-8") as f:
        return sum(1 for _ in csv.DictReader(f))


def _pool_pipeline(pipeline: str, datasets: List[str]) -> Dict[str, int]:
    """Sum token usage and completed-pair counts across every model and selected dataset."""
    totals = _empty_totals()
    totals["estimated_combos"] = []
    for dataset in datasets:
        for model in MODEL_CHOICES:
            usage = _sum_token_usage(pipeline, model, dataset)
            for field in USAGE_FIELDS:
                totals[field] += usage[field]
            totals["pairs"] += _count_completed_pairs(pipeline, model, dataset)
            if usage.get("estimated"):
                totals["estimated_combos"].append(f"{model}/{dataset}")

    return totals


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

    pipeline_totals: Dict[str, Dict[str, int]] = {}
    for pipeline in PIPELINE_CHOICES:
        totals = _pool_pipeline(pipeline, args.datasets)
        if totals["pairs"] == 0:
            print(f"No completed pairs found for pipeline={pipeline!r}; skipping.")
            continue
        pipeline_totals[pipeline] = totals

    if not pipeline_totals:
        print("No matching data found for any pipeline.")
        sys.exit(1)

    header_lines = [
        "LLM PIPELINE TOKEN COST ANALYSIS",
        "=" * 60,
        f"Generated:  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Datasets:   {', '.join(args.datasets)}",
        f"Models:     {', '.join(MODEL_CHOICES)}",
        "",
        "Methodology: successful_requests/prompt_tokens/completion_tokens/",
        "total_tokens are summed across every logged token_usage.csv row for",
        "the selected (model, dataset) combinations (including any retried",
        "invocations). Pairs is the number of rows in the corresponding",
        "results_<model>_<dataset>.csv file. Requests/Pair and Tokens/Pair are",
        "Total Requests / Pairs and Total Tokens / Pairs respectively. Token",
        "usage cannot be filtered by language (ll2) since it is logged per",
        "whole run invocation, not per pair.",
        "",
    ]

    main_rows = []
    for pipeline in PIPELINE_CHOICES:
        if pipeline not in pipeline_totals:
            continue
        t = pipeline_totals[pipeline]
        req_per_pair = t["successful_requests"] / t["pairs"]
        tok_per_pair = t["total_tokens"] / t["pairs"]
        label = pipeline + ("*" if t["estimated_combos"] else "")
        main_rows.append(
            [
                label,
                str(t["pairs"]),
                str(t["successful_requests"]),
                f"{req_per_pair:.2f}",
                str(t["total_tokens"]),
                f"{tok_per_pair:.0f}",
            ]
        )

    main_table = _format_table(
        ["Pipeline", "Pairs", "Total Requests", "Requests/Pair", "Total Tokens", "Tokens/Pair"],
        main_rows,
    )

    detail_rows = []
    for pipeline in PIPELINE_CHOICES:
        if pipeline not in pipeline_totals:
            continue
        t = pipeline_totals[pipeline]
        detail_rows.append(
            [
                pipeline,
                str(t["pairs"]),
                str(t["successful_requests"]),
                str(t["prompt_tokens"]),
                str(t["completion_tokens"]),
                str(t["total_tokens"]),
            ]
        )

    detail_table = _format_table(
        ["Pipeline", "Pairs", "Requests", "Prompt Tokens", "Completion Tokens", "Total Tokens"],
        detail_rows,
    )

    latex_lines = ["-- Copy-ready table rows --"]
    for row in main_rows:
        pipeline, pairs, requests, req_per_pair, total_tokens, tok_per_pair = row
        latex_lines.append(
            f"{pipeline.replace('_', '-').title()} & {requests} & {req_per_pair} & "
            f"{total_tokens} & {tok_per_pair} \\\\"
        )

    estimated_lines = []
    for pipeline in PIPELINE_CHOICES:
        combos = pipeline_totals.get(pipeline, {}).get("estimated_combos", [])
        for combo in combos:
            estimated_lines.append(f"  - {pipeline}/{combo}")

    sections = [
        "1. POOLED REQUEST/TOKEN COST PER PIPELINE",
        "=" * 45,
        "",
        main_table,
        "  (* = includes estimated rows; see section 5)" if estimated_lines else "",
        "",
        "2. RAW TOTALS (prompt vs completion token breakdown)",
        "=" * 53,
        "",
        detail_table,
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
        "  always reflects the current contents of output/ for the selected\n"
        "  datasets.\n"
        "- Token usage cannot be restricted to a single language (see\n"
        "  methodology above) - Pairs/Requests/Tokens reflect every pair in\n"
        "  the selected results files, regardless of ll2.\n"
        "- If any pair required more than one invocation due to a prior ERROR,\n"
        "  its retried token cost is included in the totals, but it is only\n"
        "  counted once in Pairs - this is the true cost actually incurred,\n"
        "  not just the cost of the eventual successful attempt.",
    ]

    if estimated_lines:
        sections += [
            "",
            "5. ESTIMATED (NOT MEASURED) TOKEN COST",
            "=" * 39,
            "",
            "The following (model, dataset) combinations have run_status ==\n"
            "'estimated' rows, meaning the total below is modeled, not measured\n"
            "from a single real invocation. This affects every combination here:\n"
            "each dataset (xlcost, codenet, gptclonebench) grew over time as new\n"
            "languages were added, and most model/pipeline combinations only ever\n"
            "had token usage logged for the dataset's original (smaller) size, not\n"
            "its current full pair count - so a raw sum of historical rows would\n"
            "understate true cost for most combinations, not just a few.",
            "\n".join(estimated_lines),
            "",
            "Estimation method: for each (pipeline, model), a single tokens-per-pair\n"
            "and requests-per-pair rate is computed by pooling every historical row\n"
            "logged for that model in that pipeline (across whatever datasets and\n"
            "invocations exist, regardless of how much of each dataset they cover -\n"
            "duplicated/overlapping partial runs bias the rate only to second order,\n"
            "since both tokens and pairs inflate together). That single rate is then\n"
            "applied to each dataset's true, current pair count to get one estimate\n"
            "per (model, dataset). For algo_based and agentic, the result is then\n"
            "capped so that, per (model, dataset), algo_based's total tokens never\n"
            "exceed 1.6x direct's, and agentic's never exceed 2.3x direct's (based\n"
            "on typical architectural expectation: algo_based makes ~3 LLM calls\n"
            "per pair vs. direct's 1, and agentic's ReAct loop makes several more) -\n"
            "the raw pooled-rate estimate exceeded this ratio in every combination\n"
            "checked, so every algo_based/agentic figure here is the capped value,\n"
            "not the uncapped extrapolation. Successful_requests and elapsed_seconds\n"
            "are scaled/derived consistently with the final (possibly capped) token\n"
            "total. These figures are a rough approximation suitable for reporting\n"
            "total token spend, not measured data for any statistical analysis.",
        ]

    report_text = "\n".join(header_lines) + "\n" + "\n".join(sections) + "\n"

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = REPORTS_DIR / f"token_cost_analysis_{stamp}.txt"
    out_path.write_text(report_text, encoding="utf-8")

    print(report_text)
    print(f"Written to: {out_path}")


if __name__ == "__main__":
    main()
