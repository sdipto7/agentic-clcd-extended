#!/usr/bin/env python3
"""Run CLCDSA on-the-fly cosine scoring on this project's Java-Python pairs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASELINE_DIR = Path(__file__).resolve().parent
OFFICIAL_DIR = BASELINE_DIR / "official" / "CLCDSA_Code"
SRC_DIR = OFFICIAL_DIR / "src"
ANTLR_JAR = OFFICIAL_DIR / "libs" / "antlr-4.9.1-complete.jar"
BUILD_DIR = BASELINE_DIR / "build" / "classes"
DATASET_PATHS = {
    "xlcost": PROJECT_ROOT / "data" / "xlcost.json",
    "codenet": PROJECT_ROOT / "data" / "codenet.json",
    "gptclonebench": PROJECT_ROOT / "data" / "gptclonebench.json",
}


def _only_java_python(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Restrict to Java-Python pairs; data/*.json now also carries Java-C++/C# rows."""
    return [row for row in rows if str(row.get("ll2", "")).strip() == "Python"]


def load_rows(dataset: str) -> list[dict[str, Any]]:
    with DATASET_PATHS[dataset].open(encoding="utf-8") as f:
        rows = json.load(f)
    if not isinstance(rows, list):
        raise ValueError(f"expected JSON list in {DATASET_PATHS[dataset]}")
    return _only_java_python(rows)


def write_snippets(dataset: str, rows: list[dict[str, Any]]) -> Path:
    prepared = BASELINE_DIR / "prepared" / dataset
    if prepared.exists():
        shutil.rmtree(prepared)
    java_dir = prepared / "java"
    python_dir = prepared / "python"
    java_dir.mkdir(parents=True, exist_ok=True)
    python_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = prepared / "pairs.csv"
    with metadata_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["pair_id", "ground_truth"])
        writer.writeheader()
        for row in rows:
            pair_id = str(row["pair_id"])
            (java_dir / f"{pair_id}.java").write_text(str(row["codeA"]), encoding="utf-8")
            (python_dir / f"{pair_id}.py").write_text(str(row["codeB"]), encoding="utf-8")
            writer.writerow({"pair_id": pair_id, "ground_truth": int(row["label"])})
    return prepared


def compile_extractor() -> None:
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    cmd = [
        "javac",
        "-cp",
        f"{ANTLR_JAR}:{SRC_DIR}",
        "-sourcepath",
        f"{SRC_DIR}:{BASELINE_DIR}",
        "-d",
        str(BUILD_DIR),
        str(BASELINE_DIR / "PairFeatureExtractor.java"),
    ]
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)


def run_extractor(language: str, input_dir: Path, output_csv: Path) -> None:
    cmd = [
        "java",
        "-cp",
        f"{BUILD_DIR}:{ANTLR_JAR}",
        "com.company.PairFeatureExtractor",
        language,
        str(input_dir),
        str(output_csv),
    ]
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)


def read_features(path: Path) -> dict[str, list[float]]:
    features: dict[str, list[float]] = {}
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            features[str(row["pair_id"])] = [float(row[f"f{i}_{name}"]) for i, name in enumerate([
                "variables",
                "arguments",
                "operators",
                "expressions",
                "loops",
                "operands",
                "exceptions",
                "exception_clauses",
                "mccabe",
            ], start=1)]
    return features


def cosine(left: list[float], right: list[float]) -> float:
    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return numerator / (left_norm * right_norm)


def best_threshold(scored_rows: list[dict[str, Any]]) -> float:
    best = (0.5, -1.0)
    y_true = [int(row["ground_truth"]) for row in scored_rows]
    for threshold in np.linspace(0.0, 1.0, 201):
        y_pred = [1 if row["score"] >= threshold else 0 for row in scored_rows]
        score = f1_score(y_true, y_pred, zero_division=0)
        if score > best[1]:
            best = (float(threshold), float(score))
    return best[0]


def write_outputs(dataset: str, rows: list[dict[str, Any]], threshold: float, suffix: str) -> None:
    output_dir = PROJECT_ROOT / "output" / "clcdsa"
    run_dir = BASELINE_DIR / "runs" / dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)

    result_path = output_dir / f"results_clcdsa_onthefly_{dataset}_{suffix}.csv"
    score_path = run_dir / f"scores_onthefly_{suffix}.csv"

    with result_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["pair_id", "ll1", "ll2", "ground_truth", "predicted_label"])
        writer.writeheader()
        for row in rows:
            pred = row["score"] >= threshold
            writer.writerow({
                "pair_id": row["pair_id"],
                "ll1": row.get("ll1", ""),
                "ll2": row.get("ll2", ""),
                "ground_truth": row["ground_truth"],
                "predicted_label": "CLONE" if pred else "NOT_CLONE",
            })

    with score_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["pair_id", "ll1", "ll2", "ground_truth", "score", "threshold", "predicted_label"],
        )
        writer.writeheader()
        for row in rows:
            pred = row["score"] >= threshold
            writer.writerow({
                "pair_id": row["pair_id"],
                "ll1": row.get("ll1", ""),
                "ll2": row.get("ll2", ""),
                "ground_truth": row["ground_truth"],
                "score": f"{row['score']:.8f}",
                "threshold": f"{threshold:.8f}",
                "predicted_label": "CLONE" if pred else "NOT_CLONE",
            })

    print(f"results={result_path}")
    print(f"scores={score_path}")


def print_metrics(rows: list[dict[str, Any]], threshold: float) -> None:
    y_true = [int(row["ground_truth"]) for row in rows]
    y_pred = [1 if row["score"] >= threshold else 0 for row in rows]
    print(f"threshold={threshold:.4f}")
    print(f"accuracy={accuracy_score(y_true, y_pred):.4f}")
    print(f"precision={precision_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"recall={recall_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"f1={f1_score(y_true, y_pred, zero_division=0):.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=sorted(DATASET_PATHS), required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--tune-threshold-on-full-set",
        action="store_true",
        help="Diagnostic only: chooses the best threshold on the same full set being evaluated.",
    )
    args = parser.parse_args()

    if not ANTLR_JAR.is_file():
        raise FileNotFoundError(f"missing CLCDSA ANTLR jar: {ANTLR_JAR}")

    rows = load_rows(args.dataset)
    prepared = write_snippets(args.dataset, rows)
    compile_extractor()
    run_extractor("java", prepared / "java", prepared / "java_features.csv")
    run_extractor("python", prepared / "python", prepared / "python_features.csv")

    java_features = read_features(prepared / "java_features.csv")
    python_features = read_features(prepared / "python_features.csv")

    scored_rows = []
    missing = []
    for row in rows:
        pair_id = str(row["pair_id"])
        if pair_id not in java_features or pair_id not in python_features:
            missing.append(pair_id)
            continue
        scored_rows.append({
            "pair_id": pair_id,
            "ll1": row.get("ll1", ""),
            "ll2": row.get("ll2", ""),
            "ground_truth": int(row["label"]),
            "score": cosine(java_features[pair_id], python_features[pair_id]),
        })

    if missing:
        print(f"warning: missing features for {len(missing)} pairs; scoring {len(scored_rows)} pairs", file=sys.stderr)

    suffix = "full"
    threshold = args.threshold
    if args.tune_threshold_on_full_set:
        threshold = best_threshold(scored_rows)
        suffix = "full_tuned"
    print(f"dataset={args.dataset} pairs={len(scored_rows)}")
    print_metrics(scored_rows, threshold)
    write_outputs(args.dataset, scored_rows, threshold, suffix)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
