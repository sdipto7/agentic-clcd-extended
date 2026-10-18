#!/usr/bin/env python3
"""Run a CodeBERT cosine baseline with raw-data calibration and full data testing."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASELINE_DIR = Path(__file__).resolve().parent
EVAL_DATASET_PATHS = {
    "xlcost": PROJECT_ROOT / "data" / "xlcost.json",
    "codenet": PROJECT_ROOT / "data" / "codenet.json",
    "gptclonebench": PROJECT_ROOT / "data" / "gptclonebench.json",
}
RAW_DATASET_PATHS = {
    "xlcost": PROJECT_ROOT / "raw_data" / "java_xl.json",
    "codenet": PROJECT_ROOT / "raw_data" / "java_cn.json",
    # GPTCloneBench has no separate flat raw file with extra Java-X language
    # pairs like xlcost/codenet do; its own Java-C# rows (disjoint from the
    # Java-Python eval rows) serve the same calibration role.
    "gptclonebench": PROJECT_ROOT / "data" / "gptclonebench.json",
}


def _load_json_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing dataset file: {path}")
    with path.open(encoding="utf-8") as f:
        rows = json.load(f)
    if not isinstance(rows, list):
        raise ValueError(f"Expected a JSON array in {path}")
    return rows


def _row_label(row: dict[str, Any]) -> int:
    if "label" in row:
        return int(row["label"])
    row_type = str(row.get("type", "")).strip().lower()
    if row_type == "clone":
        return 1
    if row_type in {"nonclone", "non-clone", "not_clone", "not clone"}:
        return 0
    raise ValueError(f"Cannot infer label from row type={row.get('type')!r}")


def _pair_key(row: dict[str, Any]) -> tuple[str, str, str, str, str]:
    return (
        str(row.get("codeA", "")),
        str(row.get("codeB", "")),
        str(row.get("problem_id_1", "")),
        str(row.get("problem_id_2", "")),
        str(row.get("type", "")),
    )


def _with_pair_ids(rows: list[dict[str, Any]], prefix: str) -> list[dict[str, Any]]:
    normalized = []
    for index, row in enumerate(rows):
        item = dict(row)
        item["pair_id"] = str(item.get("pair_id") or f"{prefix}_{index}")
        item["label"] = _row_label(item)
        normalized.append(item)
    return normalized


def _only_java_python(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Restrict to Java-Python pairs; data/*.json now also carries Java-C++/C# rows."""
    return [row for row in rows if str(row.get("ll2", "")).strip() == "Python"]


def _load_eval_rows(dataset: str) -> list[dict[str, Any]]:
    return _with_pair_ids(_only_java_python(_load_json_rows(EVAL_DATASET_PATHS[dataset])), dataset)


def _load_calibration_rows(dataset: str, eval_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    raw_rows = _load_json_rows(RAW_DATASET_PATHS[dataset])
    eval_keys = {_pair_key(row) for row in eval_rows}
    calibration_rows = []
    for row in raw_rows:
        if _pair_key(row) in eval_keys:
            continue
        if str(row.get("ll1", "")).lower() != "java":
            continue
        # The Java-Python rows in raw_data are exactly the final evaluation pairs.
        # Keep calibration disjoint by using the other raw Java-X language pairs only.
        if str(row.get("ll2", "")).lower() == "python":
            continue
        calibration_rows.append(row)
    return _with_pair_ids(calibration_rows, f"raw_{dataset}")


def _stratified_train_valid_split(
    rows: list[dict[str, Any]],
    seed: int,
    train_ratio: float,
) -> dict[str, list[dict[str, Any]]]:
    by_label = {0: [], 1: []}
    for row in rows:
        by_label[int(row["label"])].append(row)

    rng = random.Random(seed)
    splits = {"train": [], "valid": []}
    for label_rows in by_label.values():
        label_rows = list(label_rows)
        rng.shuffle(label_rows)
        train_end = int(len(label_rows) * train_ratio)
        splits["train"].extend(label_rows[:train_end])
        splits["valid"].extend(label_rows[train_end:])

    for split_rows in splits.values():
        split_rows.sort(key=lambda row: str(row["pair_id"]))
    return splits


def _encode_texts(
    model: AutoModel,
    tokenizer: Any,
    texts: list[str],
    device: torch.device,
    batch_size: int,
    max_length: int,
) -> torch.Tensor:
    embeddings = []
    model.eval()
    with torch.no_grad():
        for start in tqdm(range(0, len(texts), batch_size), desc="codebert encode"):
            batch_texts = [" ".join(text.replace("\n", " ").strip().split()) for text in texts[start : start + batch_size]]
            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            token_embeddings = outputs.last_hidden_state
            mask = attention_mask.unsqueeze(-1)
            sentence_embeddings = (token_embeddings * mask).sum(1) / mask.sum(1).clamp(min=1)
            embeddings.append(F.normalize(sentence_embeddings, p=2, dim=1).cpu())
    return torch.cat(embeddings, dim=0)


def _score_rows(
    rows: list[dict[str, Any]],
    model: AutoModel,
    tokenizer: Any,
    device: torch.device,
    batch_size: int,
    max_length: int,
) -> list[dict[str, Any]]:
    left_embeddings = _encode_texts(model, tokenizer, [str(row["codeA"]) for row in rows], device, batch_size, max_length)
    right_embeddings = _encode_texts(model, tokenizer, [str(row["codeB"]) for row in rows], device, batch_size, max_length)
    scores = (left_embeddings * right_embeddings).sum(dim=1).tolist()
    return [
        {
            "pair_id": str(row["pair_id"]),
            "ll1": row.get("ll1", ""),
            "ll2": row.get("ll2", ""),
            "label": int(row["label"]),
            "score": float(score),
        }
        for row, score in zip(rows, scores)
    ]


def _best_threshold(rows: list[dict[str, Any]]) -> float:
    best_threshold = 0.0
    best_f1 = -1.0
    y_true = [row["label"] for row in rows]
    for threshold in np.linspace(-1.0, 1.0, 401):
        y_pred = [1 if row["score"] >= threshold else 0 for row in rows]
        score = f1_score(y_true, y_pred, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_threshold = float(threshold)
    return best_threshold


def _print_metrics(name: str, rows: list[dict[str, Any]], threshold: float) -> None:
    y_true = [row["label"] for row in rows]
    y_pred = [1 if row["score"] >= threshold else 0 for row in rows]
    print(name)
    print(f"threshold={threshold:.4f}")
    print(f"accuracy={accuracy_score(y_true, y_pred):.4f}")
    print(f"precision={precision_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"recall={recall_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"f1={f1_score(y_true, y_pred, zero_division=0):.4f}")


def _write_pair_ids(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(f"{row['pair_id']}\n")


def _write_outputs(dataset: str, rows: list[dict[str, Any]], threshold: float) -> None:
    output_path = PROJECT_ROOT / "output" / "codebert" / f"results_codebert_{dataset}_full.csv"
    scores_path = BASELINE_DIR / "runs" / dataset / "scores_full.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scores_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["pair_id", "ll1", "ll2", "ground_truth", "predicted_label"])
        writer.writeheader()
        for row in rows:
            pred = row["score"] >= threshold
            writer.writerow(
                {
                    "pair_id": row["pair_id"],
                    "ll1": row.get("ll1", ""),
                    "ll2": row.get("ll2", ""),
                    "ground_truth": row["label"],
                    "predicted_label": "CLONE" if pred else "NOT_CLONE",
                }
            )

    with scores_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["pair_id", "ll1", "ll2", "ground_truth", "score", "threshold", "predicted_label"])
        writer.writeheader()
        for row in rows:
            pred = row["score"] >= threshold
            writer.writerow(
                {
                    "pair_id": row["pair_id"],
                    "ll1": row.get("ll1", ""),
                    "ll2": row.get("ll2", ""),
                    "ground_truth": row["label"],
                    "score": f"{row['score']:.8f}",
                    "threshold": f"{threshold:.8f}",
                    "predicted_label": "CLONE" if pred else "NOT_CLONE",
                }
            )

    print(f"results={output_path}")
    print(f"scores={scores_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=sorted(EVAL_DATASET_PATHS), required=True)
    parser.add_argument("--model-name", default="microsoft/codebert-base")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--calibration-train-ratio", type=float, default=0.80)
    parser.add_argument("--no-cuda", action="store_true")
    args = parser.parse_args()

    eval_rows = _load_eval_rows(args.dataset)
    calibration_rows = _load_calibration_rows(args.dataset, eval_rows)
    if not calibration_rows:
        raise ValueError(
            f"No calibration rows found in {RAW_DATASET_PATHS[args.dataset]} after excluding final evaluation pairs."
        )

    splits = _stratified_train_valid_split(calibration_rows, args.seed, args.calibration_train_ratio)
    split_dir = BASELINE_DIR / "prepared" / args.dataset
    _write_pair_ids(split_dir / "raw_train_pair_ids.txt", splits["train"])
    _write_pair_ids(split_dir / "raw_valid_pair_ids.txt", splits["valid"])
    _write_pair_ids(split_dir / "full_eval_pair_ids.txt", eval_rows)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print(f"device={device}")
    print(f"model={args.model_name}")
    print(
        f"raw_calibration_pairs={len(calibration_rows)} "
        f"raw_train_pairs={len(splits['train'])} raw_valid_pairs={len(splits['valid'])} "
        f"full_eval_pairs={len(eval_rows)}"
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModel.from_pretrained(args.model_name).to(device)

    train_scores = _score_rows(splits["train"], model, tokenizer, device, args.batch_size, args.max_length)
    threshold = _best_threshold(train_scores)
    _print_metrics("raw calibration train metrics", train_scores, threshold)

    valid_scores = _score_rows(splits["valid"], model, tokenizer, device, args.batch_size, args.max_length)
    _print_metrics("raw calibration validation metrics", valid_scores, threshold)

    eval_scores = _score_rows(eval_rows, model, tokenizer, device, args.batch_size, args.max_length)
    _print_metrics("full data test metrics", eval_scores, threshold)
    _write_outputs(args.dataset, eval_scores, threshold)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
