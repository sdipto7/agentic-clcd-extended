#!/usr/bin/env python3
"""Filter a project result CSV to a prepared CodeBERT split's pair IDs."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


BASELINE_DIR = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Existing project result CSV.")
    parser.add_argument("--dataset", required=True, choices=("xlcost", "codenet"))
    parser.add_argument("--split", default="test", choices=("train", "valid", "test"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prepared-dir", type=Path, default=BASELINE_DIR / "prepared")
    args = parser.parse_args()

    pair_ids_path = args.prepared_dir / args.dataset / f"{args.split}_pair_ids.txt"
    if not pair_ids_path.is_file():
        raise FileNotFoundError(f"Missing split pair IDs: {pair_ids_path}. Run CodeBERT first.")

    wanted = {line.strip() for line in pair_ids_path.open(encoding="utf-8") if line.strip()}
    with args.input.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = [row for row in reader if row.get("pair_id") in wanted]
        fieldnames = reader.fieldnames

    if not fieldnames:
        raise ValueError(f"No CSV header found in {args.input}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {args.output}")
    if len(rows) != len(wanted):
        print(f"warning: expected {len(wanted)} pair IDs from split, found {len(rows)} in input CSV", file=sys.stderr)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)

