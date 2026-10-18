#!/usr/bin/env python3
"""
Normalize experiment datasets to a common Java-X schema with stable IDs, where X is
each language in ``TARGET_LANGUAGE_IDENTIFIERS`` (Python, C++, C#).

Run once from the project root before experiments that use ``data/*.json``. Rerunning
always rebuilds each output file from raw_data/ from scratch, so it is safe to run
again after adding a new language to ``TARGET_LANGUAGE_IDENTIFIERS``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, List

# Allow `python prepare_dataset.py` without installing as a package.
_PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.core.constants import (  # noqa: E402
    DATASET_CODENET,
    DATASET_XLCOST,
    JAVA_LANGUAGE_IDENTIFIER,
    PYTHON_LANGUAGE_IDENTIFIER,
    CPP_LANGUAGE_IDENTIFIER,
    CSHARP_LANGUAGE_IDENTIFIER,
    RAW_JAVA_CN_PATH,
    RAW_JAVA_XL_PATH,
    DATA_CODENET_PATH,
    DATA_XLCOST_PATH,
)

# ll2 languages extracted from each raw benchmark, in this fixed order, so that
# pair_id numbering is stable and reproducible on every run: rerunning this
# script always rebuilds the full Java-Python + Java-C++ + Java-C# dataset
# from raw_data/ from scratch, it never depends on what data/*.json already has.
TARGET_LANGUAGE_IDENTIFIERS: List[str] = [
    PYTHON_LANGUAGE_IDENTIFIER, 
    CPP_LANGUAGE_IDENTIFIER, 
    CSHARP_LANGUAGE_IDENTIFIER
]


def _load_json_array(path: str) -> List[dict[str, Any]]:
    """Load a JSON array from disk or raise a clear error."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing raw file: {path}")

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Expected top-level JSON array in {path}")
    return data


def _keep_record(row: dict[str, Any], target_ll2: str) -> bool:
    """Return True if the row is Java (codeA) paired with 'target_ll2' (codeB)."""
    ll1 = str(row.get("ll1", "")).strip()
    ll2 = str(row.get("ll2", "")).strip()
    return ll1 == JAVA_LANGUAGE_IDENTIFIER and ll2 == target_ll2


def _add_label(row: dict[str, Any]) -> dict[str, Any]:
    """Return a shallow copy with 'label' 1 for clone and 0 for nonclone."""
    out = dict(row)
    t = str(out.get("type", "")).strip().lower()
    if t == "clone":
        out["label"] = 1
    elif t == "nonclone":
        out["label"] = 0
    else:
        raise ValueError(f"Unknown type field: {out.get('type')!r}")
    return out


def _normalize_record(
    row: dict[str, Any],
    dataset_name: str,
    pair_index: int,
) -> dict[str, Any]:
    """Normalize one row while preserving all original metadata fields."""
    out = _add_label(row)
    
    try:
        code_a = str(out["codeA"])
        code_b = str(out["codeB"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Malformed code fields: {exc}") from exc

    out["codeA"] = code_a
    out["codeB"] = code_b
    out["pair_id"] = f"{dataset_name}_{pair_index}"
    out["dataset"] = dataset_name

    return out


def _filter_dataset(
    records: List[dict[str, Any]],
    dataset_name: str,
    target_ll2: str,
    start_index: int,
) -> List[dict[str, Any]]:
    """
    Filter to Java-``target_ll2`` rows and normalize while preserving metadata.

    Args:
        records: Raw benchmark rows to filter.
        dataset_name: Logical dataset key (e.g. 'xlcost'); used for pair_id/dataset fields.
        target_ll2: Required 'll2' value to keep (the non-Java side of the pair).
        start_index: First pair_index to assign; lets callers append one
            language's block after another's without reusing pair_id values.

    Returns:
        Normalized records for this language, in raw file order.
    """
    kept: List[dict[str, Any]] = []

    for row in records:
        if not isinstance(row, dict) or not _keep_record(row, target_ll2):
            continue

        try:
            kept.append(
                _normalize_record(
                    row,
                    dataset_name=dataset_name,
                    pair_index=start_index + len(kept),
                )
            )
        except ValueError:
            continue

    return kept


def _build_dataset(
    records: List[dict[str, Any]],
    dataset_name: str,
) -> List[dict[str, Any]]:
    """
    Build the full normalized dataset by extracting each target language in order.

    Args:
        records: Raw benchmark rows (one source file, all language pairs).
        dataset_name: Logical dataset key (e.g. 'xlcost').

    Returns:
        Normalized records for all languages in ``TARGET_LANGUAGE_IDENTIFIERS``,
        concatenated in that order with sequential, non-overlapping pair_ids.
    """
    kept: List[dict[str, Any]] = []
    for language in TARGET_LANGUAGE_IDENTIFIERS:
        kept.extend(
            _filter_dataset(
                records,
                dataset_name=dataset_name,
                target_ll2=language,
                start_index=len(kept),
            )
        )

    return kept


def _write_output(path: str, rows: List[dict[str, Any]]) -> None:
    """Write pretty-printed JSON array."""
    parent = os.path.dirname(path)
    os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
        f.write("\n")


def _summarize(name: str, path: str, rows: List[dict[str, Any]], discarded: int) -> None:
    """Print human-readable statistics, including a per-language (ll2) breakdown."""
    clones = sum(1 for r in rows if r.get("label") == 1)
    non_clones = sum(1 for r in rows if r.get("label") == 0)
    print(f"=== {name} ===")
    print(f"Output file: {path}")
    print(f"Records kept: {len(rows)} (clones={clones}, non-clones={non_clones})")
    print(f"Records discarded: {discarded}")

    languages_seen: List[str] = []
    for row in rows:
        language = str(row.get("ll2", ""))
        if language not in languages_seen:
            languages_seen.append(language)

    for language in languages_seen:
        subset = [r for r in rows if str(r.get("ll2", "")) == language]
        sub_clones = sum(1 for r in subset if r.get("label") == 1)
        sub_non_clones = sum(1 for r in subset if r.get("label") == 0)
        print(
            f"  ll2={language}: {len(subset)} pairs "
            f"(clones={sub_clones}, non-clones={sub_non_clones})"
        )


def main() -> None:
    """Build and write the normalized Java-X datasets under the data directory."""

    for label, dataset_name, source_path, out_path in (
        (
            "XLCoST (java_xl.json -> xlcost.json)",
            DATASET_XLCOST,
            RAW_JAVA_XL_PATH,
            DATA_XLCOST_PATH,
        ),
        (
            "CodeNet (java_cn.json -> codenet.json)",
            DATASET_CODENET,
            RAW_JAVA_CN_PATH,
            DATA_CODENET_PATH,
        ),
    ):

        records = _load_json_array(source_path)
        kept = _build_dataset(records, dataset_name=dataset_name)

        # Rows not claimed by any language in TARGET_LANGUAGE_IDENTIFIERS.
        discarded = len(records) - len(kept)

        _write_output(out_path, kept)
        _summarize(label, out_path, kept, discarded)
        print()


if __name__ == "__main__":
    main()
