#!/usr/bin/env python3
"""Run standard Open-NiCad directly on the Java/Python final-pair source files.

This is intentionally a direct-source baseline, not the algorithm-summary NiCad
variant. Open-NiCad's cross-system mode accepts one language grammar at a time,
so Java/Python source pairs are tested with native extensions and a selected
NiCad language. With language=java, the Python side is not extracted; with
language=py, the Java side is not extracted. That limitation is part of the
experiment result for direct cross-language source comparison.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASELINE_DIR = Path(__file__).resolve().parent
NICAD_ALGO_DIR = PROJECT_ROOT / "baselines" / "nicad_algorithms"
DEPS_DIR = NICAD_ALGO_DIR / "official" / "deps"
LOCAL_OPENTXL_DIR = DEPS_DIR / "opentxl"
LOCAL_TURINGPLUS_DIR = DEPS_DIR / "turingplus_patched"
LOCAL_NICAD_ROOT = NICAD_ALGO_DIR / "official" / "Open-NiCad"
DATASET_PATHS = {
    "xlcost": PROJECT_ROOT / "data" / "xlcost.json",
    "codenet": PROJECT_ROOT / "data" / "codenet.json",
    "gptclonebench": PROJECT_ROOT / "data" / "gptclonebench.json",
}


class NiCadSourceError(RuntimeError):
    pass


def _load_json(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f"Missing file: {path}")
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _row_label(row: dict[str, Any]) -> int:
    if "label" in row:
        return int(row["label"])
    row_type = str(row.get("type", "")).strip().lower()
    if row_type == "clone":
        return 1
    if row_type in {"nonclone", "non-clone", "not_clone", "not clone"}:
        return 0
    raise ValueError(f"Cannot infer label for pair {row.get('pair_id')}")


def _safe_pair_id(pair_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "_", pair_id)


def _only_java_python(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Restrict to Java-Python pairs; data/*.json now also carries Java-C++/C# rows."""
    return [row for row in rows if str(row.get("ll2", "")).strip() == "Python"]


def prepare_dataset(dataset: str, force: bool = False) -> tuple[Path, Path]:
    rows = _only_java_python(_load_json(DATASET_PATHS[dataset]))
    prepared_root = BASELINE_DIR / "prepared" / dataset
    java_dir = prepared_root / f"{dataset}_direct_java"
    python_dir = prepared_root / f"{dataset}_direct_python"
    map_path = prepared_root / "pair_map.tsv"

    if force and prepared_root.exists():
        shutil.rmtree(prepared_root)
    if force:
        clone_root = LOCAL_NICAD_ROOT / "nicadclones" / f"{dataset}_direct_java"
        if clone_root.exists():
            shutil.rmtree(clone_root)
    java_dir.mkdir(parents=True, exist_ok=True)
    python_dir.mkdir(parents=True, exist_ok=True)

    with map_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            delimiter="\t",
            fieldnames=["pair_id", "ll1", "ll2", "label", "java_file", "python_file"],
        )
        writer.writeheader()
        for row in rows:
            pair_id = str(row["pair_id"])
            safe_pair = _safe_pair_id(pair_id)
            java_file = java_dir / f"{safe_pair}__java.java"
            python_file = python_dir / f"{safe_pair}__python.py"
            java_file.write_text(str(row.get("codeA", "")), encoding="utf-8")
            python_file.write_text(str(row.get("codeB", "")), encoding="utf-8")
            writer.writerow(
                {
                    "pair_id": pair_id,
                    "ll1": row.get("ll1", ""),
                    "ll2": row.get("ll2", ""),
                    "label": _row_label(row),
                    "java_file": str(java_file.resolve()),
                    "python_file": str(python_file.resolve()),
                }
            )

    print(f"java_dir={java_dir.resolve()}")
    print(f"python_dir={python_dir.resolve()}")
    print(f"pair_map={map_path.resolve()}")
    print(f"pairs={len(rows)} files={len(rows) * 2}")
    return java_dir, python_dir


def _find_nicadcross() -> tuple[list[str], Path]:
    on_path = shutil.which("nicadcross")
    if on_path:
        return [on_path], BASELINE_DIR
    for candidate in (LOCAL_NICAD_ROOT / "bin" / "nicadcross", LOCAL_NICAD_ROOT / "nicadcross"):
        if candidate.is_file():
            return [str(candidate.resolve())], LOCAL_NICAD_ROOT.resolve()
    raise NiCadSourceError("Could not find nicadcross. Run/setup baselines/nicad_algorithms first.")


def _ensure_local_turingplus_links() -> None:
    lib_dir = LOCAL_TURINGPLUS_DIR / "lib"
    if not lib_dir.is_dir():
        return
    link_targets = {
        "/tmp/scanparse.x": lib_dir / "scanparse.x",
        "/tmp/semantic1.x": lib_dir / "semantic1.x",
        "/tmp/semantic2.x": lib_dir / "semantic2.x",
        "/tmp/tp2c.x": lib_dir / "tp2c.x",
        "/tmp/ssl.x": lib_dir / "ssl.x",
        "/tmp/tlibUNIX64.a": lib_dir / "tlibUNIX64.a",
        "/tmp/tlibUNIX64u.a": lib_dir / "tlibUNIX64u.a",
        "/tmp/tp2cUNIX64.mdp": lib_dir / "tp2cUNIX64.mdp",
        "/tmp/tp2cUNIX32.mdp": lib_dir / "tp2cUNIX32.mdp",
        "/tmp/cc": lib_dir / "cc",
        "/tmp/as": lib_dir / "as",
        "/tmp/ld": lib_dir / "ld",
        "/tmp/UNIX64": LOCAL_TURINGPLUS_DIR / "include" / "UNIX64",
        "/tmp/UNIX32": LOCAL_TURINGPLUS_DIR / "include" / "UNIX32",
        "/tmp/common": LOCAL_TURINGPLUS_DIR / "include" / "common",
    }
    for link_name, target in link_targets.items():
        link_path = Path(link_name)
        resolved_target = target.resolve()
        if link_path.exists() or link_path.is_symlink():
            if link_path.is_symlink() and link_path.resolve() == resolved_target:
                continue
            raise NiCadSourceError(f"Cannot create local Turing+ link because {link_path} already exists")
        link_path.symlink_to(resolved_target)


def _nicad_env() -> dict[str, str]:
    env = os.environ.copy()
    path_parts: list[str] = []
    if LOCAL_OPENTXL_DIR.is_dir():
        env["TXLBIN"] = str((LOCAL_OPENTXL_DIR / "bin").resolve())
        env["TXLLIB"] = str((LOCAL_OPENTXL_DIR / "lib").resolve())
        path_parts.append(env["TXLBIN"])
    if LOCAL_TURINGPLUS_DIR.is_dir():
        _ensure_local_turingplus_links()
        path_parts.append(str((LOCAL_TURINGPLUS_DIR / "bin").resolve()))
    if path_parts:
        env["PATH"] = os.pathsep.join(path_parts + [env.get("PATH", "")])
    return env


def _find_clone_xml(system1_dir: Path, system2_dir: Path) -> Path | None:
    system1_name = system1_dir.name
    system2_name = system2_dir.name
    roots = [LOCAL_NICAD_ROOT / "nicadclones" / system1_name, LOCAL_NICAD_ROOT / "nicadclones"]
    candidates: list[Path] = []
    for root in roots:
        if root.exists():
            candidates.extend(root.rglob(f"*{system1_name}*{system2_name}*-clones-*.xml"))
            candidates.extend(root.rglob("*-crossclones-*.xml"))
            candidates.extend(root.rglob("*-clones-*.xml"))
    candidates = [p for p in candidates if "classes" not in p.name and "withsource" not in p.name]
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def run_nicadcross(dataset: str, language: str, config: str) -> Path | None:
    java_dir = BASELINE_DIR / "prepared" / dataset / f"{dataset}_direct_java"
    python_dir = BASELINE_DIR / "prepared" / dataset / f"{dataset}_direct_python"
    if not java_dir.is_dir() or not python_dir.is_dir():
        raise FileNotFoundError(f"Prepared direct-source directories missing for {dataset}")
    nicad_cmd, nicad_cwd = _find_nicadcross()
    cmd = [*nicad_cmd, "functions", language, str(java_dir.resolve()), str(python_dir.resolve()), config]
    print("running=", " ".join(cmd))
    print(f"nicad_cwd={nicad_cwd}")
    completed = subprocess.run(
        cmd,
        cwd=nicad_cwd,
        env=_nicad_env(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    log_path = BASELINE_DIR / f"runs_{dataset}_{language}.log"
    log_path.write_text(completed.stdout, encoding="utf-8")
    print(f"log={log_path.resolve()}")
    if completed.returncode != 0:
        empty_cross_language = "Extracted 0 functions" in completed.stdout and "Renaming failed" in completed.stdout
        if empty_cross_language:
            print("warning=NiCad extracted 0 functions on one side; treating as zero direct cross-language clone detections")
            return None
        raise NiCadSourceError(f"NiCad cross-source run failed with exit code {completed.returncode}. See {log_path}")
    clone_xml = _find_clone_xml(java_dir, python_dir)
    if clone_xml is None:
        print("clone_xml=None")
    else:
        print(f"clone_xml={clone_xml.resolve()}")
    return clone_xml


def _load_pair_map(dataset: str) -> list[dict[str, str]]:
    map_path = BASELINE_DIR / "prepared" / dataset / "pair_map.tsv"
    with map_path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def _file_key(value: str) -> str:
    return Path(value).name


def parse_clone_pairs(xml_path: Path | None) -> set[frozenset[str]]:
    if xml_path is None:
        return set()
    text = xml_path.read_text(encoding="utf-8", errors="replace")
    pairs: set[frozenset[str]] = set()
    try:
        root = ET.fromstring(text)
        for clone in root.iter("clone"):
            files = []
            for source in clone.iter("source"):
                file_name = source.attrib.get("file") or source.attrib.get("filename") or source.attrib.get("sourcefile")
                if file_name:
                    files.append(_file_key(file_name))
            if len(files) >= 2:
                pairs.add(frozenset(files[:2]))
    except ET.ParseError:
        for block in re.findall(r"<clone\b.*?</clone>", text, flags=re.DOTALL | re.IGNORECASE):
            files = [_file_key(match) for match in re.findall(r'file="([^"]+)"', block)]
            if len(files) >= 2:
                pairs.add(frozenset(files[:2]))
    return pairs


def convert_predictions(dataset: str, language: str, clone_xml: Path | None) -> Path:
    pair_rows = _load_pair_map(dataset)
    clone_pairs = parse_clone_pairs(clone_xml)
    output_path = PROJECT_ROOT / "output" / "nicad_source" / f"results_nicad_source_{dataset}_{language}_full.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["pair_id", "ll1", "ll2", "ground_truth", "predicted_label"])
        writer.writeheader()
        for row in pair_rows:
            pair_key = frozenset({_file_key(row["java_file"]), _file_key(row["python_file"])})
            writer.writerow(
                {
                    "pair_id": row["pair_id"],
                    "ll1": row.get("ll1", ""),
                    "ll2": row.get("ll2", ""),
                    "ground_truth": row["label"],
                    "predicted_label": "CLONE" if pair_key in clone_pairs else "NOT_CLONE",
                }
            )
    print(f"results={output_path.resolve()}")
    print(f"reported_clone_pairs={len(clone_pairs)}")
    print(f"evaluated_pairs={len(pair_rows)}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=sorted(DATASET_PATHS))
    parser.add_argument("--language", default="java", choices=["java", "py"])
    parser.add_argument("--config", default="default-report")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--convert-only", action="store_true")
    parser.add_argument("--clone-xml", type=Path, default=None)
    args = parser.parse_args()

    if not args.convert_only:
        prepare_dataset(args.dataset, force=args.force)
    if args.prepare_only:
        return

    clone_xml = args.clone_xml
    if clone_xml is None and not args.convert_only:
        clone_xml = run_nicadcross(args.dataset, args.language, args.config)
    convert_predictions(args.dataset, args.language, clone_xml)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
