#!/usr/bin/env python3
"""Run NiCad on generated algorithm summaries converted to Python functions.

This adapter evaluates whether NiCad can identify clone pairs after the LLM has
already abstracted each source program into an algorithm description. It is not a
standard source-code NiCad baseline; it is NiCad over algorithm representations.
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
DEPS_DIR = BASELINE_DIR / "official" / "deps"
LOCAL_OPENTXL_DIR = DEPS_DIR / "opentxl"
LOCAL_TURINGPLUS_DIR = DEPS_DIR / "turingplus_patched"
LOCAL_NICAD_ROOT = BASELINE_DIR / "official" / "Open-NiCad"
DATASET_PATHS = {
    "xlcost": PROJECT_ROOT / "data" / "xlcost.json",
    "codenet": PROJECT_ROOT / "data" / "codenet.json",
    "gptclonebench": PROJECT_ROOT / "data" / "gptclonebench.json",
}
ALGORITHM_PATHS = {
    "xlcost": PROJECT_ROOT / "output" / "algo_based" / "algorithms_claude-3-haiku_xlcost.json",
    "codenet": PROJECT_ROOT / "output" / "algo_based" / "algorithms_claude-3-haiku_codenet.json",
    "gptclonebench": PROJECT_ROOT / "output" / "algo_based" / "algorithms_claude-3-haiku_gptclonebench.json",
}


class NiCadError(RuntimeError):
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


def _python_literal_lines(text: str) -> list[str]:
    lines = [line.strip() for line in str(text).replace("\r\n", "\n").split("\n")]
    return [line for line in lines if line]


def _write_algorithm_function(path: Path, function_name: str, text: str) -> None:
    lines = _python_literal_lines(text)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(f"def {function_name}():\n")
        if not lines:
            f.write("    return ''\n")
            return
        for index, line in enumerate(lines):
            f.write(f"    step_{index:04d} = {line!r}\n")
        joined = ", ".join(f"step_{index:04d}" for index in range(len(lines)))
        f.write(f"    return '\\n'.join([{joined}])\n")


def _only_java_python(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Restrict to Java-Python pairs; data/*.json now also carries Java-C++/C# rows."""
    return [row for row in rows if str(row.get("ll2", "")).strip() == "Python"]


def prepare_dataset(dataset: str, force: bool = False) -> Path:
    data_rows = _only_java_python(_load_json(DATASET_PATHS[dataset]))
    algorithms = _load_json(ALGORITHM_PATHS[dataset])
    prepared_dir = BASELINE_DIR / "prepared" / dataset / f"{dataset}_algorithms_py"
    map_path = BASELINE_DIR / "prepared" / dataset / "pair_map.tsv"

    if force and prepared_dir.exists():
        shutil.rmtree(prepared_dir)
    prepared_dir.mkdir(parents=True, exist_ok=True)
    map_path.parent.mkdir(parents=True, exist_ok=True)

    missing = []
    with map_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            delimiter="\t",
            fieldnames=["pair_id", "ll1", "ll2", "label", "java_file", "python_file", "java_function", "python_function"],
        )
        writer.writeheader()
        for row in data_rows:
            pair_id = str(row.get("pair_id"))
            if str(row.get("ll2", "")).strip() != "Python":
                missing.append(pair_id)
                continue
            algo = algorithms.get(pair_id)
            if not algo:
                missing.append(pair_id)
                continue
            java_algorithm = algo.get("java_algorithm")
            python_algorithm = algo.get("ll2_algorithm")
            if java_algorithm is None or python_algorithm is None:
                missing.append(pair_id)
                continue

            safe_pair = re.sub(r"[^A-Za-z0-9_]", "_", pair_id)
            java_file = prepared_dir / f"{safe_pair}__java_algorithm.py"
            python_file = prepared_dir / f"{safe_pair}__python_algorithm.py"
            java_function = f"{safe_pair}__java_algorithm"
            python_function = f"{safe_pair}__python_algorithm"
            _write_algorithm_function(java_file, java_function, str(java_algorithm))
            _write_algorithm_function(python_file, python_function, str(python_algorithm))
            writer.writerow(
                {
                    "pair_id": pair_id,
                    "ll1": row.get("ll1", ""),
                    "ll2": row.get("ll2", ""),
                    "label": _row_label(row),
                    "java_file": str(java_file.resolve()),
                    "python_file": str(python_file.resolve()),
                    "java_function": java_function,
                    "python_function": python_function,
                }
            )

    if missing:
        raise ValueError(f"Missing algorithm summaries for {len(missing)} pairs, first few: {missing[:5]}")
    print(f"prepared_dir={prepared_dir.resolve()}")
    print(f"pair_map={map_path.resolve()}")
    print(f"pairs={len(data_rows)} functions={len(data_rows) * 2}")
    return prepared_dir


def _candidate_nicad_commands(nicad_root: Path | None) -> list[tuple[list[str], Path]]:
    commands: list[tuple[list[str], Path]] = []
    on_path = shutil.which("nicad")
    if on_path:
        commands.append(([on_path], BASELINE_DIR))
    if nicad_root:
        root = nicad_root.resolve()
        for candidate in (root / "bin" / "nicad", root / "nicad"):
            if candidate.is_file():
                commands.append(([str(candidate)], root))
    local_root = LOCAL_NICAD_ROOT
    for candidate in (local_root / "bin" / "nicad", local_root / "nicad"):
        if candidate.is_file():
            commands.append(([str(candidate.resolve())], local_root.resolve()))
    return commands


def _find_nicad(nicad_root: Path | None) -> tuple[list[str], Path]:
    commands = _candidate_nicad_commands(nicad_root)
    if not commands:
        raise NiCadError(
            "NiCad was not found. Install Open-NiCad or pass --nicad-root. "
            "Expected either `nicad` on PATH or baselines/nicad_algorithms/official/Open-NiCad/bin/nicad."
        )
    return commands[0]


def _ensure_local_turingplus_links() -> None:
    lib_dir = LOCAL_TURINGPLUS_DIR / "lib"
    bin_dir = LOCAL_TURINGPLUS_DIR / "bin"
    if not lib_dir.is_dir() or not bin_dir.is_dir():
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
            raise NiCadError(f"Cannot create local Turing+ link because {link_path} already exists")
        link_path.symlink_to(resolved_target)


def _nicad_env() -> dict[str, str]:
    env = os.environ.copy()
    path_parts = []
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


def run_nicad(dataset: str, nicad_root: Path | None, config: str) -> Path:
    prepared_dir = BASELINE_DIR / "prepared" / dataset / f"{dataset}_algorithms_py"
    if not prepared_dir.is_dir():
        raise FileNotFoundError(f"Prepared algorithm directory missing: {prepared_dir}")
    nicad_cmd, nicad_cwd = _find_nicad(nicad_root)
    cmd = [*nicad_cmd, "functions", "py", str(prepared_dir.resolve()), config]
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
    log_path = BASELINE_DIR / f"runs_{dataset}.log"
    log_path.write_text(completed.stdout, encoding="utf-8")
    print(f"log={log_path.resolve()}")
    if completed.returncode != 0:
        raise NiCadError(f"NiCad failed with exit code {completed.returncode}. See {log_path}")
    return _find_clone_xml(prepared_dir, config)


def _find_clone_xml(prepared_dir: Path, config: str) -> Path:
    search_roots = [
        LOCAL_NICAD_ROOT / "nicadclones",
        BASELINE_DIR / "nicadclones",
        prepared_dir.parent / "nicadclones",
        prepared_dir,
    ]
    candidates: list[Path] = []
    for root in search_roots:
        if root.exists():
            candidates.extend(root.rglob("*-clones-*.xml"))
            candidates.extend(root.rglob("*-blind-clones-*.xml"))
    candidates = [p for p in candidates if "classes" not in p.name and "withsource" not in p.name]
    if not candidates:
        raise NiCadError("Could not find NiCad clone-pairs XML output after the run.")
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    print(f"clone_xml={candidates[0].resolve()}")
    return candidates[0]


def _load_pair_map(dataset: str) -> list[dict[str, str]]:
    map_path = BASELINE_DIR / "prepared" / dataset / "pair_map.tsv"
    with map_path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def _canonical_file(value: str) -> str:
    # NiCad reports paths inside its nicadclones copy, while our pair map keeps
    # the original prepared paths. The generated filenames are unique per run.
    return Path(value).name


def _file_from_source_node(node: ET.Element) -> str | None:
    for attr in ("file", "filename", "sourcefile"):
        if attr in node.attrib:
            return _canonical_file(node.attrib[attr])
    return None


def parse_clone_pairs(xml_path: Path) -> set[frozenset[str]]:
    text = xml_path.read_text(encoding="utf-8", errors="replace")
    pairs: set[frozenset[str]] = set()

    try:
        root = ET.fromstring(text)
        for clone in root.iter("clone"):
            files = []
            for source in clone.iter("source"):
                file_name = _file_from_source_node(source)
                if file_name:
                    files.append(file_name)
            if len(files) >= 2:
                pairs.add(frozenset(files[:2]))
    except ET.ParseError:
        pass

    if pairs:
        return pairs

    # Fallback for NiCad XML-like variants.
    for block in re.findall(r"<clone\b.*?</clone>", text, flags=re.DOTALL | re.IGNORECASE):
        files = [_canonical_file(match) for match in re.findall(r'file="([^"]+)"', block)]
        if len(files) >= 2:
            pairs.add(frozenset(files[:2]))
    return pairs


def convert_predictions(dataset: str, xml_path: Path) -> Path:
    pair_rows = _load_pair_map(dataset)
    clone_pairs = parse_clone_pairs(xml_path)
    output_path = PROJECT_ROOT / "output" / "nicad_algorithms" / f"results_nicad_algorithms_{dataset}_full.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["pair_id", "ll1", "ll2", "ground_truth", "predicted_label"])
        writer.writeheader()
        for row in pair_rows:
            pair_key = frozenset({_canonical_file(row["java_file"]), _canonical_file(row["python_file"])})
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
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--convert-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--nicad-root", type=Path, default=None)
    parser.add_argument("--clone-xml", type=Path, default=None)
    parser.add_argument("--config", default="default-report")
    args = parser.parse_args()

    if not args.convert_only:
        prepare_dataset(args.dataset, force=args.force)
    if args.prepare_only:
        return

    clone_xml = args.clone_xml
    if clone_xml is None:
        clone_xml = run_nicad(args.dataset, args.nicad_root, args.config)
    convert_predictions(args.dataset, clone_xml)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
