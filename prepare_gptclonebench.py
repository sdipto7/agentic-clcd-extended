#!/usr/bin/env python3
"""
Prepare a single balanced Java-X GPTCloneBench dataset (X in TARGET_LANGUAGES,
currently Python and C#), mirroring how prepare_dataset.py builds one combined
file per benchmark (xlcost.json, codenet.json).

The source archive mainly contains positive GPT-generated clone pairs. For each
target language this script extracts GPT-4 Java-to-X pairs from
cross_java_to_other/, keeps X snippets that pass a language-appropriate validity
check, and constructs deterministic non-clone pairs with a mixture of
challenging and safer negatives. Each language's clone+nonclone block is
appended in TARGET_LANGUAGES order into one output file, with pair_id assigned
sequentially by position across the whole file (gptclonebench_0, gptclonebench_1,
...) and a single shared "dataset" value ("gptclonebench") on every row.

Note: the archive's non-Python translation track is plain C, not C++ (C++/cpp
fence tags appear only a handful of times across the whole corpus) - so "csharp"
is the only additional language currently supported here.
"""

from __future__ import annotations

import ast
import json
import random
import re
import statistics
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_ZIP = PROJECT_ROOT / "raw_data" / "gptclonebench" / "cross_language_part_2.zip"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "gptclonebench.json"
DEFAULT_AUDIT = PROJECT_ROOT / "data" / "gptclonebench_audit.json"

DATASET_NAME = "gptclonebench"
SEED = 42
CLONE_COUNT_PER_LANGUAGE = 600
NONCLONE_COUNT_PER_LANGUAGE = 600
HARD_NEGATIVE_TARGET_PER_LANGUAGE = 528

# Languages extracted from raw_data/, in this fixed order, so that pair_id
# numbering is stable and reproducible on every run: rerunning this script
# always rebuilds the full Java-Python + Java-C# dataset from the raw archive
# from scratch, it never depends on what data/*.json already has.
TARGET_LANGUAGES: list[str] = ["python", "csharp"]

JAVA_ENTRY_RE = re.compile(r"^cross_java_to_other/Gpt4D_Clone(\d+)\.java$")
FENCED_BLOCK_RE = re.compile(r"```([A-Za-z0-9+#]*)\s*\n(.*?)```", re.DOTALL)
# Matches section-header lines like "C#:", "In Python:", "C# Code:", "For C:" that
# GPT-4 uses to separate per-language translations within one output. Deliberately
# narrow (only these known language names) so it never matches unrelated
# colon-terminated code lines like "else:" or "case 1:".
LANGUAGE_MARKER_LINE_RE = re.compile(
    r"(?:^|\n)[ \t]*(?:\d+\.\s*)?(?:in\s+|for\s+)?(C\#|C\+\+|C\s*Sharp|CSharp|Python|C)"
    r"(?:\s+code|\s+implementation)?[ \t]*:[ \t]*",
    re.IGNORECASE,
)
TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z_0-9]*|\d+|==|!=|<=|>=|&&|\|\||[{}()[\];,.:+\-*/%<>]")
JAVA_METHOD_RE = re.compile(
    r"(?:public|private|protected|static|final|synchronized|native|\s)+"
    r"[\w<>\[\], ?]+\s+([A-Za-z_][A-Za-z_0-9]*)\s*\("
)
PY_METHOD_RE = re.compile(r"^\s*def\s+([A-Za-z_][A-Za-z_0-9]*)\s*\(", re.MULTILINE)
CSHARP_METHOD_RE = re.compile(
    r"(?:public|private|protected|internal|static|virtual|override|abstract|sealed|"
    r"async|readonly|extern|unsafe|partial|new|\s)+"
    r"[\w<>\[\], ?]+\s+([A-Za-z_][A-Za-z_0-9]*)\s*\("
)


def python_parse_error(code: str) -> str | None:
    """Return an error description if ``code`` fails to parse as Python, else None."""
    try:
        ast.parse(code)
    except SyntaxError as exc:
        return f"{exc.__class__.__name__}: {exc.msg} at line {exc.lineno}"
    return None


def csharp_sanity_check(code: str) -> str | None:
    """
    Return an error description if ``code`` looks structurally broken, else None.

    No C# compiler is available here, so this is a lightweight heuristic (balanced
    braces/parens and a minimum non-trivial length) rather than a real syntax check.
    It mainly catches GPT outputs truncated mid-block.
    """
    stripped = code.strip()
    if len(stripped) < 10:
        return "TooShort: extracted block has fewer than 10 characters"
    if stripped.count("{") != stripped.count("}"):
        return "UnbalancedBraces: '{' and '}' counts differ"
    if stripped.count("(") != stripped.count(")"):
        return "UnbalancedParens: '(' and ')' counts differ"
    return None


@dataclass(frozen=True)
class LanguageSpec:
    """Everything that differs between target languages (X in Java-X)."""

    key: str  # short key used in audit field prefixes, e.g. "python", "csharp"
    ll2: str  # value written to the "ll2" field, e.g. "Python", "C#"
    fence_tags: frozenset[str]  # accepted ```lang fence tags, lowercased
    prefix_names: tuple[str, ...]  # leading "<name> code:" prefixes to strip
    fallback_keywords: tuple[str, ...]  # keyword sniff for untagged fenced blocks
    marker_names: tuple[str, ...]  # "<Name>:" marker line variants (no fence case)
    method_name_re: re.Pattern[str]
    validity_check: Callable[[str], Optional[str]]


LANGUAGE_SPECS: dict[str, LanguageSpec] = {
    "python": LanguageSpec(
        key="python",
        ll2="Python",
        fence_tags=frozenset({"python", "py"}),
        prefix_names=("python code", "python implementation", "python"),
        fallback_keywords=("def ", "import ", "print(", "input("),
        marker_names=("Python",),
        method_name_re=PY_METHOD_RE,
        validity_check=python_parse_error,
    ),
    "csharp": LanguageSpec(
        key="csharp",
        ll2="C#",
        fence_tags=frozenset({"csharp", "c#", "cs"}),
        prefix_names=("c# code", "csharp code", "c# implementation", "c#", "csharp"),
        fallback_keywords=("class ", "public ", "static ", "void ", "using "),
        marker_names=("C#", "CSharp", "C Sharp"),
        method_name_re=CSHARP_METHOD_RE,
        validity_check=csharp_sanity_check,
    ),
}


@dataclass(frozen=True)
class PositivePair:
    clone_id: int
    source_name: str
    java_code: str
    other_code: str
    java_tokens: frozenset[str]
    other_tokens: frozenset[str]
    java_method_names: frozenset[str]
    other_method_names: frozenset[str]


def normalize_newlines(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n").strip()


def token_set(text: str) -> frozenset[str]:
    return frozenset(t.lower() for t in TOKEN_RE.findall(text))


def jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def extract_java_and_gpt_output(raw_text: str) -> tuple[str, str]:
    text = normalize_newlines(raw_text)
    if "#====================" in text:
        java_part, gpt_part = text.split("#====================", 1)
    elif "#gpt output" in text:
        java_part, gpt_part = text.split("#gpt output", 1)
    else:
        java_part, gpt_part = text, ""

    java_part = re.sub(r"^\s*#input\s*", "", java_part, flags=re.IGNORECASE)
    gpt_part = re.sub(r"^\s*#gpt output=*\s*", "", gpt_part, flags=re.IGNORECASE)
    return normalize_newlines(java_part), normalize_newlines(gpt_part)


def clean_language_block(block: str, spec: LanguageSpec) -> str:
    code = normalize_newlines(block)
    for prefix_name in spec.prefix_names:
        pattern = rf"^\s*{re.escape(prefix_name)}\s*:\s*"
        new_code = re.sub(pattern, "", code, flags=re.IGNORECASE)
        if new_code != code:
            code = new_code
            break
    return normalize_newlines(code)


def _canonical_marker_language(name: str) -> str:
    """Normalize a marker-line language name to a stable key for comparison."""
    key = re.sub(r"\s+", "", name.strip().lower())
    if key in ("c#", "csharp"):
        return "csharp"
    if key in ("c++", "cpp"):
        return "cpp"
    return key  # "python", "c", or anything else as-is


def extract_via_markers(gpt_output: str, spec: LanguageSpec) -> str | None:
    """
    Extract a block anchored to an explicit "<Language>:" section header.

    Handles multi-language GPT outputs where every translation is given under its
    own untagged fenced block (no ```lang on the fence itself), only separated by
    a marker line such as "C#:" or "Python:". Each candidate segment runs only up
    to the NEXT marker line of any language (never to end-of-string), so a later
    section can never be swallowed into an earlier one.

    Within a matching segment, only a genuine fenced code block is accepted. A
    segment with no fenced block (e.g. GPT wrote "the direct equivalent doesn't
    exist in C#, similar functionality could be implemented using DLLImport...")
    is prose about the absence of a translation, not code, and is correctly
    treated as "no translation" here - a keyword sniff on raw prose is not safe,
    since ordinary sentences about code trivially contain words like "using" or
    "class".
    """
    target_canon = {_canonical_marker_language(name) for name in spec.marker_names}
    matches = list(LANGUAGE_MARKER_LINE_RE.finditer(gpt_output))
    if not matches:
        return None

    fenced_candidates: list[str] = []
    for index, match in enumerate(matches):
        if _canonical_marker_language(match.group(1)) not in target_canon:
            continue
        segment_start = match.end()
        segment_end = matches[index + 1].start() if index + 1 < len(matches) else len(gpt_output)
        segment = gpt_output[segment_start:segment_end]

        block_match = FENCED_BLOCK_RE.search(segment)
        if not block_match:
            continue
        block_text = clean_language_block(block_match.group(2), spec)
        if block_text:
            fenced_candidates.append(block_text)

    return fenced_candidates[-1] if fenced_candidates else None


def extract_language(gpt_output: str, spec: LanguageSpec) -> str | None:
    blocks = FENCED_BLOCK_RE.findall(gpt_output)
    tagged_blocks: list[str] = []
    fallback_blocks: list[str] = []

    for language, block in blocks:
        language_key = language.strip().lower()
        block_text = clean_language_block(block, spec)
        if not block_text:
            continue
        if language_key in spec.fence_tags:
            tagged_blocks.append(block_text)
        elif language_key == "":
            # Only genuinely untagged blocks are fallback candidates. A block
            # explicitly tagged as some OTHER known language (e.g. ```python
            # when we're looking for C#) must never be keyword-sniffed into
            # the wrong language - loose keywords like "class " match both.
            fallback_blocks.append(block_text)

    if tagged_blocks:
        return tagged_blocks[-1]

    marker_anchored = extract_via_markers(gpt_output, spec)
    if marker_anchored:
        return marker_anchored

    candidates = [
        block
        for block in fallback_blocks
        if any(keyword in block for keyword in spec.fallback_keywords)
    ]
    if candidates:
        return candidates[-1]

    return None


def method_names(
    java_code: str, other_code: str, spec: LanguageSpec
) -> tuple[frozenset[str], frozenset[str]]:
    java_names = frozenset(name.lower() for name in JAVA_METHOD_RE.findall(java_code))
    other_names = frozenset(name.lower() for name in spec.method_name_re.findall(other_code))
    return java_names, other_names


def load_gpt4_java_other_pairs(
    zip_path: Path, spec: LanguageSpec
) -> tuple[list[PositivePair], dict[str, Any]]:
    if not zip_path.is_file():
        raise FileNotFoundError(f"GPTCloneBench archive not found: {zip_path}")

    valid: list[PositivePair] = []
    empty_or_missing = 0
    invalid_count = 0
    invalid_examples: list[dict[str, str]] = []

    with zipfile.ZipFile(zip_path) as archive:
        java_entries = sorted(
            (
                (int(match.group(1)), info.filename)
                for info in archive.infolist()
                if (match := JAVA_ENTRY_RE.match(info.filename))
            ),
            key=lambda item: item[0],
        )

        for clone_id, filename in java_entries:
            raw_text = archive.read(filename).decode("utf-8", errors="replace")
            java_code, gpt_output = extract_java_and_gpt_output(raw_text)
            other_code = extract_language(gpt_output, spec)
            if not other_code:
                empty_or_missing += 1
                continue

            validity_error = spec.validity_check(other_code)
            if validity_error:
                invalid_count += 1
                if len(invalid_examples) < 10:
                    invalid_examples.append({"source_name": filename, "error": validity_error})
                continue

            java_names, other_names = method_names(java_code, other_code, spec)
            valid.append(
                PositivePair(
                    clone_id=clone_id,
                    source_name=filename,
                    java_code=java_code,
                    other_code=other_code,
                    java_tokens=token_set(java_code),
                    other_tokens=token_set(other_code),
                    java_method_names=java_names,
                    other_method_names=other_names,
                )
            )

    audit = {
        "source_archive": str(zip_path),
        "gpt_model_side": f"GPT-4 Java-to-{spec.ll2}",
        "java_files": len(java_entries),
        "valid_pairs": len(valid),
        "empty_or_missing": empty_or_missing,
        "validity_failures": invalid_count,
        "validity_failure_examples": invalid_examples,
    }
    return valid, audit


def choose_positive_subset(valid_pairs: list[PositivePair], count: int, rng: random.Random) -> list[PositivePair]:
    unique_pairs: dict[str, PositivePair] = {}
    shuffled = list(valid_pairs)
    rng.shuffle(shuffled)
    for pair in shuffled:
        unique_pairs.setdefault(pair.java_code, pair)

    unique_values = list(unique_pairs.values())
    if len(unique_values) < count:
        raise ValueError(
            f"Need {count} unique-Java valid positive pairs, found {len(unique_values)} "
            f"from {len(valid_pairs)} valid pairs"
        )
    selected = rng.sample(unique_values, count)
    return sorted(selected, key=lambda pair: pair.clone_id)


def candidate_allowed(anchor: PositivePair, candidate: PositivePair, java_sim: float) -> bool:
    if anchor.clone_id == candidate.clone_id:
        return False
    if java_sim >= 0.90:
        return False
    if anchor.java_method_names and candidate.java_method_names:
        if anchor.java_method_names & candidate.java_method_names:
            return False
    return True


def build_negative_pairs(
    positives: list[PositivePair],
    all_valid_pairs: list[PositivePair],
    rng: random.Random,
    count: int,
    hard_target: int,
    forbidden_exact_code_pairs: set[tuple[str, str]],
    spec: LanguageSpec,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    Build nonclone rows for one language. Rows are returned without pair_id/dataset
    set (the caller assigns those once the full multi-language file is assembled).
    """
    negatives: list[dict[str, Any]] = []
    used_pairs: set[tuple[int, int]] = set()
    used_exact_code_pairs: set[tuple[str, str]] = set(forbidden_exact_code_pairs)

    hard_anchors = rng.sample(positives, min(hard_target, len(positives)))
    hard_anchor_ids = {pair.clone_id for pair in hard_anchors}

    for anchor in positives:
        candidates: list[tuple[float, float, PositivePair]] = []
        for candidate in all_valid_pairs:
            java_sim = jaccard(anchor.java_tokens, candidate.java_tokens)
            if not candidate_allowed(anchor, candidate, java_sim):
                continue
            cross_sim = jaccard(anchor.java_tokens, candidate.other_tokens)
            candidates.append((java_sim, cross_sim, candidate))

        if not candidates:
            raise ValueError(f"No valid negative candidate found for clone_id={anchor.clone_id}")

        if anchor.clone_id in hard_anchor_ids:
            bucket = "hard"
            preferred = [c for c in candidates if c[0] >= 0.30]
            if not preferred:
                preferred = [c for c in candidates if c[0] >= 0.20]
                bucket = "medium"
            if not preferred:
                preferred = candidates
                bucket = "regular"
            preferred.sort(key=lambda item: (item[0], item[1], -item[2].clone_id), reverse=True)
        else:
            bucket = "regular"
            preferred = [c for c in candidates if c[0] < 0.30] or candidates
            preferred.sort(key=lambda item: (abs(len(anchor.java_code) - len(item[2].other_code)), -item[1]))

        chosen: tuple[float, float, PositivePair] | None = None
        for candidate_tuple in preferred:
            candidate = candidate_tuple[2]
            pair_key = (anchor.clone_id, candidate.clone_id)
            exact_pair_key = (anchor.java_code, candidate.other_code)
            if pair_key not in used_pairs and exact_pair_key not in used_exact_code_pairs:
                chosen = candidate_tuple
                used_pairs.add(pair_key)
                used_exact_code_pairs.add(exact_pair_key)
                break
        if chosen is None:
            raise ValueError(
                f"No unique exact negative candidate found for clone_id={anchor.clone_id}"
            )

        java_sim, cross_sim, negative_source = chosen
        negatives.append(
            {
                "codeA": anchor.java_code,
                "codeB": negative_source.other_code,
                "ll1": "Java",
                "ll2": spec.ll2,
                "problem_id_1": f"gpt4_clone_{anchor.clone_id}",
                "problem_id_2": f"gpt4_clone_{negative_source.clone_id}",
                "type": "nonclone",
                "label": 0,
                "source": "GPTCloneBench",
                "gpt_model_side": f"GPT-4 Java-to-{spec.ll2}",
                "java_source_name": anchor.source_name,
                "other_source_name": negative_source.source_name,
                "java_clone_id": anchor.clone_id,
                "other_clone_id": negative_source.clone_id,
                "negative_sampling_seed": seed,
                "negative_difficulty": bucket,
                "java_java_jaccard": round(java_sim, 6),
                "java_negative_other_jaccard": round(cross_sim, 6),
            }
        )

    negatives = negatives[:count]
    bucket_counts: dict[str, int] = {}
    java_sims = []
    cross_sims = []
    for row in negatives:
        bucket_counts[row["negative_difficulty"]] = bucket_counts.get(row["negative_difficulty"], 0) + 1
        java_sims.append(float(row["java_java_jaccard"]))
        cross_sims.append(float(row["java_negative_other_jaccard"]))

    audit = {
        "requested_nonclone_pairs": count,
        "requested_hard_negative_target": hard_target,
        "negative_bucket_counts": bucket_counts,
        "negative_java_java_jaccard": describe(java_sims),
        "negative_java_other_jaccard": describe(cross_sims),
        "rejected_as_probable_near_duplicate_rule": "candidate Java-Java Jaccard >= 0.90 or shared Java method name",
    }
    return negatives, audit


def describe(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    return {
        "min": round(min(values), 6),
        "median": round(statistics.median(values), 6),
        "mean": round(statistics.mean(values), 6),
        "max": round(max(values), 6),
    }


def duplicate_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    pair_ids = [row["pair_id"] for row in rows]
    labels = [int(row["label"]) for row in rows]
    exact_code_pairs = {(row["codeA"], row["codeB"]) for row in rows}
    validity_failures = []
    for row in rows:
        spec = LANGUAGE_SPECS[_language_key_for_ll2(row["ll2"])]
        error = spec.validity_check(row["codeB"])
        if error:
            validity_failures.append({"pair_id": row["pair_id"], "error": error})

    return {
        "total_records": len(rows),
        "clone_records": sum(1 for label in labels if label == 1),
        "nonclone_records": sum(1 for label in labels if label == 0),
        "unique_pair_ids": len(set(pair_ids)),
        "duplicate_pair_ids": len(pair_ids) - len(set(pair_ids)),
        "unique_exact_code_pairs": len(exact_code_pairs),
        "duplicate_exact_code_pairs": len(rows) - len(exact_code_pairs),
        "validity_failures_in_output": validity_failures[:20],
        "validity_failure_count_in_output": len(validity_failures),
    }


def _language_key_for_ll2(ll2: str) -> str:
    for key, spec in LANGUAGE_SPECS.items():
        if spec.ll2 == ll2:
            return key
    raise ValueError(f"Unknown ll2 value: {ll2!r}")


def build_language_block(
    language: str,
    zip_path: Path,
    seed: int,
    clone_count: int,
    nonclone_count: int,
    hard_negative_target: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build one language's clone+nonclone rows (pair_id/dataset not yet assigned)."""
    spec = LANGUAGE_SPECS[language]
    rng = random.Random(seed)
    valid_pairs, extraction_audit = load_gpt4_java_other_pairs(zip_path, spec)
    positives = choose_positive_subset(valid_pairs, clone_count, rng)

    clone_rows = [
        {
            "codeA": pair.java_code,
            "codeB": pair.other_code,
            "ll1": "Java",
            "ll2": spec.ll2,
            "problem_id_1": f"gpt4_clone_{pair.clone_id}",
            "problem_id_2": f"gpt4_clone_{pair.clone_id}",
            "type": "clone",
            "label": 1,
            "source": "GPTCloneBench",
            "gpt_model_side": f"GPT-4 Java-to-{spec.ll2}",
            "java_source_name": pair.source_name,
            "other_source_name": pair.source_name,
            "java_clone_id": pair.clone_id,
            "other_clone_id": pair.clone_id,
        }
        for pair in positives
    ]

    nonclone_rows, negative_audit = build_negative_pairs(
        positives=positives,
        all_valid_pairs=valid_pairs,
        rng=rng,
        count=nonclone_count,
        hard_target=hard_negative_target,
        forbidden_exact_code_pairs={(row["codeA"], row["codeB"]) for row in clone_rows},
        spec=spec,
        seed=seed,
    )

    audit = {
        "language": language,
        "seed": seed,
        "extraction": extraction_audit,
        "positive_sampling": {
            "requested_clone_pairs": clone_count,
            "selected_clone_pairs": len(clone_rows),
            "selected_clone_id_min": min(pair.clone_id for pair in positives),
            "selected_clone_id_max": max(pair.clone_id for pair in positives),
        },
        "negative_sampling": negative_audit,
    }
    return clone_rows + nonclone_rows, audit


def build_dataset(
    zip_path: Path = DEFAULT_ZIP,
    output: Path = DEFAULT_OUTPUT,
    audit_output: Path = DEFAULT_AUDIT,
    seed: int = SEED,
    clone_count_per_language: int = CLONE_COUNT_PER_LANGUAGE,
    nonclone_count_per_language: int = NONCLONE_COUNT_PER_LANGUAGE,
    hard_negative_target_per_language: int = HARD_NEGATIVE_TARGET_PER_LANGUAGE,
) -> None:
    """Build one combined Java-Python + Java-C# file, appended in TARGET_LANGUAGES order."""
    all_rows: list[dict[str, Any]] = []
    per_language_audit: dict[str, Any] = {}

    for language in TARGET_LANGUAGES:
        rows, audit = build_language_block(
            language=language,
            zip_path=zip_path,
            seed=seed,
            clone_count=clone_count_per_language,
            nonclone_count=nonclone_count_per_language,
            hard_negative_target=hard_negative_target_per_language,
        )
        # Clones first, then nonclones, within each language's block - matches
        # the ordering the single-language file previously used.
        rows.sort(key=lambda row: int(row["label"]) == 0)
        all_rows.extend(rows)
        per_language_audit[language] = audit

    for index, row in enumerate(all_rows):
        row["pair_id"] = f"{DATASET_NAME}_{index}"
        row["dataset"] = DATASET_NAME

    output.parent.mkdir(parents=True, exist_ok=True)
    audit_output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", encoding="utf-8") as f:
        json.dump(all_rows, f, indent=2)

    audit = {
        "dataset": DATASET_NAME,
        "languages": TARGET_LANGUAGES,
        "output_file": str(output),
        "per_language": per_language_audit,
        "output_validation": duplicate_stats(all_rows),
    }
    with audit_output.open("w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2)

    print(f"Wrote {len(all_rows)} records to {output}")
    print(f"Wrote audit report to {audit_output}")
    print(json.dumps(audit["output_validation"], indent=2))
    for language in TARGET_LANGUAGES:
        subset = [r for r in all_rows if r["ll2"] == LANGUAGE_SPECS[language].ll2]
        clones = sum(1 for r in subset if r["label"] == 1)
        non_clones = sum(1 for r in subset if r["label"] == 0)
        print(f"  ll2={LANGUAGE_SPECS[language].ll2}: {len(subset)} pairs (clones={clones}, non-clones={non_clones})")


if __name__ == "__main__":
    build_dataset()
