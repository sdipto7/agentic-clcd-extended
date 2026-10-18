"""
LangChain tools for the Pipeline 3 ReAct agent: skills, comparison helper, result I/O.
"""

from __future__ import annotations

import time
import json
from typing import Any, Optional

from langchain_core.tools import tool

from src.core.constants import (
    COMPARE_TYPE_ALGORITHM,
    COMPARE_TYPE_SOURCE_CODE,
    SKILL_CACHE_REFRESH_INTERVAL_COUNTER,
)
from src.core.logger import get_logger
from src.io.result_writer import ResultWriter
from src.inference.agentic.skills import SKILL_REGISTRY

logger = get_logger(__name__)

_skill_body_cache: dict[str, str] = {}
_invocation_counter: int = 0
_active_writer: Optional[ResultWriter] = None
_context_pair_id: Optional[str] = None
_context_dataset: Optional[str] = None
_context_ground_truth: Optional[int] = None
_context_ll1: Optional[str] = None
_context_ll2: Optional[str] = None
_pair_start_time: Optional[float] = None
_write_result_called: bool = False
_last_predicted_label: Optional[str] = None
_algorithms_by_pair: dict[str, dict[str, str]] = {}


def get_recorded_algorithms() -> dict[str, dict[str, str]]:
    """Return all recorded algorithms collected during the run."""
    return _algorithms_by_pair


def _clear_skill_cache() -> None:
    """Clear the in-memory skill body cache of loaded skill instructions for fresh load."""
    global _skill_body_cache

    _skill_body_cache.clear()

    logger.info("Skill body cache cleared.")


def _refresh_skill_cache_if_needed() -> None:
    """Clear the skill cache if the invocation counter has reached the refresh threshold."""
    global _invocation_counter

    if _invocation_counter >= SKILL_CACHE_REFRESH_INTERVAL_COUNTER:
        _clear_skill_cache()
        _invocation_counter = 0


def set_active_result_writer(
    writer: Optional[ResultWriter],
    pair_id: Optional[str] = None,
    dataset: Optional[str] = None,
    ground_truth: Optional[int] = None,
    ll1: Optional[str] = None,
    ll2: Optional[str] = None,
) -> None:
    """
    Bind the ResultWriter and current pair metadata before each agent invocation.

    Args:
        writer: Writer instance for this run, or None to clear.
        pair_id: Identifier for the active pair.
        dataset: Dataset name for the active pair.
        ground_truth: 1 clone, 0 non-clone.
        ll1: Language of codeA for the active pair (always "Java" in this project).
        ll2: Language of codeB for the active pair (e.g. "Python", "C++", "C#").
    """
    global _active_writer, _context_pair_id, _context_dataset, _context_ground_truth
    global _pair_start_time, _write_result_called, _last_predicted_label, _invocation_counter
    global _context_ll1, _context_ll2

    if writer is not None:
        _invocation_counter += 1
        _refresh_skill_cache_if_needed()

    _active_writer = writer
    _context_pair_id = pair_id
    _context_dataset = dataset
    _context_ground_truth = ground_truth
    _context_ll1 = ll1
    _context_ll2 = ll2
    _pair_start_time = time.perf_counter() if writer is not None else None
    _write_result_called = False
    _last_predicted_label = None


def was_write_result_called() -> bool:
    """Return True if the agent invoked ``write_result`` for the current pair."""
    return _write_result_called


def get_last_predicted_label() -> Optional[str]:
    """Return the verdict last written by ``write_result`` for the active pair, if any."""
    return _last_predicted_label


def _parse_json_input(data: str) -> dict:
    """
    Extract the first JSON object (dict) from a string.

    Handles inputs like pure JSON, JSON with trailing text, or prefix + JSON + suffix
    by scanning for '{' and using ``json.JSONDecoder().raw_decode`` from that point.

    Args:
        data: Raw tool input string.

    Returns:
        Parsed dict if found; otherwise {}.
    """
    cleaned = data.strip()
    for idx, ch in enumerate(cleaned):
        if ch != "{":
            continue

        try:
            decoder = json.JSONDecoder()
            parsed, _ = decoder.raw_decode(cleaned[idx:])
        except json.JSONDecodeError:
            continue

        if isinstance(parsed, dict):
            return parsed

    return {}


@tool
def list_skills() -> str:
    """
    Return all available skills with their names and descriptions.
    Call this when you are unsure what skills exist or want to
    decide which skill is relevant for your current task.

    Returns:
        A formatted list of skill names and their descriptions.
    """
    if not SKILL_REGISTRY:
        return "No skills were discovered on disk."

    lines = ["=== SKILLS (load with load_skill) ==="]
    for info in sorted(SKILL_REGISTRY.values(), key=lambda x: x["name"]):
        lines.append(f"SKILL: {info['name']}")
        lines.append(f"  USE WHEN: {info['description']}")
        lines.append("")

    return "\n".join(lines)


@tool
def load_skill(data: str) -> str:
    """
    Retrieve the full instructions for a skill by name.
    Call this before performing any task that a skill covers.
    Never assume skill content without loading it first.
    Use list_skills first if you are unsure of which skill to use.

    Pass a JSON string with exactly this key:
        skill_name: exact name of the skill to load

    Args:
        data: JSON string containing skill_name (e.g., {"skill_name": "algorithm_extraction"}).
    Returns:
        Full skill instructions as plain text, or an error message if not found.
    """
    parsed = _parse_json_input(data)
    skill_name = str(next(iter(parsed.values()), data.strip())).strip()

    if not skill_name:
        known = ", ".join(sorted(SKILL_REGISTRY.keys())) or "(none)"
        return f"No skill name provided. Known skills: {known}"

    if skill_name in _skill_body_cache:
        return f"Skill '{skill_name}' is already loaded. Use the instructions you already have."

    entry = SKILL_REGISTRY.get(skill_name)
    if not entry:
        known = ", ".join(sorted(SKILL_REGISTRY.keys())) or "(none)"
        return f"Unknown skill {skill_name!r}. Known skills: {known}"

    body = entry["body"]
    _skill_body_cache[skill_name] = body
    logger.info("Skill loaded into cache: %s", skill_name)

    return body


@tool
def compare_and_decide(data: str) -> str:
    """
    Display two code fragments or algorithms side by side for comparison.
    Call this when you need to analyze two pieces of content together
    before making a clone detection decision.

    Pass a JSON string with exactly these keys:
        content_a      : Java source code or Algorithm A
        content_b      : source code in another language or Algorithm B
        comparison_type: source_code or algorithm

    Args:
        data: JSON string containing content_a, content_b, comparison_type.
    Returns:
        A formatted string showing both fragments labeled and side by side.
    """
    parsed = _parse_json_input(data)
    if not parsed:
        return (
            f"Invalid JSON: {data!r}. "
            "Pass a JSON string with keys: content_a, content_b, comparison_type."
        )

    content_a = str(parsed.get("content_a", "")).strip()
    content_b = str(parsed.get("content_b", "")).strip()
    comparison_type = str(parsed.get("comparison_type", "")).strip()

    if not content_a or not content_b:
        return "Both content_a and content_b must be non-empty."

    if comparison_type not in (COMPARE_TYPE_SOURCE_CODE, COMPARE_TYPE_ALGORITHM):
        return (
            f"Invalid comparison_type {comparison_type!r}. "
            f'Use "{COMPARE_TYPE_SOURCE_CODE}" or "{COMPARE_TYPE_ALGORITHM}".'
        )

    label_left = (
        "Java" if comparison_type == COMPARE_TYPE_SOURCE_CODE
        else "Algorithm A"
    )
    label_right = (
        "ll2" if comparison_type == COMPARE_TYPE_SOURCE_CODE
        else "Algorithm B"
    )

    return (
        f"=== {label_left} ===\n{content_a}\n\n"
        f"=== {label_right} ===\n{content_b}\n"
    )


@tool
def write_result(data: str) -> str:
    """
    Record the final clone detection verdict for the current pair.
    Call this exactly once after completing your analysis.
    Duplicate calls will be rejected.

    Pass a JSON string with exactly these keys:
        verdict   : CLONE or NOT_CLONE
        confidence: float between 0.0 and 1.0
        reasoning : max 50 words explaining your decision

    Args:
        data: JSON string containing verdict, confidence, and reasoning.
    Returns:
        Confirmation message if recorded successfully, or an error description.
    """
    global _write_result_called, _last_predicted_label

    if _write_result_called:
        return "A result was already recorded for this pair; do not call write_result again."

    if _active_writer is None:
        return "No active result writer configured; cannot record."

    if (
        _context_pair_id is None
        or _context_dataset is None
        or _context_ground_truth is None
        or _context_ll1 is None
        or _context_ll2 is None
    ):
        return "Internal error: pair context missing for write_result."

    parsed = _parse_json_input(data)
    if not parsed:
        return (
            f"Invalid JSON: {data!r}. "
            "Pass a JSON string with keys: verdict, confidence, reasoning."
        )

    verdict = str(parsed.get("verdict", "")).strip().upper()
    reasoning = str(parsed.get("reasoning", "")).strip()
    try:
        confidence = float(parsed.get("confidence", 0.5))
    except (TypeError, ValueError):
        confidence = 0.5

    if verdict in ("NOT CLONE", "NON_CLONE"):
        verdict = "NOT_CLONE"

    if verdict not in ("CLONE", "NOT_CLONE"):
        return f"Invalid verdict {verdict!r}; use CLONE or NOT_CLONE."

    confidence = max(0.0, min(1.0, confidence))

    elapsed = 0.0
    if _pair_start_time is not None:
        elapsed = time.perf_counter() - _pair_start_time

    _active_writer.record_result(
        pair_id=_context_pair_id,
        ll1=_context_ll1,
        ll2=_context_ll2,
        ground_truth=_context_ground_truth,
        predicted_label=verdict,
    )
    _write_result_called = True
    _last_predicted_label = verdict

    logger.info(
        "write_result tool: pair_id=%s ground_truth=%s predicted=%s time=%.3fs",
        _context_pair_id,
        _context_ground_truth,
        verdict,
        elapsed,
    )

    return f"Recorded result for {_context_pair_id}."


@tool
def record_algorithms(data: str) -> str:
    """
    Save the extracted algorithms for the current pair.
    Call this exactly once per pair, only after extracting algorithms
    from BOTH the Java fragment and the second fragment in another language.
    Only call this on algorithm-based detection runs.

    Pass a JSON string with exactly these keys:
        java_algorithm: algorithm extracted from the Java fragment
        ll2_algorithm : algorithm extracted from the second (non-Java) fragment -
                        use this exact literal key name "ll2_algorithm" regardless
                        of what language that fragment is actually written in.

    Args:
        data: JSON string containing exactly these keys: java_algorithm, ll2_algorithm.
    Returns:
        Confirmation message if saved successfully, or an error description.
    """
    if _context_pair_id is None:
        return "No active pair context; cannot record algorithms."

    if _context_pair_id in _algorithms_by_pair:
        return (
            f"Algorithms already recorded for {_context_pair_id}. "
            "Do not call record_algorithms more than once per pair."
        )

    parsed = _parse_json_input(data)
    if not parsed:
        return (
            f"Invalid JSON: {data!r}. "
            "Pass a JSON string with keys: java_algorithm, ll2_algorithm."
        )

    java_algorithm = str(parsed.get("java_algorithm", "")).strip()
    ll2_algorithm = str(parsed.get("ll2_algorithm", "")).strip()

    if not java_algorithm or not ll2_algorithm:
        return (
            "Both java_algorithm and ll2_algorithm must be non-empty. "
            "Extract algorithms from both fragments before calling this tool."
        )

    _algorithms_by_pair[_context_pair_id] = {
        "java_algorithm": java_algorithm,
        "ll2_algorithm": ll2_algorithm,
    }

    return f"Recorded algorithms for {_context_pair_id}."


def get_agent_tools() -> list[Any]:
    """Return all BaseTool instances used by Pipeline 3."""
    return [list_skills, load_skill, compare_and_decide, write_result, record_algorithms]
