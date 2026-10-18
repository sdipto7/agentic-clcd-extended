"""
Helpers to parse LLM outputs for clone-detection JSON and verdict fallbacks.

Shared by direct and algorithm-based workflows to avoid duplication.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Optional, Tuple

from src.core.constants import CLONE, ERROR, NOT_CLONE

logger = logging.getLogger(__name__)


def _strip_code_fence(text: str) -> str:
    """Remove a single surrounding markdown ``` fence if present."""
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.split("\n")

        if lines[0].startswith("```"):
            lines = lines[1:]

        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()

    return stripped


def _parse_detection_json(raw_text: str) -> Optional[dict[str, Any]]:
    """
    Parse a JSON object with verdict, confidence, and reasoning from model text.

    Args:
        raw_text: Raw model output, optionally wrapped in markdown fences.

    Returns:
        Parsed dict if valid, else None.
    """
    text = _strip_code_fence(raw_text)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # Try first {...} block
        match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text, re.DOTALL)
        if not match:
            return None
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None

    if not isinstance(data, dict):
        return None

    if "verdict" not in data:
        return None

    return data


def _normalize_verdict_str(value: Any) -> Optional[str]:
    """
    Map assorted verdict strings to CLONE or NOT_CLONE.

    Args:
        value: Raw verdict field from JSON.

    Returns:
        CLONE, NOT_CLONE, or None if unrecognized.
    """
    if value is None:
        return None

    s = str(value).strip().upper()
    if s in ("CLONE", "1", "TRUE", "YES"):
        return CLONE

    if s in ("NOT_CLONE", "NOT CLONE", "NON_CLONE", "NONCLONE", "0", "FALSE", "NO"):
        return NOT_CLONE

    return None


def extract_verdict_from_text(raw_text: str) -> Optional[str]:
    """
    Heuristically find CLONE / NOT_CLONE in free-form text.

    Args:
        raw_text: Unstructured model output.

    Returns:
        CLONE or NOT_CLONE if found, else None.
    """
    upper = raw_text.upper()
    # Prefer explicit NOT_CLONE tokens first to avoid substring issues
    if "NOT_CLONE" in upper or "NOT CLONE" in upper or "NON_CLONE" in upper:
        return NOT_CLONE

    if re.search(r"\bCLONE\b", upper) and "NOT" not in upper.split("CLONE")[0][-10:]:
        return CLONE

    if "CLONE" in upper:
        return CLONE

    return None


def interpret_llm_response(raw_text: str) -> Tuple[str, float, str]:
    """
    Convert model output into verdict, confidence, and reasoning.

    On JSON failure, falls back to substring verdict search and confidence 0.5.

    Args:
        raw_text: Raw LLM response body.

    Returns:
        Tuple of (verdict, confidence, reasoning). Verdict may be ERROR if empty.
    """
    parsed = _parse_detection_json(raw_text)

    if parsed is not None:
        verdict = _normalize_verdict_str(parsed.get("verdict"))
        confidence_raw = parsed.get("confidence", 0.5)

        try:
            confidence = float(confidence_raw)
        except (TypeError, ValueError):
            confidence = 0.5
        confidence = max(0.0, min(1.0, confidence))

        reasoning = str(parsed.get("reasoning", "")).strip()
        
        if verdict is None:
            logger.warning(
                "Parsed JSON but verdict missing or invalid; using fallback. Raw: %s",
                raw_text[:500],
            )
            verdict = extract_verdict_from_text(raw_text) or ERROR
            confidence = 0.5

        return verdict, confidence, reasoning

    logger.warning("JSON parse failed; using text fallback. Raw: %s", raw_text[:500])
    
    verdict = extract_verdict_from_text(raw_text)
    if verdict is None:
        return ERROR, 0.5, ""
    
    return verdict, 0.5, "(recovered from non-JSON output)"
