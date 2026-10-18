"""
LLM prompt templates for Pipeline 1 (direct) and Pipeline 2 (algorithm-based).

All user-facing model instructions for those pipelines live here exclusively.
"""

from __future__ import annotations

# Pipeline 1: single-call cross-language clone decision from raw source.
DIRECT_DETECTION_PROMPT: str = """You are an expert in program analysis and cross-language code equivalence.

BACKGROUND:
Two code fragments are cross-language clones if they implement the same computational logic and produce the same output for the same input, regardless of the programming language, syntax, naming conventions, or library choices used.

TASK:
Determine whether the Java and {language_b} code fragments below are clones.

REASONING APPROACH — follow this order:
1. Analyze the Java and {language_b} code fragments independently: identify their inputs, outputs, data structures, control flow, and how they handle edge cases and errors.
2. Compare their behavior: check operation ordering, equivalent conditions, and matching handling of empty inputs, boundary values, and errors.
3. Judge whether both code fragments implement the same overall logic and produce the same output for the same input.

Java fragment:
{codeA}

{language_b} fragment:
{codeB}

OUTPUT RULES:
- Respond with a single JSON object only.
- No markdown fences, no text before or after the JSON.
- Use exactly these keys:
    "verdict"    : "CLONE" if functionally identical, "NOT_CLONE" otherwise
    "confidence" : a float between 0.0 and 1.0 representing how certain you are of your verdict based on the evidence
    "reasoning"  : max 100 words citing the specific behavioral evidence that determined your verdict

Example output shape:
{{"verdict": "CLONE", "confidence": 0.92, "reasoning": "Both iterate and aggregate the same way."}}
"""

# Pipeline 2 step 1 & 2: extract language-agnostic algorithm from one fragment.
ALGO_EXTRACTION_PROMPT: str = """You are an expert in program analysis and algorithm extraction.

BACKGROUND:
Your task is to convert a {language} code fragment into a precise, language-agnostic algorithm. This algorithm will be used 
to compare the computational logic of two code fragments written in different programming languages to determine if they are clones.
Accuracy and completeness are critical — every logical detail must be preserved.

SOURCE CODE ({language}):
{source_code}

EXTRACTION RULES - follow this order:
1. Start with a single sentence summarizing what the overall code does.
2. Describe every function including the entry point as a FUNCTION block.
3. Inside each FUNCTION block, use numbered plain-English steps only. No syntax from any programming language whatsoever.
4. Preserve all logic exactly - every condition, loop bound, branch, validation check, early exit, and error path must appear.
5. Use generic data structure names only: list, map, set, queue, stack.
6. Do not include comments, type annotations, import descriptions, or any language-specific observations in the algorithm.

OUTPUT FORMAT (plain text only):
Summary: <one sentence describing overall behavior>
FUNCTION <name>
  1. <first step in plain English>
  2. <second step in plain English>
  ...
FUNCTION <name>
  1. ...
"""

# Pipeline 2 step 3: compare two extracted algorithms for semantic equivalence.
ALGO_DETECTION_PROMPT: str = """You are an expert in program analysis and algorithm equivalence.

BACKGROUND:
You are given two pseudocode algorithms extracted from code fragments written in different programming languages - Algorithm A from Java and Algorithm B from {language_b}.
Your task is to determine whether both algorithms implement the same computational logic, meaning they are cross-language code clones.
This comparison is purely at the logical level - language, syntax, and naming differences are already eliminated in the pseudocode.

Algorithm A (extracted from Java):
{algorithm_a}

Algorithm B (extracted from {language_b}):
{algorithm_b}

REASONING APPROACH — follow this order:
1. Read Algorithm A and Algorithm B independently: identify their entry points, the role of each function, data flow, and how they handle edge cases and errors.
2. Compare step by step: align loops, conditions, data structure operations, and return values conceptually — not by line count.
3. Judge whether both algorithms implement the same overall logic and produce the same output for the same input.

OUTPUT RULES:
- Respond with a single JSON object only.
- No markdown fences, no text before or after the JSON.
- Use exactly these keys:
    "verdict"    : "CLONE" if the algorithms are semantically equivalent, "NOT_CLONE" otherwise
    "confidence" : a float between 0.0 and 1.0 representing how certain you are of your verdict based on the evidence
    "reasoning"  : max 100 words citing the specific steps that matched or diverged to justify your verdict

Example output shape:
{{"verdict": "CLONE", "confidence": 0.91, "reasoning": "Both algorithms iterate through the list maintaining a running maximum and return it after the loop. Edge case handling for empty input is identical in both - returning null immediately. Operation ordering and conditions match exactly."}}
"""
