---
name: algorithm_extraction
description: Instructions for how to extract a language-agnostic algorithm from source code through your own reasoning. Load this skill only once as it covers extraction rules for both the Java fragment and the second fragment, whichever language that one is written in.
---

# Algorithm Extraction
Load this skill once. Use the instructions to extract algorithms from both the Java fragment and the second fragment in another language before calling record_algorithms tool. Later, the algorithms will be used to compare the logic of two code fragments written in different programming languages to determine if they are clones.

## Extraction Rules - follow this order
1. Start with a single sentence summarizing what the overall code does.
2. Describe every function including the entry point as a FUNCTION block.
3. Inside each FUNCTION block, use numbered plain-English steps only. No syntax from any programming language whatsoever.
4. Preserve all logic exactly - every condition, loop bound, branch, validation check, early exit, and error path must appear.
5. Use generic data structure names only: list, map, set, queue, stack.
6. Do not include comments, type annotations, import descriptions, or any language-specific observations in the algorithm.

## Important Rules:
- Extract Algorithm A from Java first.
- Extract Algorithm B from the second fragment.
- Only then call record_algorithms tool with both algorithms together.
- Do NOT call record_algorithms tool after extracting just one algorithm.

## After extracting algorithms from both fragments
Pass a JSON string to record_algorithms tool with exactly these keys:

**"java_algorithm"** : the algorithm you extracted from the Java fragment

**"ll2_algorithm"** : the algorithm you extracted from the second fragment.

## Algorithm Format (plain text only)
Summary: one sentence describing overall behavior.

FUNCTION: 'name'

  1. first step in plain English

  2. second step in plain English
  ...

FUNCTION: 'name'

  1. first step in plain English

  2. second step in plain English
  ...
