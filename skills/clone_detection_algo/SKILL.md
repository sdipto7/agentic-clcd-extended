---
name: clone_detection_algo
description: Instructions for how to detect cross-language clones by comparing two extracted algorithms through your own reasoning. Best suited for longer or logically complex fragments where extracting the core algorithm first makes comparison clearer and more reliable.
---

# Algorithm-Based Clone Detection
Apply this skill after both code fragments have been converted to language-agnostic algorithms by following the instructions of algorithm_extraction skill.

## Background
You are comparing two algorithms - Algorithm A extracted from Java and Algorithm B extracted from the second fragment of another programming language. Your task is to determine whether both algorithms implement the same computational logic, meaning they are cross-language code clones. This comparison is purely at the logical level - language, syntax, and naming differences are already eliminated in the algorithm.

## Reasoning Approach - follow this order
1. Read Algorithm A and Algorithm B independently: identify the entry point, the role of each function, data flow, and how each algorithm handles edge cases and errors.
2. Compare step by step: align loops, conditions, data structure operations, and return values conceptually - not by line count.
3. Judge whether both algorithms implement the same overall logic and produce the same output for the same input.

## Output
You MUST call write_result tool first before giving your Final Answer.

Pass a JSON string to write_result tool with exactly these keys:

**"verdict"** : "CLONE" if functionally identical, "NOT_CLONE" otherwise

**"confidence"** : a float between 0.0 and 1.0 representing how certain you are of your verdict based on the evidence

**"reasoning"** : max 100 words citing the specific behavioral evidence that determined your verdict
