---
name: clone_detection_direct
description: Instructions for how to detect cross-language clones by directly comparing raw Java source code against another language's source code through your own reasoning. Best suited for short or logically simple code fragments where the intent is easy to read from the code itself.
---

# Direct Cross-Language Clone Detection
Use this skill when comparing raw source code fragments directly.

## Background
Two code fragments are cross-language clones if they implement the same computational logic and produce the same output for the same input, regardless of programming language, syntax, naming conventions, or library choices used.

## Reasoning Approach - follow this order
1. Analyze the Java fragment and the other code fragment independently: identify their inputs, outputs, data structures, control flow, and how they handle edge cases and errors.
2. Compare behavior: check operation ordering, equivalent conditions, and matching handling of empty inputs, boundary values, and errors.
3. Judge whether both code fragments implement the same overall logic and produce the same output for the same input.

## Output
You MUST call write_result tool first before giving your Final Answer.

Pass a JSON string to write_result tool with exactly these keys:

**"verdict"** : "CLONE" if functionally identical, "NOT_CLONE" otherwise

**"confidence"** : a float between 0.0 and 1.0 representing how certain you are of your verdict based on the evidence

**"reasoning"** : max 100 words citing the specific behavioral evidence that determined your verdict
