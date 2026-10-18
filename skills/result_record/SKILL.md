---
name: result_record
description: Instructions for how to record your final clone detection verdict by calling the write_result tool.
---

# Recording a Clone Detection Result
Use this skill after you have completed your analysis and formed a final verdict. This is the last step for every pair.

## Before Calling write_result tool - Confirm You Have
1. `pair_id` for the active pair (provided in the task text),
2. Your final verdict: exactly `CLONE` or `NOT_CLONE`.
3. A confidence score between 0.0 and 1.0 representing how certain you are of your verdict based on the evidence.
4. A reasoning string of max 50 words citing the specific behavioral evidence that determined your verdict.

## Calling write_result tool
Pass a single JSON string as the data argument:
{"verdict": "your verdict", "confidence": "your confidence score", "reasoning": "your reasoning"}

The tool automatically binds the pair_id, dataset, and timing - do not pass these yourself.

## Important Rules
- Do not call `write_result` tool before completing your full analysis.
- Call `write_result` tool exactly once per pair.
