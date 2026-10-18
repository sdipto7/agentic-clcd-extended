"""
Pipeline 3: ReAct agent with autonomous skill loading and tool use.
"""

from __future__ import annotations

import time
from typing import Any, List

from tqdm import tqdm

from src.inference.agentic.agent import build_react_executor
from src.core.constants import (
    ERROR,
    LABEL_TO_VERDICT,
    NOT_CLONE,
    PIPELINE_AGENTIC,
)
from src.core.logger import get_logger
from src.io.result_writer import ResultWriter
from src.inference.agentic.tools import (
    get_last_predicted_label,
    set_active_result_writer,
    was_write_result_called,
    get_recorded_algorithms,
)
from src.inference.llm_helper import pace_api_call
from src.inference.llm_response_parser import extract_verdict_from_text
from src.io.algorithm_writer import save_algorithm_pair

logger = get_logger(__name__)


def _goal_message(rec: dict[str, Any]) -> str:
    """
    Format the detection task for the agent. The agent autonomously decides how to use its skills and tools to complete the task.

    Args:
        rec: a normalized dataset record with pair_id, codeA, codeB

    Returns:
        Goal string passed to the agent executor.
    """
    language_b = rec["ll2"]
    return (
        f"Determine whether the following Java and {language_b} code fragments "
        f"are cross-language clones (pair ID: {rec['pair_id']}).\n\n"
        f"Java code:\n{rec['codeA']}\n\n"
        f"{language_b} code:\n{rec['codeB']}\n\n"
        f"Use your available skills and tools to reach a verdict, "
        f"then record your final result using write_result."
    )


def run_agentic_workflow(
    llm: Any,
    records: List[dict[str, Any]],
    writer: ResultWriter,
    model_alias: str,
) -> None:
    """
    Execute Pipeline 3: one ReAct episode per record.

    Args:
        llm: Chat model from :mod:`src.llm`.
        records: Normalized dataset rows.
        writer: ResultWriter for this run.
        model_alias: Model key for tqdm label.
    """
    executor = build_react_executor(llm)

    for position, record in enumerate(tqdm(records, desc=f"agentic/{model_alias}"), start=1):
        set_active_result_writer(
            writer,
            pair_id=record["pair_id"],
            dataset=record["dataset"],
            ground_truth=record["label"],
            ll1=record["ll1"],
            ll2=record["ll2"],
        )

        t0 = time.perf_counter()

        agent_output_text = ""
        try:
            result = executor.invoke({"input": _goal_message(record)})
            agent_output_text = str(result.get("output", ""))
        except Exception as exc:
            logger.exception("Agent crashed for pair %s: %s", record["pair_id"], exc)

        elapsed = time.perf_counter() - t0

        # Save algorithm for the pair if the agent used the algo_based approach
        # and the write_result is successfuly executed (identified by presence of recorded algorithms).
        pair_id = record["pair_id"]
        algorithms_by_pair = get_recorded_algorithms()
        if was_write_result_called() and pair_id in algorithms_by_pair:
            save_algorithm_pair(
                pipeline=PIPELINE_AGENTIC,
                model_alias=model_alias,
                dataset_name=record["dataset"],
                algorithms_by_pair={pair_id: algorithms_by_pair[pair_id]},
            )

        if was_write_result_called():
            final_verdict = get_last_predicted_label()
        else:
            # The agent gave a Final Answer without calling write_result (violates
            # the system prompt's rule #3). Recover a verdict from that free-text
            # answer rather than discarding the pair as ERROR. This is a fallback,
            # not a tool-verified result: it was not produced through the
            # structured write_result call, so it carries a weaker methodological
            # guarantee than tool-verified verdicts.
            final_verdict = extract_verdict_from_text(agent_output_text)
            if final_verdict is not None:
                logger.warning(
                    "Agent did not call write_result for %s; recovered verdict %s "
                    "from Final Answer text (untrusted fallback, not tool-verified).",
                    record["pair_id"],
                    final_verdict,
                )
            else:
                final_verdict = ERROR
                logger.warning(
                    "Agent did not call write_result for %s and no verdict could be "
                    "recovered from its Final Answer; recording ERROR.",
                    record["pair_id"],
                )

            writer.record_result(
                pair_id=record["pair_id"],
                ll1=record["ll1"],
                ll2=record["ll2"],
                ground_truth=record["label"],
                predicted_label=final_verdict,
            )

        logger.info(
            "pair_id=%s pipeline=%s ground_truth=%s predicted=%s time=%.3fs",
            record["pair_id"],
            PIPELINE_AGENTIC,
            LABEL_TO_VERDICT.get(record["label"], NOT_CLONE),
            final_verdict,
            elapsed,
        )

        # defensive cleanup to prevent information leakage across records
        set_active_result_writer(None)

        pace_api_call()
