"""
Pipeline 2: extract algorithms per fragment, compare in a third LLM call.
"""

from __future__ import annotations

import time
from typing import Any, List

from tqdm import tqdm

from src.core.constants import (
    ERROR,
    JAVA_LANGUAGE_IDENTIFIER,
    LABEL_TO_VERDICT,
    NOT_CLONE,
    PIPELINE_ALGO_BASED,
)
from src.core.logger import get_logger
from src.inference.prompts import ALGO_DETECTION_PROMPT, ALGO_EXTRACTION_PROMPT
from src.io.result_writer import ResultWriter
from src.inference.llm_helper import invoke_with_single_retry, pace_api_call
from src.inference.llm_response_parser import interpret_llm_response
from src.io.algorithm_writer import save_algorithm_pair

logger = get_logger(__name__)


def run_algo_based_workflow(
    llm: Any,
    records: List[dict[str, Any]],
    writer: ResultWriter,
    model_alias: str,
) -> None:
    """
    Execute Algorithm-Based Pipeline (three calls per pair) over all records.

    Args:
        llm: Chat model from :mod:`src.llm`.
        records: Normalized dataset rows.
        writer: ResultWriter for this run.
        model_alias: Model key for tqdm label.
    """
    for position, record in enumerate(tqdm(records, desc=f"algo_based/{model_alias}"), start=1):
        t0 = time.perf_counter()
        pair_id = record["pair_id"]

        language_b = record["ll2"]
        java_to_algo_extraction_prompt = ALGO_EXTRACTION_PROMPT.format(language=JAVA_LANGUAGE_IDENTIFIER, source_code=record["codeA"])
        other_to_algo_extraction_prompt = ALGO_EXTRACTION_PROMPT.format(language=language_b, source_code=record["codeB"])

        algo_java = invoke_with_single_retry(llm, java_to_algo_extraction_prompt).strip()
        pace_api_call()

        algo_b = invoke_with_single_retry(llm, other_to_algo_extraction_prompt).strip()
        pace_api_call()

        if not algo_java:
            algo_java = "(extraction failed — empty response)"
            logger.error("Java algorithm extraction empty for %s", pair_id)

        if not algo_b:
            algo_b = "(extraction failed — empty response)"
            logger.error("%s algorithm extraction empty for %s", language_b, pair_id)

        clone_detection_prompt = ALGO_DETECTION_PROMPT.format(
            algorithm_a=algo_java, algorithm_b=algo_b, language_b=language_b
        )

        llm_response_raw = invoke_with_single_retry(llm, clone_detection_prompt)

        elapsed = time.perf_counter() - t0

        if not llm_response_raw.strip():
            verdict, confidence, reasoning = ERROR, 0.5, "LLM compare call failed after retry."
            logger.error("Empty compare response for pair %s", pair_id)
        else:
            verdict, confidence, reasoning = interpret_llm_response(llm_response_raw)

        writer.record_result(
            pair_id=pair_id,
            ll1=record["ll1"],
            ll2=record["ll2"],
            ground_truth=record["label"],
            predicted_label=verdict,
        )

        if verdict != ERROR:
            save_algorithm_pair(
                pipeline=PIPELINE_ALGO_BASED,
                model_alias=model_alias,
                dataset_name=record["dataset"],
                algorithms_by_pair={
                    pair_id: {
                        "java_algorithm": algo_java,
                        "ll2_algorithm": algo_b,
                    },
                },
            )

        logger.info(
            "pair_id=%s pipeline=%s ground_truth=%s predicted=%s time=%.3fs",
            pair_id,
            PIPELINE_ALGO_BASED,
            LABEL_TO_VERDICT.get(record["label"], NOT_CLONE),
            verdict,
            elapsed,
        )

        pace_api_call()
