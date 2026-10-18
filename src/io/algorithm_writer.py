import json
import os

from src.core.constants import OUTPUT_DIR_BY_PIPELINE


def save_algorithm_pair(
    *,
    pipeline: str,
    model_alias: str,
    dataset_name: str,
    algorithms_by_pair: dict[str, dict[str, str]],
) -> None:
    """
    Write extracted algorithms for one or more pairs to a single JSON file.

    The output is written under the pipeline-specific output directory as
    ``algorithms_<model_alias>_<dataset_name>.json``.

    Args:
        pipeline: Current running pipeline
        model_alias: Model alias for the current run
        dataset_name: Dataset name for the current run
        algorithms_by_pair: Pair_id -> {"java_algorithm": ..., "ll2_algorithm": ...}.

    Returns:
        None.
    """
    out_dir = OUTPUT_DIR_BY_PIPELINE[pipeline]
    os.makedirs(out_dir, exist_ok=True)

    json_name = f"algorithms_{model_alias}_{dataset_name}.json"
    json_path = os.path.join(out_dir, json_name)

    existing_algorithms_by_pair: dict[str, dict[str, str]] = {}
    if os.path.isfile(json_path):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)

            if isinstance(loaded, dict):
                existing_algorithms_by_pair = loaded

        except (OSError, json.JSONDecodeError):
            existing_algorithms_by_pair = {}

    for pair_id, algo in algorithms_by_pair.items():
        existing_algorithms_by_pair[pair_id] = algo

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(existing_algorithms_by_pair, f, ensure_ascii=False, indent=2)
