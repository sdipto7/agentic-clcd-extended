"""
Central configuration: verdict labels, pipeline and dataset names,
file paths, model mappings, and runtime tuning.

All other modules import configuration from here only.
"""

from __future__ import annotations

import os

PROJECT_ROOT: str = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))

CLONE: str = "CLONE"
NOT_CLONE: str = "NOT_CLONE"
ERROR: str = "ERROR"

PIPELINE_DIRECT: str = "direct"
PIPELINE_ALGO_BASED: str = "algo_based"
PIPELINE_AGENTIC: str = "agentic"

DATASET_XLCOST: str = "xlcost"
DATASET_CODENET: str = "codenet"
DATASET_GPTCLONEBENCH: str = "gptclonebench"
DATASET_CLCDSA: str = "clcdsa"

RAW_JAVA_XL_PATH: str = os.path.join(PROJECT_ROOT, "raw_data", "java_xl.json")
RAW_JAVA_CN_PATH: str = os.path.join(PROJECT_ROOT, "raw_data", "java_cn.json")

DATA_XLCOST_PATH: str = os.path.join(PROJECT_ROOT, "data", "xlcost.json")
DATA_CODENET_PATH: str = os.path.join(PROJECT_ROOT, "data", "codenet.json")
DATA_GPTCLONEBENCH_PATH: str = os.path.join(PROJECT_ROOT, "data", "gptclonebench.json")
DATA_CLCDSA_PATH: str = os.path.join(PROJECT_ROOT, "data", "clcdsa.json")

DATASET_FILE_MAP: dict[str, str] = {
    DATASET_XLCOST: DATA_XLCOST_PATH,
    DATASET_CODENET: DATA_CODENET_PATH,
    DATASET_GPTCLONEBENCH: DATA_GPTCLONEBENCH_PATH,
    DATASET_CLCDSA: DATA_CLCDSA_PATH,
}

MODEL_MAP: dict[str, str] = {
    "deepseek-v3.2": "deepseek/deepseek-v3.2",
    "llama-3.3-nemotron": "nvidia/llama-3.3-nemotron-super-49b-v1.5",
    "claude-3-haiku": "anthropic/claude-3-haiku",
    "gpt-4o-mini": "openai/gpt-4o-mini-2024-07-18",
}

OPENAI_API_BASE_URL: str = "https://openrouter.ai/api/v1"

LABEL_TO_VERDICT: dict[int, str] = {1: CLONE, 0: NOT_CLONE}

OUTPUT_BASE_DIR: str = os.path.join(PROJECT_ROOT, "output")
OUTPUT_DIR_BY_PIPELINE: dict[str, str] = {
    PIPELINE_DIRECT: os.path.join(OUTPUT_BASE_DIR, PIPELINE_DIRECT),
    PIPELINE_ALGO_BASED: os.path.join(OUTPUT_BASE_DIR, PIPELINE_ALGO_BASED),
    PIPELINE_AGENTIC: os.path.join(OUTPUT_BASE_DIR, PIPELINE_AGENTIC),
}

LOGS_DIR: str = os.path.join(PROJECT_ROOT, "logs")
LOG_FILE_PATH: str = os.path.join(LOGS_DIR, "experiment.log")

API_CALL_DELAY_SECONDS: float = 0.5

API_RETRY_BACKOFF_SECONDS: float = 3.0

PYTHON_LANGUAGE_IDENTIFIER: str = "Python"

CPP_LANGUAGE_IDENTIFIER: str = "C++"

CSHARP_LANGUAGE_IDENTIFIER: str = "C#"

JAVA_LANGUAGE_IDENTIFIER: str = "Java"

AGENT_MAX_ITERATIONS: int = 10

SKILL_CACHE_REFRESH_INTERVAL_COUNTER: int = 10

PIPELINE_CHOICES: list[str] = [PIPELINE_DIRECT, PIPELINE_ALGO_BASED, PIPELINE_AGENTIC]
MODEL_CHOICES: list[str] = list(MODEL_MAP.keys())
DATASET_CHOICES: list[str] = [
    DATASET_XLCOST,
    DATASET_CODENET,
    DATASET_GPTCLONEBENCH,
    DATASET_CLCDSA,
]

RESULTS_CSV_PREFIX: str = "results"

COMPARE_TYPE_SOURCE_CODE: str = "source_code"
COMPARE_TYPE_ALGORITHM: str = "algorithm"
