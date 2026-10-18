# Agentic-CLCD

Cross-language code clone detection (Java ↔ Python) with three experiment pipelines:

- **direct**: one LLM call per pair on raw source code
- **algo_based**: extract language-agnostic algorithms (2 calls) + compare algorithms (1 call)
- **agentic**: ReAct-style agent that uses tools + loads skills from the `skills/` folder

The project supports resumable runs and run-level token/cost logging.

---

## Repository structure

- Entry points
  - `main.py` — run one experiment (pipeline × model × dataset)
  - `prepare_dataset.py` — build normalized datasets under `data/`
  - `evaluate.py` — evaluate result CSVs and write reports
- Source packages
  - `src/core/` — constants, paths, logging
  - `src/io/` — dataset loader, result writer, token usage writer
  - `src/inference/` — OpenRouter LLM client, prompts, retry/pacing, response parsing
  - `src/workflows/` — direct, algo_based, agentic workflows
- Skills (used by agentic workflow)
  - `skills/` — Markdown `SKILL.md` files grouped by category
- Raw inputs
  - `raw_data/` — raw dataset JSONs used by dataset preparation

---

## Setup

### 1) Create a virtual environment and install dependencies

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2) Configure OpenRouter

Set the environment variable by creating a file named `.env` in the repo root:

```dotenv
OPENROUTER_API_KEY=YOUR_KEY_HERE
```

The LLM client is implemented in `src/inference/llm.py` and uses OpenRouter’s OpenAI-compatible API base URL.

---

## Prepare datasets

```bash
python prepare_dataset.py
```

This generates normalized datasets under `data/` (for example: `data/xlcost.json` and `data/codenet.json`). Pipelines will error if these files do not exist.

---

## Run experiments

Run one pipeline/model/dataset combination:

```bash
python main.py --pipeline direct --model deepseek_v3 --dataset xlcost
python main.py --pipeline algo_based --model deepseek_v3 --dataset xlcost
python main.py --pipeline agentic --model deepseek_v3 --dataset xlcost
```

To see valid choices (pipelines, models, datasets):

```bash
python main.py --help
```

Pipelines are implemented in:

- `src/workflows/direct_workflow.py`
- `src/workflows/algo_based_workflow.py`
- `src/workflows/agentic_workflow.py`

---

## Resumability and retry behavior

Re-running the same command is safe:

- If a pair already has a successful (non-`ERROR`) row in the results CSV, it is skipped.
- If a pair previously produced `ERROR`, it will be retried (stale `ERROR` rows are removed before retry).

This behavior is implemented in `src/io/result_helper.py`.

---

## Outputs

The project writes outputs under an `output/` folder in the repo root:

- `output/direct/`
- `output/algo_based/`
- `output/agentic/`

### 1) Per-pair results CSV

Each run appends rows to a file named like:

- `results_<model>_<dataset>.csv`

Example:

- `output/direct/results_deepseek_v3_xlcost.csv`

Rows include the pair id, ground truth, predicted label.

### 2) Run-level token usage CSV (crash-safe)

Each run appends one row to:

- `output/<pipeline>/token_usage.csv`

Token usage is recorded even when the run is interrupted or crashes (the status will reflect that).

Columns:

- `pipeline`, `model`, `dataset`, `pairs`, `elapsed_seconds`, `run_status`,
  `successful_requests`, `prompt_tokens`, `completion_tokens`, `total_tokens`

Token logging is implemented in `src/io/token_usage_writer.py`.

### 3) Extracted algorithm JSON (algorithm-based pipeline)

The algorithm-based pipeline may write extracted algorithms to:

- `output/algo_based/algorithms_<model>_<dataset>.json`

This is handled by `src/io/algorithm_writer.py`.

---

## Logging and console output

- Logs are written to a file under the `logs/` folder (created automatically):
  - `logs/experiment.log`
- No console logging handler is attached by default, so the terminal output should mainly be the `tqdm` progress bar.

Logging is configured in `src/core/logger.py`.

---

## Evaluation

Evaluate experiment outputs (recursive scan under `output/` by default):

```bash
python evaluate.py
```

Evaluate a single CSV:

```bash
python evaluate.py --file output/direct/results_deepseek_v3_xlcost.csv
```

Evaluation is done in `evaluate.py`.

---

## Troubleshooting

### Missing dataset files

If you see errors about missing `data/java_python_*.json` files, run:

```bash
python prepare_dataset.py
```

and confirm the raw inputs exist under `raw_data/`.

### Missing skills

If the agentic pipeline cannot find skills, confirm the `skills/` folder exists at the repo root and contains the expected `SKILL.md` files. The project root path is defined in `src/core/constants.py`.

### Agentic parsing failures

ReAct-style agents are sensitive to output formatting. If you see frequent parsing/format errors in the agentic pipeline, try:

- a more instruction-following model alias
- reducing temperature (default is deterministic)
- tightening prompts and parsing tolerance in the agentic workflow components under `src/inference/` and `src/workflows/agentic_workflow.py`
