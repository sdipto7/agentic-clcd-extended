#!/usr/bin/env bash
set -euo pipefail

DATASET="${1:?Usage: bash baselines/codebert/run_codebert.sh <xlcost|codenet>}"

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${BASE_DIR}/../.." && pwd)"
PYTHON_BIN="${PROJECT_ROOT}/venv/bin/python"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="python"
fi

"${PYTHON_BIN}" "${BASE_DIR}/run_codebert.py" --dataset "${DATASET}"
