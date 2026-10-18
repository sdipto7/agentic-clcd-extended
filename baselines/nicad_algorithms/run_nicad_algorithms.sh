#!/usr/bin/env bash
set -euo pipefail

DATASET="${1:?Usage: bash baselines/nicad_algorithms/run_nicad_algorithms.sh <xlcost|codenet>}"
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${BASE_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/venv/bin/python}"

"${PYTHON_BIN}" "${BASE_DIR}/run_nicad_algorithms.py" --dataset "${DATASET}" --force
