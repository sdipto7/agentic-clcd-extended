#!/usr/bin/env bash
set -euo pipefail

DATASET="${1:-xlcost}"
THRESHOLD="${2:-0.5}"

cd "$(dirname "$0")/../.."

venv/bin/python baselines/clcdsa/run_clcdsa.py --dataset "$DATASET" --threshold "$THRESHOLD"
