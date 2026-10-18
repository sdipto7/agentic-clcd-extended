#!/usr/bin/env bash
set -euo pipefail
DATASET="${1:?Usage: $0 <xlcost|codenet> [java|py]}"
LANGUAGE="${2:-java}"
python baselines/nicad_source/run_nicad_source.py --dataset "$DATASET" --language "$LANGUAGE" --force
