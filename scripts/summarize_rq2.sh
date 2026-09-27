#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-results/rq2/runs}"
DRIVEFUZZ_SRC="${DRIVEFUZZ_SRC:-$PROJECT_ROOT/baselines/drivefuzz/src}"
DRIVEFUZZ_RESULTS_ROOT="${DRIVEFUZZ_RESULTS_ROOT:-$PROJECT_ROOT/results/rq2/runs/drivefuzz}"
OUTPUT_DIR="${OUTPUT_DIR:-results/rq2/summary}"

cd "$PROJECT_ROOT"

"$PYTHON_BIN" -m evaluation.rq2.summarize_results \
  --experiment-root "$EXPERIMENT_ROOT" \
  --drivefuzz-src "$DRIVEFUZZ_SRC" \
  --drivefuzz-results-root "$DRIVEFUZZ_RESULTS_ROOT" \
  --output-dir "$OUTPUT_DIR"
