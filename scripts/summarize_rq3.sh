#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
RESULTS_ROOT="${RESULTS_ROOT:-results}"
MANIFEST="${MANIFEST:-configs/rq3_manifest.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-results/rq3/summary}"
EXPECTED_BUDGET_MINUTES="${EXPECTED_BUDGET_MINUTES:-240}"

cd "$PROJECT_ROOT"

"$PYTHON_BIN" -m evaluation.rq3.summarize_results \
  --results-root "$RESULTS_ROOT" \
  --manifest "$MANIFEST" \
  --output-dir "$OUTPUT_DIR" \
  --expected-budget-minutes "$EXPECTED_BUDGET_MINUTES"
