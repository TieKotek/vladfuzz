#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python is unavailable. Activate a VLAD-Fuzz Conda environment first." >&2
  exit 1
fi

cd "$PROJECT_ROOT"
bash -n scripts/*.sh
"$PYTHON_BIN" -m compileall -q \
  backends evaluation scenario scenario_construction \
  vladfuzz_runtime vladfuzz_workflows tools scripts
"$PYTHON_BIN" tools/check_environment.py --repository-only
"$PYTHON_BIN" -m unittest discover -s tests -v
"$PYTHON_BIN" -m unittest discover -s evaluation -v
"$PYTHON_BIN" -m unittest discover -s evaluation/rq1/tests -v
