#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

# shellcheck source=scripts/load_env.sh
source "$SCRIPT_DIR/load_env.sh"

SEEDS_ROOT="${SEEDS_ROOT:-test_cases}"
STATIC_ROOT="${STATIC_ROOT:-static_scenarios}"
MANIFEST="${MANIFEST:-$SEEDS_ROOT/manifest.jsonl}"

cd "$PROJECT_ROOT"
"$PYTHON_BIN" scenario_construction/build_instruction_manifest.py \
  --seeds-root "$SEEDS_ROOT" \
  --static-root "$STATIC_ROOT" \
  --output "$MANIFEST" \
  --overwrite
