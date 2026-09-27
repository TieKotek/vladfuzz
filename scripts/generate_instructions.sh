#!/usr/bin/env bash
set -euo pipefail

# Generate natural-language instructions for seed test cases.
# MODE can be route_prior, no_prior, or both.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

# shellcheck source=scripts/load_env.sh
source "$SCRIPT_DIR/load_env.sh"

SEEDS_ROOT="${SEEDS_ROOT:-test_cases}"
MODE="${MODE:-both}"
GENERATOR_MODEL="${GENERATOR_MODEL:-qwen3.6-plus-2026-04-02}"
NUM_COMMANDS="${NUM_COMMANDS:-1}"
LIMIT="${LIMIT:-}"
OVERWRITE="${OVERWRITE:-0}"
ROUTE_PRIOR_PROMPT="${ROUTE_PRIOR_PROMPT:-scenario_construction/prompt_prior.txt}"
NO_PRIOR_PROMPT="${NO_PRIOR_PROMPT:-scenario_construction/prompt_no_prior.txt}"

cd "$PROJECT_ROOT"

cmd=(
  "$PYTHON_BIN" scenario_construction/generate_instructions.py
  --seeds-root "$SEEDS_ROOT"
  --mode "$MODE"
  --route-prior-prompt "$ROUTE_PRIOR_PROMPT"
  --no-prior-prompt "$NO_PRIOR_PROMPT"
  --model "$GENERATOR_MODEL"
  --num-commands "$NUM_COMMANDS"
)

if [[ -n "$LIMIT" ]]; then
  cmd+=(--limit "$LIMIT")
fi

if [[ "$OVERWRITE" == "1" ]]; then
  cmd+=(--overwrite)
fi

echo "Generating $MODE instructions under $SEEDS_ROOT"
"${cmd[@]}"
