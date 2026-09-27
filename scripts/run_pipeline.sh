#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# shellcheck source=scripts/load_env.sh
source "$SCRIPT_DIR/load_env.sh"

MODEL="${MODEL:-simlingo}"
SCENARIO_ID="${SCENARIO_ID:-town01_t_junction_1}"
TIME_BUDGET_MINUTES="${TIME_BUDGET_MINUTES:-10}"
MANAGE_CARLA="${MANAGE_CARLA:-1}"
PYTHON_BIN="${PYTHON_BIN:-python}"
TEST_CASES_ROOT="${TEST_CASES_ROOT:-test_cases/quickstart}"
MANIFEST="${MANIFEST:-$TEST_CASES_ROOT/manifest.jsonl}"

cd "$PROJECT_ROOT"
"$PYTHON_BIN" tools/check_environment.py --model "$MODEL"

cleanup() {
  if [[ "$MANAGE_CARLA" == "1" ]]; then
    "$SCRIPT_DIR/carla_manager.sh" stop
  fi
}
trap cleanup EXIT INT TERM

if [[ "$MANAGE_CARLA" == "1" ]]; then
  "$SCRIPT_DIR/carla_manager.sh" start "pipeline_${SCENARIO_ID}"
else
  "$SCRIPT_DIR/carla_manager.sh" wait-ready
fi

SCENARIO_ID="$SCENARIO_ID" "$SCRIPT_DIR/generate_static_scenarios.sh"

MODEL="$MODEL" \
STATIC_SCENARIO="static_scenarios/${SCENARIO_ID}.json" \
OUTPUT_ROOT="$TEST_CASES_ROOT" \
SEEDS_PER_SCENARIO=1 \
MAX_SEEDS_PER_SCENARIO=1 \
"$SCRIPT_DIR/generate_seeds.sh"

SEEDS_ROOT="$TEST_CASES_ROOT" \
MODE=route_prior \
NUM_COMMANDS=1 \
"$SCRIPT_DIR/generate_instructions.sh"

SEEDS_ROOT="$TEST_CASES_ROOT" \
STATIC_ROOT=static_scenarios \
MANIFEST="$MANIFEST" \
"$SCRIPT_DIR/build_manifest.sh"

MODEL="$MODEL" \
SELECTED_ROOT="$TEST_CASES_ROOT" \
MANIFEST="$MANIFEST" \
MANIFEST_ID="${SCENARIO_ID}/seed_1" \
NUM_COMMANDS=1 \
TIME_BUDGET_MINUTES="$TIME_BUDGET_MINUTES" \
"$SCRIPT_DIR/run_vladfuzz.sh"
