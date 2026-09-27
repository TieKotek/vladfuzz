#!/usr/bin/env bash
set -euo pipefail

# Generate dynamic seed test cases from static scenario JSON files.
# Start CARLA before running this script. By default this does not load a VLA
# model; it only uses shared CARLA route/scenario utilities.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

# shellcheck source=scripts/load_env.sh
source "$SCRIPT_DIR/load_env.sh"

MODEL="${MODEL:-simlingo}"
STATIC_ROOT="${STATIC_ROOT:-static_scenarios}"
STATIC_SCENARIO="${STATIC_SCENARIO:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-test_cases}"
STRATEGY="${STRATEGY:-diverse}"
MAX_SEEDS_PER_SCENARIO="${MAX_SEEDS_PER_SCENARIO:-5}"
CANDIDATES_PER_GROUP="${CANDIDATES_PER_GROUP:-1}"
SEEDS_PER_SCENARIO="${SEEDS_PER_SCENARIO:-5}"
RANDOM_SEED="${RANDOM_SEED:-0}"
GPU_ID="${GPU_ID:-0}"
CARLA_HOST="${CARLA_HOST:-localhost}"
TM_SEED="${TM_SEED:-0}"
TIMEOUT="${TIMEOUT:-30}"
FRAME_RATE="${FRAME_RATE:-20}"
OVERWRITE="${OVERWRITE:-0}"
DYNAMIC_FILTER="${DYNAMIC_FILTER:-0}"
DURATION_FRAMES="${DURATION_FRAMES:-500}"
MIN_NPC_COUNT="${MIN_NPC_COUNT:-0}"
MAX_NPC_COUNT="${MAX_NPC_COUNT:-0}"
MIN_SPEED_DIFF_PERC="${MIN_SPEED_DIFF_PERC:-0.0}"
MAX_SPEED_DIFF_PERC="${MAX_SPEED_DIFF_PERC:-0.5}"
MIN_TARGET_DISTANCE="${MIN_TARGET_DISTANCE:-10.0}"
MAX_ROUTE_DISTANCE="${MAX_ROUTE_DISTANCE:-200.0}"

cd "$PROJECT_ROOT"

cmd=(
  "$PYTHON_BIN" scenario_construction/seed_generator.py
  --model "$MODEL"
  --static-root "$STATIC_ROOT"
  --output-root "$OUTPUT_ROOT"
  --strategy "$STRATEGY"
  --max-seeds-per-scenario "$MAX_SEEDS_PER_SCENARIO"
  --candidates-per-group "$CANDIDATES_PER_GROUP"
  --seeds-per-scenario "$SEEDS_PER_SCENARIO"
  --random-seed "$RANDOM_SEED"
  --gpu-id "$GPU_ID"
  --host "$CARLA_HOST"
  --tm-seed "$TM_SEED"
  --timeout "$TIMEOUT"
  --frame-rate "$FRAME_RATE"
  --duration-frames "$DURATION_FRAMES"
  --min-npc-count "$MIN_NPC_COUNT"
  --max-npc-count "$MAX_NPC_COUNT"
  --min-speed-diff-perc "$MIN_SPEED_DIFF_PERC"
  --max-speed-diff-perc "$MAX_SPEED_DIFF_PERC"
  --min-target-distance "$MIN_TARGET_DISTANCE"
  --max-route-distance "$MAX_ROUTE_DISTANCE"
)

if [[ -n "$STATIC_SCENARIO" ]]; then
  cmd+=(--static-scenario "$STATIC_SCENARIO")
fi

if [[ "$OVERWRITE" == "1" ]]; then
  cmd+=(--overwrite)
fi

if [[ "$DYNAMIC_FILTER" == "1" ]]; then
  cmd+=(--dynamic-filter)
fi

echo "Generating dynamic seeds from $STATIC_ROOT into $OUTPUT_ROOT"
"${cmd[@]}"
