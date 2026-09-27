#!/usr/bin/env bash
set -euo pipefail

# Run Instruction-CF for one selected manifest seed.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
MODEL="${MODEL:-simlingo}"
GPU_ID="${GPU_ID:-0}"
MANIFEST="${MANIFEST:-configs/experiment_manifest.jsonl}"
MANIFEST_ID="${MANIFEST_ID:-town01_t_junction_1/seed_1}"
TIME_BUDGET_MINUTES="${TIME_BUDGET_MINUTES:-240}"
SCENARIO_DURATION_FRAMES="${SCENARIO_DURATION_FRAMES:-500}"
SUCCESS_DISTANCE="${SUCCESS_DISTANCE:-5.0}"
MIN_NPC_COUNT="${MIN_NPC_COUNT:-0}"
MAX_NPC_COUNT="${MAX_NPC_COUNT:-3}"
RANDOM_SEED="${RANDOM_SEED:-0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/rq2/runs/instruction_counterfactual}"
ORACLE_CHECKS="${ORACLE_CHECKS:-default}"
VLADFUZZ_STRICT_LOCAL_ASSETS="${VLADFUZZ_STRICT_LOCAL_ASSETS:-1}"

cd "$PROJECT_ROOT"

# shellcheck source=scripts/huggingface_env.sh
source "$PROJECT_ROOT/scripts/huggingface_env.sh"

"$PYTHON_BIN" -m vladfuzz_workflows.instruction_counterfactual_testing \
  --model "$MODEL" \
  --gpu-id "$GPU_ID" \
  --manifest "$MANIFEST" \
  --manifest-id "$MANIFEST_ID" \
  --time-budget-minutes "$TIME_BUDGET_MINUTES" \
  --scenario-duration-frames "$SCENARIO_DURATION_FRAMES" \
  --success-distance "$SUCCESS_DISTANCE" \
  --min-npc-count "$MIN_NPC_COUNT" \
  --max-npc-count "$MAX_NPC_COUNT" \
  --random-seed "$RANDOM_SEED" \
  --output-root "$OUTPUT_ROOT" \
  --oracle-checks "$ORACLE_CHECKS"
