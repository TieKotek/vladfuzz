#!/usr/bin/env bash
set -euo pipefail

# Run one RQ3 ablation campaign selected by CONFIG.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
CONFIG="${CONFIG:-global_only}"
MODEL="${MODEL:-simlingo}"
GPU_ID="${GPU_ID:-0}"
MANIFEST="${MANIFEST:-configs/rq3_manifest.jsonl}"
MANIFEST_ID="${MANIFEST_ID:-town01_t_junction_1/seed_1}"
TIME_BUDGET_MINUTES="${TIME_BUDGET_MINUTES:-240}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/rq3/runs/$CONFIG}"
RANDOM_SEED="${RANDOM_SEED:-0}"
SCENARIO_DURATION_FRAMES="${SCENARIO_DURATION_FRAMES:-500}"
SUCCESS_DISTANCE="${SUCCESS_DISTANCE:-5.0}"
ORACLE_CHECKS="${ORACLE_CHECKS:-default}"
MUTATION_DEPTH="${MUTATION_DEPTH:-2}"
OPERATORS="${OPERATORS:-all}"
POP_SIZE="${POP_SIZE:-8}"
NGEN="${NGEN:-999}"
CXPB="${CXPB:-0.7}"
MUTPB="${MUTPB:-1.0}"
MIN_NPC_COUNT="${MIN_NPC_COUNT:-0}"
MAX_NPC_COUNT="${MAX_NPC_COUNT:-3}"
LLM_PROVIDER="${LLM_PROVIDER:-deepseek}"
LLM_MODEL="${LLM_MODEL:-deepseek-v4-flash}"
VLADFUZZ_STRICT_LOCAL_ASSETS="${VLADFUZZ_STRICT_LOCAL_ASSETS:-1}"

cd "$PROJECT_ROOT"

# shellcheck source=scripts/huggingface_env.sh
source "$PROJECT_ROOT/scripts/huggingface_env.sh"

case "$CONFIG" in
  global_only)
    echo "RQ3 Global-only: $MODEL / $MANIFEST_ID / ${TIME_BUDGET_MINUTES}min"
    "$PYTHON_BIN" -m vladfuzz_workflows.nsga_optimization \
      --model "$MODEL" \
      --gpu-id "$GPU_ID" \
      --manifest "$MANIFEST" \
      --manifest-id "$MANIFEST_ID" \
      --instruction-source route_prior \
      --time-budget-minutes "$TIME_BUDGET_MINUTES" \
      --mutation-depth 0 \
      --pop-size "$POP_SIZE" \
      --ngen "$NGEN" \
      --cxpb "$CXPB" \
      --mutpb "$MUTPB" \
      --scenario-duration-frames "$SCENARIO_DURATION_FRAMES" \
      --success-distance "$SUCCESS_DISTANCE" \
      --min-npc-count "$MIN_NPC_COUNT" \
      --max-npc-count "$MAX_NPC_COUNT" \
      --random-seed "$RANDOM_SEED" \
      --method global_only \
      --output-root "$OUTPUT_ROOT" \
      --operators "$OPERATORS" \
      --oracle-checks "$ORACLE_CHECKS" \
      --llm-provider "$LLM_PROVIDER" \
      --llm-model "$LLM_MODEL"
    ;;
  local_only)
    echo "RQ3 Local-only: $MODEL / $MANIFEST_ID / ${TIME_BUDGET_MINUTES}min"
    "$PYTHON_BIN" -m vladfuzz_workflows.local_only_testing \
      --model "$MODEL" \
      --gpu-id "$GPU_ID" \
      --manifest "$MANIFEST" \
      --manifest-id "$MANIFEST_ID" \
      --time-budget-minutes "$TIME_BUDGET_MINUTES" \
      --mutation-depth "$MUTATION_DEPTH" \
      --scenario-duration-frames "$SCENARIO_DURATION_FRAMES" \
      --success-distance "$SUCCESS_DISTANCE" \
      --random-seed "$RANDOM_SEED" \
      --output-root "$OUTPUT_ROOT" \
      --operators "$OPERATORS" \
      --oracle-checks "$ORACLE_CHECKS" \
      --llm-provider "$LLM_PROVIDER" \
      --llm-model "$LLM_MODEL"
    ;;
  random_scenario_local)
    echo "RQ3 Random-scenario Local: $MODEL / $MANIFEST_ID / ${TIME_BUDGET_MINUTES}min"
    "$PYTHON_BIN" -m vladfuzz_workflows.nsga_optimization \
      --model "$MODEL" \
      --gpu-id "$GPU_ID" \
      --manifest "$MANIFEST" \
      --manifest-id "$MANIFEST_ID" \
      --instruction-source route_prior \
      --time-budget-minutes "$TIME_BUDGET_MINUTES" \
      --mutation-depth "$MUTATION_DEPTH" \
      --pop-size 0 \
      --ngen 0 \
      --cxpb 0 \
      --mutpb 0 \
      --initial-population-until-budget \
      --scenario-duration-frames "$SCENARIO_DURATION_FRAMES" \
      --success-distance "$SUCCESS_DISTANCE" \
      --min-npc-count "$MIN_NPC_COUNT" \
      --max-npc-count "$MAX_NPC_COUNT" \
      --random-seed "$RANDOM_SEED" \
      --method random_scenario_local \
      --output-root "$OUTPUT_ROOT" \
      --operators "$OPERATORS" \
      --oracle-checks "$ORACLE_CHECKS" \
      --llm-provider "$LLM_PROVIDER" \
      --llm-model "$LLM_MODEL"
    ;;
  *)
    echo "Unsupported CONFIG '$CONFIG'; expected global_only, local_only, or random_scenario_local." >&2
    exit 2
    ;;
esac
