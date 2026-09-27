#!/usr/bin/env bash
set -euo pipefail

# Run VLAD-Fuzz with a wall-clock budget on one selected manifest seed.
# By default, this script reads command_prior.json under
# artifact_inputs/formal/test_cases/<MANIFEST_ID>/ and runs the first commands
# as independent runs.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
SELECTED_ROOT="${SELECTED_ROOT:-artifact_inputs/formal/test_cases}"
STATIC_ROOT="${STATIC_ROOT:-artifact_inputs/static_scenarios}"
MODEL="${MODEL:-simlingo}"
GPU_ID="${GPU_ID:-0}"
MANIFEST="${MANIFEST:-configs/experiment_manifest.jsonl}"
MANIFEST_ID="${MANIFEST_ID:-}"
COMMAND_NAME="${COMMAND_NAME:-command_prior.json}"
COMMAND_FILE="${COMMAND_FILE:-}"
NUM_COMMANDS="${NUM_COMMANDS:-1}"
TIME_BUDGET_MINUTES="${TIME_BUDGET_MINUTES:-240}"
POP_SIZE="${POP_SIZE:-8}"
NGEN="${NGEN:-999}"
CXPB="${CXPB:-0.7}"
MUTPB="${MUTPB:-1.0}"
MUTATION_DEPTH="${MUTATION_DEPTH:-2}"
SUCCESS_DISTANCE="${SUCCESS_DISTANCE:-5.0}"
SCENARIO_DURATION_FRAMES="${SCENARIO_DURATION_FRAMES:-500}"
MIN_NPC_COUNT="${MIN_NPC_COUNT:-0}"
MAX_NPC_COUNT="${MAX_NPC_COUNT:-3}"
RANDOM_SEED="${RANDOM_SEED:-0}"
METHOD="${METHOD:-vlad_fuzz}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/rq2/runs/vladfuzz}"
OPERATORS="${OPERATORS:-all}"
ORACLE_CHECKS="${ORACLE_CHECKS:-default}"
LLM_PROVIDER="${LLM_PROVIDER:-deepseek}"
LLM_MODEL="${LLM_MODEL:-deepseek-v4-flash}"
cd "$PROJECT_ROOT"

# shellcheck source=scripts/huggingface_env.sh
source "$PROJECT_ROOT/scripts/huggingface_env.sh"

if [[ ! -f "$MANIFEST" ]]; then
  echo "Building selected manifest: $MANIFEST"
  "$PYTHON_BIN" scenario_construction/build_instruction_manifest.py \
    --seeds-root "$SELECTED_ROOT" \
    --static-root "$STATIC_ROOT" \
    --output "$MANIFEST" \
    --overwrite
fi

if [[ -z "$MANIFEST_ID" ]]; then
  MANIFEST_ID="$("$PYTHON_BIN" -c '
import json, sys
with open(sys.argv[1], "r", encoding="utf-8") as file:
    for line in file:
        if line.strip():
            print(json.loads(line)["id"])
            break
' "$MANIFEST")"
fi

if [[ -z "$MANIFEST_ID" ]]; then
  echo "No manifest rows found in $MANIFEST" >&2
  exit 1
fi

if [[ -z "$COMMAND_FILE" ]]; then
  COMMAND_FILE="$PROJECT_ROOT/$SELECTED_ROOT/$MANIFEST_ID/$COMMAND_NAME"
fi

if [[ ! -f "$COMMAND_FILE" ]]; then
  echo "Missing command file: $COMMAND_FILE" >&2
  exit 1
fi

echo "VLAD-Fuzz seed: $MANIFEST_ID"
echo "Command file: $COMMAND_FILE"
echo "Output root: $OUTPUT_ROOT"

mapfile -t COMMANDS < <("$PYTHON_BIN" -c '
import json, sys
path = sys.argv[1]
limit = int(sys.argv[2])
with open(path, "r", encoding="utf-8") as file:
    data = json.load(file)
commands = data.get("commands") or [data.get("instruction")]
commands = [command.strip() for command in commands if isinstance(command, str) and command.strip()]
if len(commands) < limit:
    raise SystemExit(f"{path} has only {len(commands)} command(s), but NUM_COMMANDS={limit}")
for command in commands[:limit]:
    print(command)
' "$COMMAND_FILE" "$NUM_COMMANDS")

for index in "${!COMMANDS[@]}"; do
  command_id=$((index + 1))
  instruction="${COMMANDS[$index]}"
  echo "Running VLAD-Fuzz command ${command_id}/${NUM_COMMANDS}: ${instruction}"

  "$PYTHON_BIN" -m vladfuzz_workflows.nsga_optimization \
    --model "$MODEL" \
    --manifest "$MANIFEST" \
    --manifest-id "$MANIFEST_ID" \
    --instruction-source manual \
    --instruction "$instruction" \
    --time-budget-minutes "$TIME_BUDGET_MINUTES" \
    --pop-size "$POP_SIZE" \
    --ngen "$NGEN" \
    --cxpb "$CXPB" \
    --mutpb "$MUTPB" \
    --mutation-depth "$MUTATION_DEPTH" \
    --success-distance "$SUCCESS_DISTANCE" \
    --scenario-duration-frames "$SCENARIO_DURATION_FRAMES" \
    --min-npc-count "$MIN_NPC_COUNT" \
    --max-npc-count "$MAX_NPC_COUNT" \
    --gpu-id "$GPU_ID" \
    --random-seed "$RANDOM_SEED" \
    --method "${METHOD}_cmd${command_id}" \
    --output-root "$OUTPUT_ROOT" \
    --operators "$OPERATORS" \
    --oracle-checks "$ORACLE_CHECKS" \
    --llm-provider "$LLM_PROVIDER" \
    --llm-model "$LLM_MODEL"
done
