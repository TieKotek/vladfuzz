#!/usr/bin/env bash
set -euo pipefail

# Run the random baseline with a wall-clock budget on one selected manifest seed.
# By default, this script reads the first frozen formal task.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
MODEL="${MODEL:-simlingo}"
GPU_ID="${GPU_ID:-0}"
MANIFEST="${MANIFEST:-configs/experiment_manifest.jsonl}"
MANIFEST_ID="${MANIFEST_ID:-town01_t_junction_1/seed_1}"
INSTRUCTION_SOURCE="${INSTRUCTION_SOURCE:-basic}"
COMMAND_FILE="${COMMAND_FILE:-$PROJECT_ROOT/artifact_inputs/formal/test_cases/$MANIFEST_ID/command_prior.json}"
NUM_COMMANDS="${NUM_COMMANDS:-1}"
TIME_BUDGET_MINUTES="${TIME_BUDGET_MINUTES:-240}"
SCENARIO_DURATION_FRAMES="${SCENARIO_DURATION_FRAMES:-500}"
MIN_NPC_COUNT="${MIN_NPC_COUNT:-0}"
MAX_NPC_COUNT="${MAX_NPC_COUNT:-3}"
RANDOM_SEED="${RANDOM_SEED:-0}"
METHOD="${METHOD:-random}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/rq2/runs/random}"
ORACLE_CHECKS="${ORACLE_CHECKS:-default}"
cd "$PROJECT_ROOT"

# shellcheck source=scripts/huggingface_env.sh
source "$PROJECT_ROOT/scripts/huggingface_env.sh"

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
  echo "Running random baseline command ${command_id}/${NUM_COMMANDS}: ${instruction}"

  "$PYTHON_BIN" -m vladfuzz_workflows.baseline_testing \
    --standalone \
    --model "$MODEL" \
    --manifest "$MANIFEST" \
    --manifest-id "$MANIFEST_ID" \
    --instruction-source manual \
    --instruction "$instruction" \
    --time-budget-minutes "$TIME_BUDGET_MINUTES" \
    --gpu-id "$GPU_ID" \
    --random-seed "$RANDOM_SEED" \
    --method "${METHOD}_cmd${command_id}" \
    --output-root "$OUTPUT_ROOT" \
    --scenario-duration-frames "$SCENARIO_DURATION_FRAMES" \
    --min-npc-count "$MIN_NPC_COUNT" \
    --max-npc-count "$MAX_NPC_COUNT" \
    --oracle-checks "$ORACLE_CHECKS"
done
