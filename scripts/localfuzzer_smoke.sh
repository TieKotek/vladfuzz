#!/usr/bin/env bash
set -euo pipefail

# Run LocalFuzzer on one selected seed. Use this to smoke-test the LLM API and
# model backend wiring before launching long experiments. By default, this uses
# the first command in the frozen formal task.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
MODEL="${MODEL:-simlingo}"
GPU_ID="${GPU_ID:-0}"
MANIFEST="${MANIFEST:-configs/experiment_manifest.jsonl}"
MANIFEST_ID="${MANIFEST_ID:-town01_t_junction_1/seed_1}"
COMMAND_FILE="${COMMAND_FILE:-$PROJECT_ROOT/artifact_inputs/formal/test_cases/$MANIFEST_ID/command_prior.json}"
COMMAND_INDEX="${COMMAND_INDEX:-1}"
MUTATION_DEPTH="${MUTATION_DEPTH:-1}"
SUCCESS_DISTANCE="${SUCCESS_DISTANCE:-5.0}"
RANDOM_SEED="${RANDOM_SEED:-0}"

cd "$PROJECT_ROOT"

INSTRUCTION="$("$PYTHON_BIN" -c '
import json, sys
path = sys.argv[1]
index = int(sys.argv[2]) - 1
with open(path, "r", encoding="utf-8") as file:
    data = json.load(file)
commands = data.get("commands") or [data.get("instruction")]
commands = [command.strip() for command in commands if isinstance(command, str) and command.strip()]
if index < 0 or index >= len(commands):
    raise SystemExit(f"COMMAND_INDEX={index + 1} is out of range for {path}; available={len(commands)}")
print(commands[index])
' "$COMMAND_FILE" "$COMMAND_INDEX")"

echo "Running LocalFuzzer smoke test with command ${COMMAND_INDEX}: ${INSTRUCTION}"

"$PYTHON_BIN" -m vladfuzz_workflows.local_fuzzer \
  --model "$MODEL" \
  --manifest "$MANIFEST" \
  --manifest-id "$MANIFEST_ID" \
  --instruction-source manual \
  --instruction "$INSTRUCTION" \
  --mutation-depth "$MUTATION_DEPTH" \
  --success-distance "$SUCCESS_DISTANCE" \
  --gpu-id "$GPU_ID" \
  --random-seed "$RANDOM_SEED"
