#!/usr/bin/env bash
set -euo pipefail

# Export selected VLAD-Fuzz manifest rows into DriveFuzz seed format.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
MANIFEST="${MANIFEST:-configs/experiment_manifest.jsonl}"
MANIFEST_ID="${MANIFEST_ID:-town01_t_junction_1/seed_1}"
OUTPUT_DIR="${OUTPUT_DIR:-baselines/drivefuzz/src/seed-vlad}"
COMMAND_FILE="${COMMAND_FILE:-$PROJECT_ROOT/artifact_inputs/formal/test_cases/$MANIFEST_ID/command_prior.json}"
COMMAND_INDEX="${COMMAND_INDEX:-1}"
LIMIT="${LIMIT:-1}"

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

"$PYTHON_BIN" tools/export_seeds_to_drivefuzz.py \
  --manifest "$MANIFEST" \
  --manifest-id "$MANIFEST_ID" \
  --output-dir "$OUTPUT_DIR" \
  --instruction-source manual \
  --instruction "$INSTRUCTION" \
  --limit "$LIMIT"
