#!/usr/bin/env bash
set -euo pipefail

# Run the random baseline for every frozen formal task.
# Each selected seed receives the same wall-clock budget.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
SELECTED_ROOT="${SELECTED_ROOT:-artifact_inputs/formal/test_cases}"
STATIC_ROOT="${STATIC_ROOT:-artifact_inputs/static_scenarios}"
MANIFEST="${MANIFEST:-configs/experiment_manifest.jsonl}"
MODEL="${MODEL:-simlingo}"
GPU_ID="${GPU_ID:-0}"
TIME_BUDGET_MINUTES="${TIME_BUDGET_MINUTES:-240}"
NUM_COMMANDS="${NUM_COMMANDS:-1}"
COMMAND_NAME="${COMMAND_NAME:-command_prior.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/rq2/runs/random}"
RANDOM_SEED="${RANDOM_SEED:-0}"
ORACLE_CHECKS="${ORACLE_CHECKS:-default}"
MIN_NPC_COUNT="${MIN_NPC_COUNT:-0}"
MAX_NPC_COUNT="${MAX_NPC_COUNT:-3}"
VLADFUZZ_STRICT_LOCAL_ASSETS="${VLADFUZZ_STRICT_LOCAL_ASSETS:-1}"
MANAGE_CARLA="${MANAGE_CARLA:-1}"
KEEP_DEBUG_ARTIFACTS="${KEEP_DEBUG_ARTIFACTS:-0}"
MAX_SEED_ATTEMPTS="${MAX_SEED_ATTEMPTS:-2}"
CONTINUE_ON_CARLA_FAILURE="${CONTINUE_ON_CARLA_FAILURE:-1}"
CARLA_RETRY_EXHAUSTED_STATUS=86
INFRA_RETRY_EXIT_CODES="${INFRA_RETRY_EXIT_CODES:-86 134 137 139}"

cd "$PROJECT_ROOT"

if [[ "$MANAGE_CARLA" == "1" ]]; then
  # shellcheck source=scripts/carla_manager.sh
  source "$PROJECT_ROOT/scripts/carla_manager.sh"
  trap carla_stop EXIT
fi

is_infra_retry_status() {
  local status="$1"
  for code in $INFRA_RETRY_EXIT_CODES; do
    if [[ "$status" == "$code" ]]; then
      return 0
    fi
  done
  return 1
}

run_seed_with_retries() {
  local manifest_id="$1"
  local command_file="$2"
  local attempt=1
  local status=0

  while (( attempt <= MAX_SEED_ATTEMPTS )); do
    if [[ "$MANAGE_CARLA" == "1" ]]; then
      set +e
      carla_start "random_${manifest_id}" "$attempt"
      start_status=$?
      set -e
      if (( start_status != 0 )); then
        echo "Failed to start CARLA for random seed: $manifest_id (attempt $attempt/$MAX_SEED_ATTEMPTS)"
        attempt=$((attempt + 1))
        continue
      fi
    fi

    set +e
    MANIFEST="$MANIFEST" \
    MANIFEST_ID="$manifest_id" \
    COMMAND_FILE="$command_file" \
    NUM_COMMANDS="$NUM_COMMANDS" \
    MODEL="$MODEL" \
    GPU_ID="$GPU_ID" \
    TIME_BUDGET_MINUTES="$TIME_BUDGET_MINUTES" \
    OUTPUT_ROOT="$OUTPUT_ROOT" \
    RANDOM_SEED="$RANDOM_SEED" \
    ORACLE_CHECKS="$ORACLE_CHECKS" \
    MIN_NPC_COUNT="$MIN_NPC_COUNT" \
    MAX_NPC_COUNT="$MAX_NPC_COUNT" \
    VLADFUZZ_STRICT_LOCAL_ASSETS="$VLADFUZZ_STRICT_LOCAL_ASSETS" \
    PYTHON_BIN="$PYTHON_BIN" \
      ./scripts/run_random.sh
    status=$?
    set -e

    if [[ "$MANAGE_CARLA" == "1" ]]; then
      if (( status == 0 )); then
        carla_stop
        return 0
      fi

      if ! carla_managed_process_alive; then
        echo "Random baseline seed failed after managed CARLA exited/crashed: $manifest_id (attempt $attempt/$MAX_SEED_ATTEMPTS)"
        carla_stop
        attempt=$((attempt + 1))
        continue
      fi

      if is_infra_retry_status "$status"; then
        echo "Random baseline seed failed with native infrastructure exit code $status: $manifest_id (attempt $attempt/$MAX_SEED_ATTEMPTS)"
        carla_stop
        attempt=$((attempt + 1))
        continue
      fi

      carla_stop
      echo "Random baseline failed while CARLA was still alive; treating as experiment error." >&2
      return "$status"
    fi

    return "$status"
  done

  echo "Random baseline seed exhausted CARLA retry attempts: $manifest_id" >&2
  return "$CARLA_RETRY_EXHAUSTED_STATUS"
}

echo "Building selected manifest: $MANIFEST"
"$PYTHON_BIN" scenario_construction/build_instruction_manifest.py \
  --seeds-root "$SELECTED_ROOT" \
  --static-root "$STATIC_ROOT" \
  --output "$MANIFEST" \
  --overwrite

mapfile -t MANIFEST_IDS < <("$PYTHON_BIN" -c '
import json, sys
with open(sys.argv[1], "r", encoding="utf-8") as file:
    for line in file:
        if line.strip():
            print(json.loads(line)["id"])
' "$MANIFEST")

echo "Selected seeds: ${#MANIFEST_IDS[@]}"

for manifest_id in "${MANIFEST_IDS[@]}"; do
  command_file="$PROJECT_ROOT/$SELECTED_ROOT/$manifest_id/$COMMAND_NAME"
  if [[ ! -f "$command_file" ]]; then
    echo "Skipping $manifest_id: missing $command_file"
    continue
  fi

  echo
  echo "===== Random seed: $manifest_id ====="
  set +e
  run_seed_with_retries "$manifest_id" "$command_file"
  seed_status=$?
  set -e
  if (( seed_status != 0 )); then
    if (( seed_status == CARLA_RETRY_EXHAUSTED_STATUS )) && [[ "$CONTINUE_ON_CARLA_FAILURE" == "1" ]]; then
      echo "Continuing after CARLA retry exhaustion for random seed: $manifest_id" >&2
      continue
    fi
    exit "$seed_status"
  fi
done
