#!/usr/bin/env bash
set -euo pipefail

# Run Instruction-CF for every seed under SELECTED_ROOT with managed CARLA recovery.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
SELECTED_ROOT="${SELECTED_ROOT:-artifact_inputs/formal/test_cases}"
STATIC_ROOT="${STATIC_ROOT:-artifact_inputs/static_scenarios}"
MANIFEST="${MANIFEST:-configs/experiment_manifest.jsonl}"
MODEL="${MODEL:-simlingo}"
GPU_ID="${GPU_ID:-0}"
TIME_BUDGET_MINUTES="${TIME_BUDGET_MINUTES:-240}"
SCENARIO_DURATION_FRAMES="${SCENARIO_DURATION_FRAMES:-500}"
SUCCESS_DISTANCE="${SUCCESS_DISTANCE:-5.0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/rq2/runs/instruction_counterfactual}"
RUN_ID="${RUN_ID:-${MODEL}_$(date +%Y%m%d_%H%M%S)}"
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
INFRA_RETRY_EXIT_CODES="${INFRA_RETRY_EXIT_CODES:-86 88 134 137 139}"

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

infrastructure_failure_reason() {
  case "$1" in
    86) echo "retry_exhausted" ;;
    88) echo "carla_rpc_timeout" ;;
    134) echo "native_abort" ;;
    137) echo "process_killed" ;;
    139) echo "segmentation_fault" ;;
    *) echo "carla_unavailable" ;;
  esac
}

archive_failed_attempt() {
  local manifest_id="$1"
  local attempt="$2"
  local reason="$3"
  local exit_code="$4"
  local run_dir_file="$5"

  "$PYTHON_BIN" tools/archive_vladfuzz_failure.py \
    --results-root "$OUTPUT_ROOT" \
    --run-dir-file "$run_dir_file" \
    --run-id "$RUN_ID" \
    --manifest-id "$manifest_id" \
    --attempt "$attempt" \
    --reason "$reason" \
    --exit-code "$exit_code"
}

run_seed_with_retries() {
  local manifest_id="$1"
  local attempt=1
  local status=0
  local run_dir_file=""

  while (( attempt <= MAX_SEED_ATTEMPTS )); do
    run_dir_file="${TMPDIR:-/tmp}/instruction-cf-${RUN_ID}-${manifest_id//\//_}-attempt${attempt}.run_dirs"
    rm -f "$run_dir_file"

    if [[ "$MANAGE_CARLA" == "1" ]]; then
      set +e
      carla_start "instruction_cf_${manifest_id}" "$attempt"
      start_status=$?
      set -e
      if (( start_status != 0 )); then
        echo "Failed to start CARLA for Instruction-CF seed: $manifest_id (attempt $attempt/$MAX_SEED_ATTEMPTS)"
        attempt=$((attempt + 1))
        continue
      fi
    fi

    set +e
    MANIFEST="$MANIFEST" \
    MANIFEST_ID="$manifest_id" \
    MODEL="$MODEL" \
    GPU_ID="$GPU_ID" \
    TIME_BUDGET_MINUTES="$TIME_BUDGET_MINUTES" \
    SCENARIO_DURATION_FRAMES="$SCENARIO_DURATION_FRAMES" \
    SUCCESS_DISTANCE="$SUCCESS_DISTANCE" \
    OUTPUT_ROOT="$OUTPUT_ROOT" \
    RANDOM_SEED="$RANDOM_SEED" \
    ORACLE_CHECKS="$ORACLE_CHECKS" \
    MIN_NPC_COUNT="$MIN_NPC_COUNT" \
    MAX_NPC_COUNT="$MAX_NPC_COUNT" \
    VLADFUZZ_STRICT_LOCAL_ASSETS="$VLADFUZZ_STRICT_LOCAL_ASSETS" \
    VLADFUZZ_RUN_DIR_FILE="$run_dir_file" \
    PYTHON_BIN="$PYTHON_BIN" \
      ./scripts/run_instruction_counterfactual.sh
    status=$?
    set -e

    if [[ "$MANAGE_CARLA" == "1" ]]; then
      if (( status == 0 )); then
        carla_stop
        rm -f "$run_dir_file"
        return 0
      fi

      if ! carla_managed_process_alive; then
        echo "Instruction-CF seed failed after managed CARLA exited: $manifest_id (attempt $attempt/$MAX_SEED_ATTEMPTS)"
        carla_stop
        archive_failed_attempt "$manifest_id" "$attempt" "carla_exited" "$status" "$run_dir_file"
        rm -f "$run_dir_file"
        attempt=$((attempt + 1))
        continue
      fi

      if is_infra_retry_status "$status"; then
        echo "Instruction-CF seed failed with infrastructure exit code $status: $manifest_id (attempt $attempt/$MAX_SEED_ATTEMPTS)"
        carla_stop
        archive_failed_attempt "$manifest_id" "$attempt" "$(infrastructure_failure_reason "$status")" "$status" "$run_dir_file"
        rm -f "$run_dir_file"
        attempt=$((attempt + 1))
        continue
      fi

      carla_stop
      echo "Instruction-CF failed while CARLA was alive; treating as experiment error." >&2
      return "$status"
    fi

    return "$status"
  done

  echo "Instruction-CF seed exhausted CARLA retry attempts: $manifest_id" >&2
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
  echo
  echo "===== Instruction-CF seed: $manifest_id ====="
  set +e
  run_seed_with_retries "$manifest_id"
  seed_status=$?
  set -e
  if (( seed_status != 0 )); then
    if (( seed_status == CARLA_RETRY_EXHAUSTED_STATUS )) && [[ "$CONTINUE_ON_CARLA_FAILURE" == "1" ]]; then
      echo "Continuing after CARLA retry exhaustion for Instruction-CF seed: $manifest_id" >&2
      continue
    fi
    exit "$seed_status"
  fi
done
