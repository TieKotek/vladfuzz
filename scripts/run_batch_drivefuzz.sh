#!/usr/bin/env bash
set -euo pipefail

# Run DriveFuzz for every frozen formal task.
# Each selected seed receives the same wall-clock budget.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
SELECTED_ROOT="${SELECTED_ROOT:-artifact_inputs/formal/test_cases}"
STATIC_ROOT="${STATIC_ROOT:-artifact_inputs/static_scenarios}"
MANIFEST="${MANIFEST:-configs/experiment_manifest.jsonl}"
MODEL="${MODEL:-simlingo}"
GPU_ID="${GPU_ID:-0}"
DRIVEFUZZ_RESULTS_ROOT="${DRIVEFUZZ_RESULTS_ROOT:-$PROJECT_ROOT/results/rq2/runs/drivefuzz}"
RUN_ID="${RUN_ID:-${MODEL}_$(date +%Y%m%d_%H%M%S)}"
TIME_BUDGET_MINUTES="${TIME_BUDGET_MINUTES:-240}"
NUM_COMMANDS="${NUM_COMMANDS:-1}"
COMMAND_NAME="${COMMAND_NAME:-command_prior.json}"
DRIVEFUZZ_OUTPUT_PREFIX="${DRIVEFUZZ_OUTPUT_PREFIX:-out-drivefuzz-${MODEL}}"
DRIVEFUZZ_SEED_PREFIX="${DRIVEFUZZ_SEED_PREFIX:-seed-vlad}"
POST_SEED_SLEEP_SECONDS="${POST_SEED_SLEEP_SECONDS:-5}"
MANAGE_CARLA="${MANAGE_CARLA:-1}"
KEEP_DEBUG_ARTIFACTS="${KEEP_DEBUG_ARTIFACTS:-0}"
MAX_SEED_ATTEMPTS="${MAX_SEED_ATTEMPTS:-3}"
CONTINUE_ON_CARLA_FAILURE="${CONTINUE_ON_CARLA_FAILURE:-1}"
CARLA_RETRY_EXHAUSTED_STATUS=86
CARLA_PORT="${CARLA_PORT:-2000}"
INFRA_RETRY_EXIT_CODES="${INFRA_RETRY_EXIT_CODES:-86 87 88 134 137 139}"
CARLA_INFRA_RESTART_COOLDOWN_SECONDS="${CARLA_INFRA_RESTART_COOLDOWN_SECONDS:-30}"
ALLOW_WALKERS="${ALLOW_WALKERS:-1}"
DRIVEFUZZ_GDB="${DRIVEFUZZ_GDB:-0}"
VLADFUZZ_STRICT_LOCAL_ASSETS="${VLADFUZZ_STRICT_LOCAL_ASSETS:-1}"

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
    86) echo "cuda_oom" ;;
    87) echo "sensor_received_no_data" ;;
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
  local output_name="$DRIVEFUZZ_OUTPUT_PREFIX-${manifest_id//\//_}"
  local carla_log=""
  local archive_args=()

  if [[ "$MANAGE_CARLA" == "1" && "$KEEP_DEBUG_ARTIFACTS" == "1" ]]; then
    carla_log="$(carla_log_dir)/$(carla_sanitize_name "drivefuzz_${manifest_id}")_attempt${attempt}.log"
    archive_args+=(--carla-log "$carla_log")
  fi

  "$PYTHON_BIN" tools/archive_drivefuzz_failure.py \
    --results-root "$DRIVEFUZZ_RESULTS_ROOT" \
    --run-id "$RUN_ID" \
    --output-name "$output_name" \
    --manifest-id "$manifest_id" \
    --attempt "$attempt" \
    --reason "$reason" \
    --exit-code "$exit_code" \
    "${archive_args[@]}"
}

cool_down_before_retry() {
  if [[ "$CARLA_INFRA_RESTART_COOLDOWN_SECONDS" != "0" ]]; then
    echo "Cooling down ${CARLA_INFRA_RESTART_COOLDOWN_SECONDS}s before restarting CARLA."
    sleep "$CARLA_INFRA_RESTART_COOLDOWN_SECONDS"
  fi
}

run_seed_with_retries() {
  local manifest_id="$1"
  local command_file="$2"
  local attempt=1
  local status=0

  while (( attempt <= MAX_SEED_ATTEMPTS )); do
    if [[ "$MANAGE_CARLA" == "1" ]]; then
      set +e
      carla_start "drivefuzz_${manifest_id}" "$attempt"
      start_status=$?
      set -e
      if (( start_status != 0 )); then
        echo "Failed to start CARLA for DriveFuzz seed: $manifest_id (attempt $attempt/$MAX_SEED_ATTEMPTS)"
        archive_failed_attempt "$manifest_id" "$attempt" "carla_start_failed" "$start_status"
        attempt=$((attempt + 1))
        if (( attempt <= MAX_SEED_ATTEMPTS )); then
          cool_down_before_retry
        fi
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
    DRIVEFUZZ_RESULTS_ROOT="$DRIVEFUZZ_RESULTS_ROOT" \
    RUN_ID="$RUN_ID" \
    OUTPUT_DIR="$DRIVEFUZZ_OUTPUT_PREFIX-${manifest_id//\//_}" \
    SEED_DIR_PREFIX="$DRIVEFUZZ_SEED_PREFIX-${manifest_id//\//_}" \
    SIM_PORT="$CARLA_PORT" \
    ALLOW_WALKERS="$ALLOW_WALKERS" \
    DRIVEFUZZ_GDB="$DRIVEFUZZ_GDB" \
    KEEP_DEBUG_ARTIFACTS="$KEEP_DEBUG_ARTIFACTS" \
    VLADFUZZ_STRICT_LOCAL_ASSETS="$VLADFUZZ_STRICT_LOCAL_ASSETS" \
    PYTHON_BIN="$PYTHON_BIN" \
      ./scripts/run_drivefuzz.sh
    status=$?
    set -e

    if [[ "$MANAGE_CARLA" == "1" ]]; then
      if (( status == 0 )); then
        carla_stop
        return 0
      fi

      if ! carla_managed_process_alive; then
        echo "DriveFuzz seed failed after managed CARLA exited/crashed: $manifest_id (attempt $attempt/$MAX_SEED_ATTEMPTS)"
        carla_stop
        archive_failed_attempt "$manifest_id" "$attempt" "carla_exited" "$status"
        attempt=$((attempt + 1))
        if (( attempt <= MAX_SEED_ATTEMPTS )); then
          cool_down_before_retry
        fi
        continue
      fi

      if is_infra_retry_status "$status"; then
        echo "DriveFuzz seed failed with native infrastructure exit code $status: $manifest_id (attempt $attempt/$MAX_SEED_ATTEMPTS)"
        carla_stop
        archive_failed_attempt "$manifest_id" "$attempt" "$(infrastructure_failure_reason "$status")" "$status"
        attempt=$((attempt + 1))
        if (( attempt <= MAX_SEED_ATTEMPTS )); then
          cool_down_before_retry
        fi
        continue
      fi

      carla_stop
      echo "DriveFuzz failed while CARLA was still alive; treating as experiment error." >&2
      return "$status"
    fi

    return "$status"
  done

  echo "DriveFuzz seed exhausted CARLA retry attempts: $manifest_id" >&2
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
echo "DriveFuzz run id: $RUN_ID"

for manifest_id in "${MANIFEST_IDS[@]}"; do
  command_file="$PROJECT_ROOT/$SELECTED_ROOT/$manifest_id/$COMMAND_NAME"
  if [[ ! -f "$command_file" ]]; then
    echo "Skipping $manifest_id: missing $command_file"
    continue
  fi

  echo
  echo "===== DriveFuzz seed: $manifest_id ====="
  set +e
  run_seed_with_retries "$manifest_id" "$command_file"
  seed_status=$?
  set -e
  if (( seed_status != 0 )); then
    if (( seed_status == CARLA_RETRY_EXHAUSTED_STATUS )) && [[ "$CONTINUE_ON_CARLA_FAILURE" == "1" ]]; then
      echo "Continuing after CARLA retry exhaustion for DriveFuzz seed: $manifest_id" >&2
      continue
    fi
    exit "$seed_status"
  fi

  if [[ "$POST_SEED_SLEEP_SECONDS" != "0" ]]; then
    sleep "$POST_SEED_SLEEP_SECONDS"
  fi
done
