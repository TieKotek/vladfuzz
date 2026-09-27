#!/usr/bin/env bash
set -euo pipefail

# Run DriveFuzz with the VLAD-Fuzz VLA backend and a wall-clock budget.
# By default, this script reads the first frozen formal task.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
DRIVEFUZZ_SRC="${DRIVEFUZZ_SRC:-$PROJECT_ROOT/baselines/drivefuzz/src}"
DRIVEFUZZ_RESULTS_ROOT="${DRIVEFUZZ_RESULTS_ROOT:-$PROJECT_ROOT/results/rq2/runs/drivefuzz}"
MODEL="${MODEL:-simlingo}"
RUN_ID="${RUN_ID:-${MODEL}_manual}"
GPU_ID="${GPU_ID:-0}"
MANIFEST="${MANIFEST:-configs/experiment_manifest.jsonl}"
MANIFEST_ID="${MANIFEST_ID:-town01_t_junction_1/seed_1}"
COMMAND_FILE="${COMMAND_FILE:-$PROJECT_ROOT/artifact_inputs/formal/test_cases/$MANIFEST_ID/command_prior.json}"
NUM_COMMANDS="${NUM_COMMANDS:-1}"
SEED_DIR_PREFIX="${SEED_DIR_PREFIX:-seed-vlad}"
OUTPUT_DIR="${OUTPUT_DIR:-out-drivefuzz-${MODEL}}"
TIME_BUDGET_MINUTES="${TIME_BUDGET_MINUTES:-240}"
MAX_CYCLES="${MAX_CYCLES:-999}"
MAX_MUTATIONS="${MAX_MUTATIONS:-8}"
SIM_HOST="${SIM_HOST:-localhost}"
SIM_PORT="${SIM_PORT:-2000}"
STRATEGY="${STRATEGY:-all}"
TIMEOUT="${TIMEOUT:-60}"
MAX_SIMULATION_SECONDS="${MAX_SIMULATION_SECONDS:-600}"
MAX_SIMULATION_FRAMES="${MAX_SIMULATION_FRAMES:-500}"
GOAL_DISTANCE_THRESHOLD="${GOAL_DISTANCE_THRESHOLD:-5}"
FOLLOW_SPECTATOR="${FOLLOW_SPECTATOR:-0}"
VIEW="${VIEW:-0}"
NO_LANE_CHECK="${NO_LANE_CHECK:-0}"
NO_OTHER_CHECK="${NO_OTHER_CHECK:-0}"
NO_STUCK_CHECK="${NO_STUCK_CHECK:-1}"
ENABLE_RED_CHECK="${ENABLE_RED_CHECK:-0}"
RELOAD_WORLD_AFTER_SIMULATION="${RELOAD_WORLD_AFTER_SIMULATION:-0}"
LOCK_TRAFFIC_LIGHTS_GREEN="${LOCK_TRAFFIC_LIGHTS_GREEN:-1}"
REUSE_VLA_BACKEND="${REUSE_VLA_BACKEND:-1}"
ALLOW_WALKERS="${ALLOW_WALKERS:-1}"
DRIVEFUZZ_GDB="${DRIVEFUZZ_GDB:-0}"
KEEP_DEBUG_ARTIFACTS="${KEEP_DEBUG_ARTIFACTS:-0}"
DRIVEFUZZ_EPHEMERAL_ROOT="${DRIVEFUZZ_EPHEMERAL_ROOT:-/tmp/vladfuzz-drivefuzz/$RUN_ID}"
if [[ "$KEEP_DEBUG_ARTIFACTS" == "1" ]]; then
  DRIVEFUZZ_SEED_ROOT="${DRIVEFUZZ_SEED_ROOT:-$DRIVEFUZZ_RESULTS_ROOT/seeds/$RUN_ID}"
  DRIVEFUZZ_NATIVE_DEBUG_DIR="${DRIVEFUZZ_NATIVE_DEBUG_DIR:-$DRIVEFUZZ_RESULTS_ROOT/native_debug/$RUN_ID}"
else
  DRIVEFUZZ_SEED_ROOT="${DRIVEFUZZ_SEED_ROOT:-$DRIVEFUZZ_EPHEMERAL_ROOT/seeds}"
  DRIVEFUZZ_NATIVE_DEBUG_DIR="${DRIVEFUZZ_NATIVE_DEBUG_DIR:-$DRIVEFUZZ_EPHEMERAL_ROOT/native_debug}"
fi
# CUDA diagnostics are infrastructure-only and can trigger native crashes in
# some backend environments. Keep them disabled by default for all DriveFuzz
# runs; set VLAD_CUDA_DIAGNOSTICS=1 explicitly when diagnosing GPU memory.
VLAD_CUDA_DIAGNOSTICS="${VLAD_CUDA_DIAGNOSTICS:-0}"
VLAD_CUDA_DIAG_STEP_INTERVAL="${VLAD_CUDA_DIAG_STEP_INTERVAL:-100}"

# shellcheck source=scripts/huggingface_env.sh
source "$PROJECT_ROOT/scripts/huggingface_env.sh"

cleanup_ephemeral_artifacts() {
  if [[ "$KEEP_DEBUG_ARTIFACTS" != "1" ]]; then
    rm -rf "$DRIVEFUZZ_EPHEMERAL_ROOT"
  fi
}
trap cleanup_ephemeral_artifacts EXIT

cd "$DRIVEFUZZ_SRC"
mkdir -p "$DRIVEFUZZ_RESULTS_ROOT/runs/$RUN_ID" "$DRIVEFUZZ_SEED_ROOT" "$DRIVEFUZZ_NATIVE_DEBUG_DIR"
export PYTHONFAULTHANDLER="${PYTHONFAULTHANDLER:-1}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
# LMDrive and BEVDriver use PyTorch 2.0, which aborts CUDA initialization when
# expandable_segments is present. SimLingo's newer allocator supports it.
case "$MODEL" in
  "lmdrive"|"bevdriver")
    if [[ "${PYTORCH_CUDA_ALLOC_CONF:-}" == *expandable_segments* ]]; then
      echo "Disabling unsupported expandable_segments allocator for $MODEL."
      unset PYTORCH_CUDA_ALLOC_CONF
    fi
    ;;
  *)
    export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
    ;;
esac
if [[ "$KEEP_DEBUG_ARTIFACTS" == "1" || "$DRIVEFUZZ_GDB" == "1" ]]; then
  ulimit -c unlimited 2>/dev/null || true
else
  ulimit -c 0 2>/dev/null || true
fi

EXTRA_ARGS=()
if [[ "$FOLLOW_SPECTATOR" == "1" ]]; then
  EXTRA_ARGS+=(--follow-spectator --view "$VIEW")
fi
if [[ "$NO_LANE_CHECK" == "1" ]]; then
  EXTRA_ARGS+=(--no-lane-check)
fi
if [[ "$NO_OTHER_CHECK" == "1" ]]; then
  EXTRA_ARGS+=(--no-other-check)
fi
if [[ "$NO_STUCK_CHECK" == "1" ]]; then
  EXTRA_ARGS+=(--no-stuck-check)
fi
if [[ "$ENABLE_RED_CHECK" == "1" ]]; then
  EXTRA_ARGS+=(--enable-red-check)
else
  EXTRA_ARGS+=(--no-red-check)
fi
if [[ "$RELOAD_WORLD_AFTER_SIMULATION" == "1" ]]; then
  EXTRA_ARGS+=(--reload-world-after-simulation)
fi
if [[ "$LOCK_TRAFFIC_LIGHTS_GREEN" == "1" ]]; then
  EXTRA_ARGS+=(--lock-traffic-lights-green)
fi
if [[ "$REUSE_VLA_BACKEND" == "0" ]]; then
  EXTRA_ARGS+=(--no-reuse-vla-backend)
fi
if [[ "$ALLOW_WALKERS" != "1" ]]; then
  EXTRA_ARGS+=(--no-walkers)
fi

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
  seed_dir="$DRIVEFUZZ_SEED_ROOT/${SEED_DIR_PREFIX}-cmd${command_id}"
  output_dir="$DRIVEFUZZ_RESULTS_ROOT/runs/$RUN_ID/${OUTPUT_DIR}-cmd${command_id}"
  cuda_diag_dir="$output_dir/cuda_diagnostics"
  echo "Running DriveFuzz command ${command_id}/${NUM_COMMANDS}: ${instruction}"

  cd "$PROJECT_ROOT"
  "$PYTHON_BIN" tools/export_seeds_to_drivefuzz.py \
    --manifest "$MANIFEST" \
    --manifest-id "$MANIFEST_ID" \
    --output-dir "$seed_dir" \
    --instruction-source manual \
    --instruction "$instruction" \
    --limit 1

  cd "$DRIVEFUZZ_SRC"
  rm -rf "$output_dir"

  FUZZER_CMD=(
    "$PYTHON_BIN" -X faulthandler fuzzer.py
    -o "$output_dir" \
    -s "$seed_dir" \
    -c "$MAX_CYCLES" \
    -m "$MAX_MUTATIONS" \
    -t vla \
    --project-root "$PROJECT_ROOT" \
    --vla-model "$MODEL" \
    --vla-gpu-id "$GPU_ID" \
    --vla-instruction-map "$seed_dir/mapping.jsonl" \
    --sim-host "$SIM_HOST" \
    --sim-port "$SIM_PORT" \
    --strategy "$STRATEGY" \
    --timeout "$TIMEOUT" \
    --max-simulation-seconds "$MAX_SIMULATION_SECONDS" \
    --max-simulation-frames "$MAX_SIMULATION_FRAMES" \
    --goal-distance-threshold "$GOAL_DISTANCE_THRESHOLD" \
    --time-budget-minutes "$TIME_BUDGET_MINUTES" \
    --repeat-seeds-until-budget \
    --cuda-diag-dir "$cuda_diag_dir" \
    "${EXTRA_ARGS[@]}"
  )

  native_log="$DRIVEFUZZ_NATIVE_DEBUG_DIR/${OUTPUT_DIR}-cmd${command_id}_native.log"
  if [[ "$KEEP_DEBUG_ARTIFACTS" == "1" ]]; then
    echo "DriveFuzz native diagnostics log: $native_log"
  fi
  printf 'Command:' > "$native_log"
  printf ' %q' "${FUZZER_CMD[@]}" >> "$native_log"
  printf '\n\n' >> "$native_log"

  if [[ "$DRIVEFUZZ_GDB" == "1" ]]; then
    if command -v gdb >/dev/null 2>&1; then
      VLAD_CUDA_DIAGNOSTICS="$VLAD_CUDA_DIAGNOSTICS" VLAD_CUDA_DIAG_STEP_INTERVAL="$VLAD_CUDA_DIAG_STEP_INTERVAL" PROJECT_ROOT="$PROJECT_ROOT" gdb -q -batch \
        -ex "set pagination off" \
        -ex "run" \
        -ex "thread apply all bt full" \
        -ex "info sharedlibrary" \
        --args "${FUZZER_CMD[@]}" 2>&1 | tee -a "$native_log"
      status=${PIPESTATUS[0]}
      if grep -Eq "SIGSEGV|Segmentation fault" "$native_log"; then
        exit 139
      fi
      if grep -Eq "SIGABRT|Aborted" "$native_log"; then
        exit 134
      fi
      if (( status != 0 )); then
        exit "$status"
      fi
      continue
    else
      echo "DRIVEFUZZ_GDB=1 requested, but gdb is not available." | tee -a "$native_log" >&2
      exit 127
    fi
  fi

  VLAD_CUDA_DIAGNOSTICS="$VLAD_CUDA_DIAGNOSTICS" VLAD_CUDA_DIAG_STEP_INTERVAL="$VLAD_CUDA_DIAG_STEP_INTERVAL" PROJECT_ROOT="$PROJECT_ROOT" "${FUZZER_CMD[@]}" 2>&1 | tee -a "$native_log"
  status=${PIPESTATUS[0]}
  if (( status != 0 )); then
    exit "$status"
  fi
done
