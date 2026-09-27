#!/usr/bin/env bash

# Shared CARLA process management helpers for long-running experiments.
# Source this file from batch scripts, or call it directly with start/stop/restart.
#
# Required configuration:
#   CARLA_ROOT=/path/to/CARLA_0.9.15
# or, equivalently:
#   CARLA_PATH=/path/to/CARLA_0.9.15
#
# Optional:
#   CARLA_RUN_SCRIPT=/path/to/custom-launch-script.sh
#   CARLA_EXECUTABLE=/path/to/CarlaUE4.sh
#   CARLA_LAUNCH_ARGS="-RenderOffScreen -nosound -vulkan"
#   CARLA_ISOLATED=1 CARLA_PORT=2010 CARLA_TM_PORT=8010

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# shellcheck source=scripts/load_env.sh
source "$SCRIPT_DIR/load_env.sh"

carla_isolated() {
  [[ "${CARLA_ISOLATED:-0}" == "1" ]]
}

carla_log_dir() {
  if [[ -n "${CARLA_LOG_DIR:-}" ]]; then
    echo "$CARLA_LOG_DIR"
  elif [[ "${KEEP_DEBUG_ARTIFACTS:-0}" == "1" ]]; then
    if carla_isolated; then
      echo "$PROJECT_ROOT/results/carla_logs/${RUN_ID:-manual}_rpc${CARLA_PORT:-2000}"
    else
      echo "$PROJECT_ROOT/results/carla_logs/${RUN_ID:-manual}"
    fi
  elif carla_isolated; then
    echo "/tmp/vladfuzz-carla/${RUN_ID:-manual}_rpc${CARLA_PORT:-2000}"
  else
    echo "/tmp/vladfuzz-carla/${RUN_ID:-manual}"
  fi
}

carla_pid_file() {
  echo "$(carla_log_dir)/carla.pid"
}

carla_cleanup_ephemeral_logs() {
  if [[ "${KEEP_DEBUG_ARTIFACTS:-0}" != "1" && -z "${CARLA_LOG_DIR:-}" ]]; then
    rm -rf "$(carla_log_dir)"
  fi
}

carla_sanitize_name() {
  local value="${1:-carla}"
  value="${value//\//_}"
  value="${value// /_}"
  value="${value//:/_}"
  echo "$value"
}

carla_pid_alive() {
  local pid="${1:-}"
  [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null
}

carla_managed_pid() {
  local pid_file
  pid_file="$(carla_pid_file)"
  if [[ -f "$pid_file" ]]; then
    cat "$pid_file"
  fi
}

carla_managed_process_alive() {
  local pid
  pid="$(carla_managed_pid)"
  carla_pid_alive "$pid"
}

carla_launch_args() {
  local mode="${VLADFUZZ_CARLA_MODE:-offscreen}"
  local args="${CARLA_LAUNCH_ARGS:-}"
  if [[ -z "$args" ]]; then
    case "$mode" in
      offscreen)
        args="-RenderOffScreen -nosound -vulkan"
        ;;
      windowed)
        args="-vulkan"
        ;;
      *)
        echo "Unsupported VLADFUZZ_CARLA_MODE: $mode (expected offscreen or windowed)" >&2
        return 2
        ;;
    esac
  fi
  if carla_isolated; then
    local rpc_port="${CARLA_PORT:-2000}"
    local streaming_port="${CARLA_STREAMING_PORT:-$((rpc_port + 1))}"
    local secondary_port="${CARLA_SECONDARY_PORT:-$((rpc_port + 2))}"
    args+=" -carla-rpc-port=${rpc_port}"
    args+=" -carla-streaming-port=${streaming_port}"
    args+=" -carla-secondary-port=${secondary_port}"
  fi
  if [[ -n "${CARLA_GRAPHICS_ADAPTER:-}" ]]; then
    args+=" -graphicsadapter=${CARLA_GRAPHICS_ADAPTER}"
  fi
  echo "$args"
}

carla_root() {
  local root="${CARLA_ROOT:-${CARLA_PATH:-}}"
  if [[ -z "$root" ]]; then
    echo "CARLA_ROOT or CARLA_PATH must be set to the CARLA installation directory." >&2
    return 1
  fi
  echo "$root"
}

carla_executable() {
  local root executable
  root="$(carla_root)" || return 1
  executable="${CARLA_EXECUTABLE:-$root/CarlaUE4.sh}"
  if [[ ! -f "$executable" ]]; then
    echo "CARLA executable not found: $executable" >&2
    return 1
  fi
  echo "$executable"
}

carla_configure_headless_env() {
  if [[ "${VLADFUZZ_CARLA_MODE:-offscreen}" == "windowed" ]]; then
    return 0
  fi
  export VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-/usr/share/vulkan/icd.d/nvidia_icd.json}"
  export SDL_VIDEODRIVER="${SDL_VIDEODRIVER:-dummy}"
  export SDL_AUDIODRIVER="${SDL_AUDIODRIVER:-dummy}"
  export DISPLAY="${DISPLAY:-}"
  if carla_isolated && [[ -z "${XDG_RUNTIME_DIR:-}" ]]; then
    export XDG_RUNTIME_DIR="/tmp/runtime-carla-${RUN_ID:-manual}-rpc${CARLA_PORT:-2000}"
  else
    export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/tmp/runtime-carla}"
  fi
  mkdir -p "$XDG_RUNTIME_DIR"
  chmod 700 "$XDG_RUNTIME_DIR" 2>/dev/null || true
}

carla_stop() {
  local pid
  pid="$(carla_managed_pid)"
  if ! carla_pid_alive "$pid"; then
    rm -f "$(carla_pid_file)"
    carla_stop_existing_processes
    carla_cleanup_ephemeral_logs
    return 0
  fi

  echo "Stopping managed CARLA process group: $pid"
  kill -TERM "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true

  local timeout="${CARLA_SHUTDOWN_TIMEOUT_SECONDS:-20}"
  local elapsed=0
  while carla_pid_alive "$pid" && (( elapsed < timeout )); do
    sleep 1
    elapsed=$((elapsed + 1))
  done

  if carla_pid_alive "$pid"; then
    echo "Managed CARLA did not stop within ${timeout}s; forcing kill."
    kill -KILL "-$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null || true
  fi

  wait "$pid" 2>/dev/null || true
  rm -f "$(carla_pid_file)"
  # The launch wrapper and CarlaUE4 process can have different PIDs. Ensure
  # no renderer child survives into the infrastructure cooldown interval.
  carla_stop_existing_processes
  carla_cleanup_ephemeral_logs
}

carla_stop_isolated_processes() {
  local port="${CARLA_PORT:-2000}"
  local pattern="^.*/CarlaUE4-Linux-Shipping .* -carla-rpc-port=${port}([[:space:]]|$)"
  local pids
  pids="$(pgrep -f "$pattern" 2>/dev/null || true)"
  if [[ -z "$pids" ]]; then
    return 0
  fi

  echo "Stopping isolated CARLA process for RPC port ${port}: $pids"
  kill -TERM $pids 2>/dev/null || true
  for pid in $pids; do
    wait "$pid" 2>/dev/null || true
  done

  local remaining=""
  for _ in $(seq 1 30); do
    remaining=""
    for pid in $pids; do
      if kill -0 "$pid" 2>/dev/null; then
        remaining+=" $pid"
      fi
    done
    [[ -z "$remaining" ]] && return 0
    sleep 0.1
  done

  kill -KILL $remaining 2>/dev/null || true
}

carla_stop_existing_processes() {
  if carla_isolated; then
    carla_stop_isolated_processes
    return 0
  fi
  if [[ "${CARLA_STOP_EXISTING:-1}" != "1" ]]; then
    return 0
  fi

  local pattern='CarlaUE4-Linux-Shipping|CarlaUE4.sh'
  local pids
  pids="$(pgrep -f "$pattern" 2>/dev/null || true)"
  if [[ -z "$pids" ]]; then
    return 0
  fi

  echo "Stopping existing CARLA processes before managed start: $pids"
  pkill -TERM -f "$pattern" 2>/dev/null || true
  sleep 3
  pkill -KILL -f "$pattern" 2>/dev/null || true
}

carla_wait_ready() {
  if [[ "${CARLA_SKIP_READY_CHECK:-0}" == "1" ]]; then
    return 0
  fi

  local host="${CARLA_HOST:-localhost}"
  local port="${CARLA_PORT:-2000}"
  local timeout="${CARLA_STARTUP_TIMEOUT_SECONDS:-300}"
  local python_bin="${CARLA_PYTHON_BIN:-${PYTHON_BIN:-python}}"
  local start_time=$SECONDS

  while (( SECONDS - start_time < timeout )); do
    if "$python_bin" - "$host" "$port" >/dev/null 2>&1 <<'PY'
import sys
import carla

host = sys.argv[1]
port = int(sys.argv[2])
client = carla.Client(host, port)
client.set_timeout(3.0)
client.get_world()
PY
    then
      echo "CARLA is ready on ${host}:${port}."
      return 0
    fi

    if ! carla_managed_process_alive; then
      echo "Managed CARLA process exited before becoming ready." >&2
      return 1
    fi

    sleep 2
  done

  echo "Timed out waiting ${timeout}s for CARLA on ${host}:${port}." >&2
  return 1
}

carla_start() {
  local seed_name="${1:-manual}"
  local attempt="${2:-1}"
  local root
  root="$(carla_root)" || return 1
  local run_script="${CARLA_RUN_SCRIPT:-}"
  local executable=""
  local log_dir
  log_dir="$(carla_log_dir)"

  carla_stop
  carla_stop_existing_processes
  mkdir -p "$log_dir"

  if carla_isolated; then
    executable="$(carla_executable)" || return 1
    run_script=""
  elif [[ -n "$run_script" ]]; then
    if [[ ! -f "$run_script" ]]; then
      echo "CARLA run script not found: $run_script" >&2
      return 1
    fi
  else
    executable="$(carla_executable)" || return 1
  fi

  local safe_seed log_file
  safe_seed="$(carla_sanitize_name "$seed_name")"
  log_file="$log_dir/${safe_seed}_attempt${attempt}.log"

  echo "Starting CARLA for ${seed_name} attempt ${attempt}; log: $log_file"
  (
    cd "$root"
    carla_configure_headless_env
    if [[ -n "$run_script" ]]; then
      exec setsid bash "$run_script"
    fi
    # shellcheck disable=SC2086
    local launch_args
    launch_args="$(carla_launch_args)" || exit $?
    exec setsid bash "$executable" $launch_args
  ) >"$log_file" 2>&1 &

  local pid=$!
  echo "$pid" >"$(carla_pid_file)"

  if ! carla_wait_ready; then
    echo "CARLA failed to start. Last log lines:" >&2
    tail -80 "$log_file" >&2 || true
    carla_stop
    return 1
  fi
}

carla_restart() {
  local seed_name="${1:-manual}"
  local attempt="${2:-1}"
  carla_stop
  carla_start "$seed_name" "$attempt"
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  action="${1:-}"
  shift || true
  case "$action" in
    start)
      carla_start "$@"
      ;;
    stop)
      carla_stop
      ;;
    restart)
      carla_restart "$@"
      ;;
    wait-ready)
      carla_wait_ready
      ;;
    *)
      echo "Usage: $0 {start|stop|restart|wait-ready} [seed-name] [attempt]" >&2
      exit 2
      ;;
  esac
fi
