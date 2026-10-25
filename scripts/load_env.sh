#!/usr/bin/env bash

# Load repository-local settings while preserving variables set by the caller.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

if [[ -f "$PROJECT_ROOT/.env" ]]; then
  # Use the Python dotenv parser without executing .env as shell code.
  if ! _vladfuzz_env_settings="$(
    "${PYTHON_BIN:-python3}" "$PROJECT_ROOT/vladfuzz_runtime/project_env.py" \
      --shell "$PROJECT_ROOT/.env"
  )"; then
    echo "Cannot load .env; activate a backend environment with python-dotenv." >&2
    return 1
  fi
  eval "$_vladfuzz_env_settings"
  unset _vladfuzz_env_settings
fi

CARLA_ROOT="${CARLA_ROOT:-${CARLA_PATH:-}}"
CARLA_PATH="${CARLA_PATH:-${CARLA_ROOT:-}}"
VLADFUZZ_MODEL_HOME="${VLADFUZZ_MODEL_HOME:-$PROJECT_ROOT/models}"
VLADFUZZ_STRICT_LOCAL_ASSETS="${VLADFUZZ_STRICT_LOCAL_ASSETS:-1}"

export PROJECT_ROOT CARLA_ROOT CARLA_PATH
export VLADFUZZ_MODEL_HOME VLADFUZZ_STRICT_LOCAL_ASSETS
