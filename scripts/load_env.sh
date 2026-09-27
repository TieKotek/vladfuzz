#!/usr/bin/env bash

# Load repository-local settings while preserving variables set by the caller.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

if [[ -f "$PROJECT_ROOT/.env" ]]; then
  set -a
  # .env is user-controlled shell syntax. Keep it private and do not commit it.
  # shellcheck disable=SC1091
  source "$PROJECT_ROOT/.env"
  set +a
fi

CARLA_ROOT="${CARLA_ROOT:-${CARLA_PATH:-}}"
CARLA_PATH="${CARLA_PATH:-${CARLA_ROOT:-}}"
VLADFUZZ_MODEL_HOME="${VLADFUZZ_MODEL_HOME:-$PROJECT_ROOT/models}"
VLADFUZZ_STRICT_LOCAL_ASSETS="${VLADFUZZ_STRICT_LOCAL_ASSETS:-1}"

export PROJECT_ROOT CARLA_ROOT CARLA_PATH
export VLADFUZZ_MODEL_HOME VLADFUZZ_STRICT_LOCAL_ASSETS
