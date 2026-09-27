#!/usr/bin/env bash
set -euo pipefail

# Generate static scenario JSON files and BEV visualizations from region specs.
# Start CARLA before running this script.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

# shellcheck source=scripts/load_env.sh
source "$SCRIPT_DIR/load_env.sh"

REGIONS="${REGIONS:-configs/scenario_regions.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-static_scenarios}"
SCENARIO_ID="${SCENARIO_ID:-}"
CARLA_HOST="${CARLA_HOST:-localhost}"
CARLA_PORT="${CARLA_PORT:-2000}"
TIMEOUT="${TIMEOUT:-60}"
MAP_WARMUP_SECONDS="${MAP_WARMUP_SECONDS:-5}"
WARMUP_TICKS="${WARMUP_TICKS:-5}"

cd "$PROJECT_ROOT"

cmd=(
  "$PYTHON_BIN" scenario_construction/scenario_constructor.py
  --regions "$REGIONS"
  --output-dir "$OUTPUT_DIR"
  --host "$CARLA_HOST"
  --port "$CARLA_PORT"
  --timeout "$TIMEOUT"
  --map-warmup-seconds "$MAP_WARMUP_SECONDS"
  --warmup-ticks "$WARMUP_TICKS"
)

if [[ -n "$SCENARIO_ID" ]]; then
  cmd+=(--scenario-id "$SCENARIO_ID")
fi

echo "Generating static scenarios into $OUTPUT_DIR"
"${cmd[@]}"
