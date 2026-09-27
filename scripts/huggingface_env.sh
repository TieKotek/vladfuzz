#!/usr/bin/env bash
# Shared Hugging Face cache and local-asset settings.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# shellcheck source=scripts/load_env.sh
source "$SCRIPT_DIR/load_env.sh"

HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"
HF_HOME="${HF_HOME:-$PROJECT_ROOT/.cache/huggingface}"
HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
VLADFUZZ_BERT_MODEL="${VLADFUZZ_BERT_MODEL:-$VLADFUZZ_MODEL_HOME/shared/bert-base-uncased}"
HF_HUB_DISABLE_TELEMETRY="${HF_HUB_DISABLE_TELEMETRY:-1}"

export HF_ENDPOINT HF_HOME HUGGINGFACE_HUB_CACHE TRANSFORMERS_CACHE
export VLADFUZZ_MODEL_HOME VLADFUZZ_BERT_MODEL VLADFUZZ_STRICT_LOCAL_ASSETS
export HF_HUB_DISABLE_TELEMETRY
mkdir -p "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$TRANSFORMERS_CACHE" "$VLADFUZZ_MODEL_HOME"
echo "Using Hugging Face endpoint: $HF_ENDPOINT"
echo "Using Hugging Face cache: $HF_HOME"
echo "Using local BERT assets: $VLADFUZZ_BERT_MODEL"
