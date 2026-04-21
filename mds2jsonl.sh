#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="$SCRIPT_DIR/.env"

# Load .env from the project root and export all keys automatically.
if [[ -f "$ENV_FILE" ]]; then
	set -a
	# shellcheck disable=SC1090
	source "$ENV_FILE"
	set +a
else
	echo "Warning: $ENV_FILE not found. Using current shell environment only." >&2
fi

# Default cache paths can be overridden from .env.
export HF_HOME="${HF_HOME:-./hf_home}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/hf_datasets}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-./hf_cache}"

# Example environment variables for the script
NPROC_PER_NODE=16
MASTER_PORT=10900
LOCAL_ROOT=./data/fineweb-edu-sample-10BT_mds
OUT_ROOT=./data/fineweb-edu-10BT_jsonl

torchrun --nproc_per_node="$NPROC_PER_NODE" --master_port="$MASTER_PORT" "$SCRIPT_DIR/mds2jsonl.py" \
	--local-root "$LOCAL_ROOT" \
	--out-root "$OUT_ROOT" \
	"$@"
