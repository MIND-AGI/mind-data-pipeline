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

INPUT_PATH=./data/*.parquet
OUT_ROOT=./data/parquet_mds
NUM_GROUPS=10
NUM_PROCESS=10
TEXT_COLUMN=text

python "$SCRIPT_DIR/parquet2mds.py" \
	--input-path "$INPUT_PATH" \
	--out-root "$OUT_ROOT" \
	--num-groups "$NUM_GROUPS" \
	--num-process "$NUM_PROCESS" \
	--text-column "$TEXT_COLUMN" \
	"$@"
