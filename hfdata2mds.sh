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
DATA_REPO=HuggingFaceFW/fineweb-edu
OUT_ROOT=./data/fineweb-edu-sample-10BT_mds
NUM_GROUPS=10
NUM_PROCESS=10
DATASET_NAME=sample-10BT
TEXT_COLUMN=text

python "$SCRIPT_DIR/hfdata2mds.py" \
	--data-repo "$DATA_REPO" \
	--out-root "$OUT_ROOT" \
	--num-groups "$NUM_GROUPS" \
	--num-process "$NUM_PROCESS" \
	--dataset-name "$DATASET_NAME" \
    --text-column "$TEXT_COLUMN" \
	"$@"
