#!/usr/bin/env bash
# Verify a Stage-2 packed MDS dataset and print detokenized samples.
#   bash verify_packed.sh                       # defaults to ./data/sft-packed-mds
#   bash verify_packed.sh --data-root <dir> --show 5 --segments-per-pack 2
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="$SCRIPT_DIR/.env"

if [[ -f "$ENV_FILE" ]]; then
	set -a
	# shellcheck disable=SC1090
	source "$ENV_FILE"
	set +a
else
	echo "Warning: $ENV_FILE not found. Using current shell environment only." >&2
fi

export HF_HOME="${HF_HOME:-./hf_home}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/hf_datasets}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-./hf_cache}"

DATA_ROOT=./data/sft-packed-mds
TOKENIZER=nvidia/Nemotron-Cascade-2-30B-A3B
PAD_TOKEN_ID=11

python "$SCRIPT_DIR/verify_packed.py" \
	--data-root "$DATA_ROOT" \
	--tokenizer "$TOKENIZER" \
	--pad-token-id "$PAD_TOKEN_ID" \
	--trust-remote-code \
	"$@"
