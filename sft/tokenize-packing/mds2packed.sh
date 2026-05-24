#!/usr/bin/env bash
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

NPROC_PER_NODE=16
MASTER_PORT=10901
LOCAL_ROOT=./data/Nemotron-Cascade-2-SFT-tokenized-MDS/
OUT_ROOT=./data/Nemotron-Cascade-2-SFT-tokenized-MDS-packed/
MAX_PACK_LENGTH=131072
# Nemotron-Cascade-2 tokenizer has eos_token_id == pad_token_id == 11.
PAD_TOKEN_ID=11
BUFFER_SIZE=200
SHUFFLE_SEED=42

torchrun --nproc_per_node="$NPROC_PER_NODE" --master_port="$MASTER_PORT" "$SCRIPT_DIR/mds2packed.py" \
	--local-root "$LOCAL_ROOT" \
	--out-root "$OUT_ROOT" \
	--max-pack-length "$MAX_PACK_LENGTH" \
	--pad-token-id "$PAD_TOKEN_ID" \
	--buffer-size "$BUFFER_SIZE" \
	--shuffle-seed "$SHUFFLE_SEED" \
	"$@"
