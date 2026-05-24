#!/usr/bin/env bash
# Single-node:  bash sft2mds.sh
# Multi-node:   bash sft2mds.sh --num-nodes N --node-rank R  # on each machine R=0..N-1
#               bash sft2mds.sh --num-nodes N --merge-only   # once on any machine after all done
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

# Defaults targeting Nemotron-Cascade-2 SFT data + tokenizer.
# Override any of these via CLI flags (forwarded by "$@").
INPUT_PATH=./data/Nemotron-Cascade-2-SFT-Data/
OUT_ROOT=./data/Nemotron-Cascade-2-SFT-tokenized-MDS/
TOKENIZER=nvidia/Nemotron-Cascade-2-30B-A3B
MAX_DOC_LENGTH=131072
NUM_PROCESS=48
# Nemotron chat template defaults to truncate_history_thinking=True (drops
# <think> blocks in non-last assistant turns). We force it False so thinking
# is kept in every turn — this also keeps the prefix-based assistant-mask
# reconstruction aligned with the full tokenize.
TEMPLATE_KWARGS='{"truncate_history_thinking": false}'

python "$SCRIPT_DIR/sft2mds.py" \
	--input-path "$INPUT_PATH" \
	--out-root "$OUT_ROOT" \
	--tokenizer "$TOKENIZER" \
	--max-doc-length "$MAX_DOC_LENGTH" \
	--num-process "$NUM_PROCESS" \
	--trust-remote-code \
	--template-kwargs "$TEMPLATE_KWARGS" \
	"$@"
