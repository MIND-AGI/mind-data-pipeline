## sft-samples

Two-stage pipeline: JSONL SFT chat data → tokenized MDS → fixed-length
**packed MDS** ready for training.

```
JSONL chat (OpenAI messages format)
   │
   │ stage 1: sft2mds.sh  (multiprocessing Pool)
   │   - apply chat template
   │   - assistant-only loss mask
   │   - drop samples longer than --max-doc-length
   ▼
tokenized MDS  ({input_ids, labels})
   │
   │ stage 2: mds2packed.sh  (torchrun)
   │   - StreamingDataset shuffle + auto-shard across ranks
   │   - greedy first-fit buffer packing into --max-pack-length packs
   │   - per-rank shard_{N}/, rank-0 merges into top-level index.json
   ▼
packed MDS  ({input_ids, labels, cu_seqlens, attn_mask})
```

### Layout

```
sft-samples/
├── formatting_utils.py    # chat template + assistant-only loss mask
├── packing_utils.py       # GreedyPacker (buffer-based first-fit)
├── sft2mds.py / .sh       # stage 1
├── mds2packed.py / .sh    # stage 2
├── test_packing.py        # pure-python invariant tests
├── requirements.txt
└── README.md
```

### Install

```bash
pip install -r requirements.txt
```

Optional `.env` (auto-sourced by both `*.sh` wrappers; ignored by git):

```dotenv
HF_HOME=./hf_home
HF_DATASETS_CACHE=./hf_home/hf_datasets
HUGGINGFACE_HUB_CACHE=./hf_cache
HUGGING_FACE_HUB_TOKEN=...   # if pulling tokenizer from HF
```

### Stage 1: JSONL chat → tokenized MDS

```bash
bash sft2mds.sh    # defaults: Nemotron-Cascade-2-SFT-Data + Nemotron-Cascade-2-30B-A3B
```

Or override anything on the CLI (the `.sh` forwards `"$@"` to the Python):

```bash
bash sft2mds.sh \
    --input-path ./data/Nemotron-Cascade-2-SFT-Data \
    --out-root   ./data/nemotron_cascade2_sft_mds \
    --tokenizer  nvidia/Nemotron-Cascade-2-30B-A3B \
    --max-doc-length 2048 \
    --num-process 16 \
    --trust-remote-code \
    --template-kwargs '{"truncate_history_thinking": false}'
```

**Input**: JSONL, one row per line. Each row is `{"messages": [...], "tools": [...optional...]}`
in OpenAI chat format (`role`, `content`, optional `tool_calls`). Extra
top-level fields (e.g. `domain`, `source`, `generator` in Nemotron-Cascade-2)
are ignored.

`--input-path` accepts a file, a glob, or a directory (recursively scanned
for `*.jsonl`).

**Output**: MDS shards under `--out-root` with columns
`{input_ids: ndarray:int32, labels: ndarray:int32}`. `labels` already has the
standard next-token-prediction shift applied (drop last input, drop first
label), with non-assistant positions set to `-100`.

**Filtering**: samples whose tokenized length exceeds `--max-doc-length` are
dropped. Per-worker counters (`written / oversize / no_assistant /
tokenize_error / too_short`) are logged at the end.

**Sharding**: file-level when `len(files) >= --num-process`, else line-stride
(every worker sees every file; line `N` is handled by worker `N % num_workers`).

**`--template-kwargs`** (JSON dict, default `{}`): forwarded to **every**
`apply_chat_template` call — both the full-sequence tokenize and each prefix
re-tokenize used to recover the assistant mask. For the Nemotron template,
default in `sft2mds.sh` is `{"truncate_history_thinking": false}` so `<think>`
blocks in earlier assistant turns are preserved. Unknown kwargs are silently
ignored by Jinja templates that don't reference them.

#### Multi-node Stage 1

Stage 1 supports running on multiple machines that share `--out-root` over
NFS, coordinated purely via CLI args (no ssh / slurm / torch.distributed).

```bash
# On node 0
bash sft2mds.sh --num-nodes 2 --node-rank 0 --num-process 32

# On node 1
bash sft2mds.sh --num-nodes 2 --node-rank 1 --num-process 32

# After ALL nodes finish, run once on any node to merge per-shard indexes
bash sft2mds.sh --out-root <same-out-root> --num-nodes 2 --merge-only
```

The global worker set has `G = num_nodes * num_process` workers. Total input
bytes are partitioned into `G` contiguous chunks; node `R` owns global ids
`[R*num_process, (R+1)*num_process)`. Each shard directory is named by its
**global** id (`shard_0` ... `shard_{G-1}`) so multiple nodes writing into
the same `--out-root` never collide.

Notes:
- `--num-nodes 1` (default) is unchanged single-node behavior — including the
  automatic `merge_index` at the end. With `--num-nodes > 1`, the auto-merge
  is skipped; you must run `--merge-only` once after all nodes finish.
- In multi-node mode the `--count-lines` pre-scan counts only the lines in
  this node's assigned byte ranges (its ~1/num_nodes slice), so each node's
  `tqdm` shows a correct per-node ETA. Pass `--no-count-lines` to skip it.
- Same `--num-nodes` / `--num-process` must be used on every node and on the
  final `--merge-only` call (the global id space depends on it).
- Each `--node-rank` value must be used **exactly once** — duplicates would
  overwrite shard directories.

### Stage 2: tokenized MDS → packed MDS

```bash
bash mds2packed.sh    # defaults read from sft2mds.sh's output
```

Or override:

```bash
bash mds2packed.sh \
    --local-root ./data/nemotron_cascade2_sft_mds \
    --out-root   ./data/nemotron_cascade2_sft_packed_mds \
    --max-pack-length 32768 \
    --pad-token-id 11 \
    --buffer-size 200 \
    --shuffle-seed 42
```

Launched via `torchrun --nproc_per_node=N` (default 16; edit `NPROC_PER_NODE`
in the `.sh`). Each rank opens `StreamingDataset(shuffle=True)` — the
streaming library auto-shards across ranks — feeds samples into a per-rank
`GreedyPacker`, and writes packs into its own `shard_{rank}/` directory.
After the final `dist.barrier()`, rank 0 merges the per-rank indexes into a
single top-level `index.json` so the output directory is one addressable MDS
dataset.

The pipeline currently **consumes the input exhaustively** — every input
sample contributes to exactly one pack, and the final under-filled pack (if
any) is padded at the tail. There is no global cap on the number of output
packs. To bound the output, cap stage 1 input volume; to grow it, iterate
stage 2 over multiple epochs (not implemented — `StreamingDataset` is
iterated once).

**Output MDS columns**:

```
input_ids  : ndarray:int32, length == max_pack_length
labels     : ndarray:int32, length == max_pack_length  (padding = -100)
cu_seqlens : ndarray:int32, starts at 0, ends at max_pack_length
attn_mask  : ndarray:int8,  1 = real token, 0 = padding (tail only)
```

`attn_mask` uses `int8` (1 byte/position) since one bit of signal is enough —
roughly 4× smaller than `int32` for that column.

**Read it back**:

```python
from streaming import StreamingDataset
ds = StreamingDataset(local="./data/nemotron_cascade2_sft_packed_mds", batch_size=1, shuffle=False)
sample = ds[0]   # numpy arrays with the dtypes above
```

#### `cu_seqlens` convention

`cu_seqlens[0] == 0` and `cu_seqlens[-1] == max_pack_length` always — the
last entry is always the full pack length, matching the FlashAttention varlen
convention. Trailing padding (if any) is encoded as a final segment:

```
# 10-token pack, three real sequences of lengths 3/2/4, one trailing pad:
cu_seqlens = [0, 3, 5, 9, 10]
attn_mask  = [1, 1, 1, 1, 1, 1, 1, 1, 1, 0]
```

When a pack is fully filled, the last segment is just the last real sequence
(no padding segment):

```
# 8-token pack, two real sequences of lengths 5/3, no padding:
cu_seqlens = [0, 5, 8]
attn_mask  = [1, 1, 1, 1, 1, 1, 1, 1]
```

Padding **only ever appears at the tail** of a pack — never between two real
sequences. This is enforced by stage 1's `--max-doc-length` filter (so every
sample fits a pack) plus the packer's first-fit policy (which never splits a
sample).

#### Packer algorithm

Per-rank `GreedyPacker` (in `packing_utils.py`), modeled on
`VeOmni/veomni/data/dynamic_batching.py::DynBszBuffer`:

1. `append(input_ids, labels)` — buffer the sample. Raises if the sample is
   longer than `max_pack_length` (should already be filtered by stage 1).
2. `emit_ready()` — once the unused-sample count crosses `--buffer-size`,
   emit packs:
   - Take the first unused sample.
   - Walk the rest of the buffer; take any sample whose length fits the
     remaining budget. Skipped samples (too big for the remaining budget)
     stay in the buffer for the next pack.
   - Pad the tail to `max_pack_length`.
3. `flush()` — at end of stream, ignore the buffer-size threshold and emit
   the remaining samples (the final pack may be under-filled).

Larger `--buffer-size` → more candidates per pack → higher fill rate, at the
cost of memory. For 32K packs with documents ≤ 2K, the default `200` gives
~96-97% fill on Nemotron-Cascade-2 SFT data.

Stage 1's `--max-doc-length` must be `≤` stage 2's `--max-pack-length`; the
defaults are 2048 / 32768.

### Reference run: `safety` subset

```
input :  3570 rows (15 MB JSONL)
stage 1 (8 workers, ~30 s)
   3570 read → 3364 written → 206 oversize dropped (> 2048 tokens)
   output: 21 MB MDS (8 shards)
stage 2 (4 ranks, ~1 s)
   3364 samples → 71 packs of 32 768 tokens
   output: 21 MB MDS (4 shards)
   fill rate: 96.7 %, avg 47.7 real segments per pack
```

### Tests

```bash
python test_packing.py
```

Pure-python invariant tests for `GreedyPacker` (no torch / no transformers).
Covers: exact-fit, under-fill padding, two-sample fit, first-fit skip-and-reuse,
oversize rejection, streaming threshold, tail-only padding.

### Known issues / future work

- **No multi-epoch in stage 2**: `StreamingDataset` is iterated once. To
  produce more packs, currently you have to enlarge the input.
- **No global pack-count cap**: stage 2 always consumes everything.
- **`_multiturn_assistant_mask` is `O(n_turns)` re-tokenizations** when the
  chat template lacks `{% generation %}` blocks (the Nemotron template
  case). For very long agentic conversations (e.g. SWE samples with 100+
  turns), this dominates stage 1 wall time. Could be optimized by caching
  prefix lengths or by extending the template with `{% generation %}` tags.
- **`compression=None`** on the MDS writer. Setting `compression='zstd'` on
  both stages would cut on-disk size another ~50% with minor CPU cost.
