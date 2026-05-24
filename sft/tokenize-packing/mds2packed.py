"""Stage 2: tokenized MDS -> packed MDS.

Launched via torchrun. Each rank:
1. Opens ``StreamingDataset(shuffle=True)`` over the stage-1 MDS dataset,
   which auto-shards samples across ranks.
2. Feeds samples into a per-rank ``GreedyPacker``.
3. Writes packs into its own MDS shard directory ``shard_{rank}/``.

After all ranks finish, rank 0 merges the per-rank indexes into a single
top-level ``index.json`` so the output directory can be opened as one MDS
dataset.

Each row in the output MDS has::

    input_ids  : ndarray:int32, length == --max-pack-length
    labels     : ndarray:int32, length == --max-pack-length  (padding = -100)
    cu_seqlens : ndarray:int32, starts at 0, ends at --max-pack-length
    attn_mask  : ndarray:int8,  1 = real token, 0 = padding (tail only)
"""

from __future__ import annotations

import argparse
import datetime
import logging
import os

import numpy as np
import torch.distributed as dist
from streaming import MDSWriter, StreamingDataset
from streaming.base.util import merge_index
from tqdm import tqdm

from packing_utils import GreedyPacker

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [rank=%(name)s] %(message)s")


MDS_COLUMNS = {
    "input_ids": "ndarray:int32",
    "labels": "ndarray:int32",
    "cu_seqlens": "ndarray:int32",
    "attn_mask": "ndarray:int8",
}


def setup_distributed() -> tuple[int, int, int]:
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    dist.init_process_group(
        backend="gloo",
        rank=rank,
        world_size=world_size,
        timeout=datetime.timedelta(days=365),
    )
    return rank, world_size, local_rank


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MDS -> packed MDS via torchrun")
    p.add_argument("--local-root", required=True, help="Input MDS directory (from sft2mds.py)")
    p.add_argument("--out-root", required=True, help="Output MDS directory (per-rank shard_{N}/ + merged index.json)")
    p.add_argument("--max-pack-length", type=int, default=32768, help="Target tokens per pack")
    p.add_argument("--pad-token-id", type=int, default=0, help="Pad token id for tail padding")
    p.add_argument("--buffer-size", type=int, default=200, help="Min unused samples before streaming emit")
    p.add_argument("--shuffle-seed", type=int, default=42)
    return p.parse_args()


def _pack_to_mds_row(pack: dict) -> dict:
    return {
        "input_ids": np.asarray(pack["input_ids"], dtype=np.int32),
        "labels": np.asarray(pack["labels"], dtype=np.int32),
        "cu_seqlens": np.asarray(pack["cu_seqlens"], dtype=np.int32),
        "attn_mask": np.asarray(pack["attn_mask"], dtype=np.int8),
    }


def main() -> int:
    args = parse_args()
    rank, world_size, _ = setup_distributed()
    logger = logging.getLogger(str(rank))

    if rank == 0:
        logger.info("distributed init complete, world_size=%d", world_size)
        os.makedirs(args.out_root, exist_ok=True)
    dist.barrier()

    dataset = StreamingDataset(
        local=args.local_root,
        batch_size=1,
        shuffle=True,
        shuffle_seed=args.shuffle_seed,
    )
    # StreamingDataset auto-shards across ranks; iterate it directly.

    shard_dir = os.path.join(args.out_root, f"shard_{rank}")
    packer = GreedyPacker(
        max_pack_length=args.max_pack_length,
        pad_token_id=args.pad_token_id,
        buffer_size=args.buffer_size,
    )

    n_in = 0
    n_out = 0

    def _write_packs(writer, packs_iter):
        nonlocal n_out
        for pack in packs_iter:
            writer.write(_pack_to_mds_row(pack))
            n_out += 1

    with MDSWriter(out=shard_dir, columns=MDS_COLUMNS, compression=None) as writer:
        iterator = tqdm(dataset, desc=f"rank{rank} pack", disable=(rank != 0))
        for sample in iterator:
            input_ids = sample["input_ids"].tolist()
            labels = sample["labels"].tolist()
            n_in += 1
            packer.append(input_ids, labels)
            _write_packs(writer, packer.emit_ready())

        # Flush remaining samples into final (possibly padded) packs.
        _write_packs(writer, packer.flush())

    logger.info("rank %d done: read=%d, packs=%d -> %s", rank, n_in, n_out, shard_dir)

    dist.barrier()

    # Rank 0 merges all per-rank shard indexes into a single top-level index.json.
    if rank == 0:
        index_files = [os.path.join(args.out_root, f"shard_{r}", "index.json") for r in range(world_size)]
        merge_index(index_files, out=args.out_root, keep_local=True)
        logger.info("merged index written to %s/index.json", args.out_root)

    dist.barrier()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
