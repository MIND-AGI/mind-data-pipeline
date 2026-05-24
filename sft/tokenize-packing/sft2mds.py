"""Stage 1: JSONL SFT -> tokenized MDS.

Reads JSONL rows in parallel, applies chat template + assistant-only loss
masking, drops samples whose tokenized length exceeds --max-doc-length, and
writes ``{input_ids: ndarray:int32, labels: ndarray:int32}`` MDS shards.

Sharding strategy:
- If ``len(files) >= --num-process``: file-level sharding (each worker takes a
  round-robin slice of files).
- Otherwise: line-stride sharding (every worker sees every file; line N is
  handled by worker N % num_workers).
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import os
import time
from glob import glob
from multiprocessing import Pool
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
from streaming import MDSWriter
from streaming.base.util import merge_index
from tqdm import tqdm

# Per-worker globals (initialized in _worker_init).
_TOKENIZER = None
_MAX_DOC_LENGTH = None
_TEMPLATE_KWARGS: Dict[str, Any] = {}
_CHARS_PER_TOKEN: float = 0.0
_PROGRESS_COUNTER: Any = None
# Batch progress increments to amortize lock contention; flushed by _flush_progress.
_PROGRESS_BATCH = 64
_PROGRESS_LOCAL = [0]

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [pid=%(process)d] %(message)s")
logger = logging.getLogger("sft2mds")


ByteChunk = Tuple[str, int, int]  # (file_path, start_byte, end_byte)
WorkerShard = Tuple[int, List[ByteChunk], str]  # (worker_id, chunks, out_dir)


def _resolve_files(input_path: str) -> List[str]:
    """Resolve --input-path to a sorted list of JSONL files.

    Supports: a single file path; a glob (including ``**`` recursive); or a
    directory (recursively scanned for ``*.jsonl``).
    """
    if os.path.isfile(input_path):
        return [input_path]
    if os.path.isdir(input_path):
        files = sorted(glob(os.path.join(input_path, "**", "*.jsonl"), recursive=True))
    else:
        files = sorted(p for p in glob(input_path, recursive=True) if os.path.isfile(p))
    if not files:
        raise FileNotFoundError(f"No JSONL files found for: {input_path}")
    return files


def _build_shards(
    files: List[str],
    num_workers: int,
    out_root: str,
    node_rank: int = 0,
    num_nodes: int = 1,
) -> List[WorkerShard]:
    """Split the total input bytes across the *global* worker set.

    Files are not the unit of work — bytes are. The global worker set has
    ``G = num_nodes * num_workers`` workers. Total bytes are partitioned into
    ``G`` contiguous chunks; this node owns global ids
    ``[node_rank * num_workers, (node_rank + 1) * num_workers)``.

    Each returned shard's directory is named by its *global* worker id, so
    multiple nodes writing into the same ``out_root`` never collide and a
    single later ``merge_index`` call can pick up everything.

    A chunk owns every JSONL line whose **start byte is in [start, end)**. The
    chunk reader skips the partial line at the front (it belongs to the
    previous chunk) and naturally reads past ``end`` to finish the last line
    whose start is still ``< end``.
    """
    if not (0 <= node_rank < num_nodes):
        raise ValueError(f"node_rank {node_rank} out of range [0, {num_nodes})")

    sized = [(f, os.path.getsize(f)) for f in files]
    total_bytes = sum(s for _, s in sized)
    if total_bytes == 0:
        raise RuntimeError("all input files are empty")

    global_workers = num_nodes * num_workers
    target = max(1, total_bytes // global_workers)

    # Allocate chunks for ALL global workers (cheap — just byte-range bookkeeping,
    # no file IO). Then slice out the ones this node owns.
    chunks_per_global: List[List[ByteChunk]] = [[] for _ in range(global_workers)]
    cur_w = 0
    cur_w_bytes = 0
    for path, size in sized:
        cursor = 0
        while cursor < size:
            if cur_w >= global_workers - 1:
                chunks_per_global[-1].append((path, cursor, size))
                cursor = size
            else:
                remaining = target - cur_w_bytes
                take = min(size - cursor, remaining)
                chunks_per_global[cur_w].append((path, cursor, cursor + take))
                cur_w_bytes += take
                cursor += take
                if cur_w_bytes >= target:
                    cur_w += 1
                    cur_w_bytes = 0

    my_global_start = node_rank * num_workers
    my_global_end = my_global_start + num_workers
    return [
        (gid, chunks_per_global[gid], os.path.join(out_root, f"shard_{gid}"))
        for gid in range(my_global_start, my_global_end)
    ]


def _iter_jsonl_chunk(path: str, start_byte: int, end_byte: int) -> Iterator[Tuple[int, Dict[str, Any]]]:
    """Yield ``(line_start_byte, parsed_row)`` for every line in this chunk.

    A line "belongs to" this chunk iff its start byte is in ``[start_byte, end_byte)``.
    The reader seeks to ``start_byte``, drops any partial line (owned by the
    previous chunk), then reads forward until the next line would start at or
    after ``end_byte``. Reading binary + decoding once per line is slightly
    faster than text mode on large files.
    """
    with open(path, "rb") as f:
        if start_byte > 0:
            # Peek the byte before start_byte. If it's '\n', start_byte is exactly
            # a line boundary and we should NOT skip; otherwise we're mid-line
            # and need to discard the rest (the previous chunk owns it).
            f.seek(start_byte - 1)
            prev = f.read(1)
            if prev != b"\n":
                f.readline()
        # else: starting at file head, no skip.

        while True:
            line_start = f.tell()
            if line_start >= end_byte:
                break
            raw = f.readline()
            if not raw:
                break
            stripped = raw.strip()
            if not stripped:
                continue
            try:
                yield line_start, json.loads(stripped)
            except json.JSONDecodeError as e:
                logger.warning("skip malformed line %s@%d (%s)", path, line_start, e)
                continue


def _worker_init(
    tokenizer_path: str,
    max_doc_length: int,
    trust_remote_code: bool,
    template_kwargs: Dict[str, Any],
    chars_per_token: float,
    progress_counter,
) -> None:
    """Pool initializer: load the tokenizer once per worker process."""
    global _TOKENIZER, _MAX_DOC_LENGTH, _TEMPLATE_KWARGS, _CHARS_PER_TOKEN, _PROGRESS_COUNTER
    from transformers import AutoTokenizer

    _TOKENIZER = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=trust_remote_code)
    _MAX_DOC_LENGTH = max_doc_length
    _TEMPLATE_KWARGS = template_kwargs or {}
    _CHARS_PER_TOKEN = chars_per_token
    _PROGRESS_COUNTER = progress_counter
    logger.info(
        "worker tokenizer loaded: %s (max_doc_length=%d, chars_per_token=%.1f, template_kwargs=%s)",
        tokenizer_path,
        max_doc_length,
        chars_per_token,
        _TEMPLATE_KWARGS,
    )


def _bump_progress(n: int = 1) -> None:
    """Per-worker progress accumulator; flushes to the shared counter in batches."""
    if _PROGRESS_COUNTER is None:
        return
    _PROGRESS_LOCAL[0] += n
    if _PROGRESS_LOCAL[0] >= _PROGRESS_BATCH:
        with _PROGRESS_COUNTER.get_lock():
            _PROGRESS_COUNTER.value += _PROGRESS_LOCAL[0]
        _PROGRESS_LOCAL[0] = 0


def _flush_progress() -> None:
    if _PROGRESS_COUNTER is None or _PROGRESS_LOCAL[0] == 0:
        return
    with _PROGRESS_COUNTER.get_lock():
        _PROGRESS_COUNTER.value += _PROGRESS_LOCAL[0]
    _PROGRESS_LOCAL[0] = 0


def _process_shard(shard: WorkerShard) -> Dict[str, int]:
    """Run one worker: read assigned lines, tokenize, write MDS shard.

    Returns counters for logging at the master.
    """
    from formatting_utils import tokenize_sample  # local import: worker only

    worker_id, chunks, out_dir = shard
    os.makedirs(out_dir, exist_ok=True)

    total_bytes = sum(e - s for _, s, e in chunks)
    ranges = ", ".join(f"{os.path.basename(p)}[{s}:{e})" for p, s, e in chunks)
    logger.info(
        "worker %d start: %d chunk(s), %.3f GB, ranges: %s",
        worker_id,
        len(chunks),
        total_bytes / 2**30,
        ranges,
    )

    columns = {"input_ids": "ndarray:int32", "labels": "ndarray:int32"}

    stats = {"read": 0, "written": 0, "oversize": 0, "no_assistant": 0, "tokenize_error": 0, "too_short": 0}

    # `exist_ok=True` lets the script be re-run; MDSWriter will overwrite.
    from formatting_utils import OversizeError  # local import: worker only

    try:
        with MDSWriter(out=out_dir, columns=columns, compression=None) as writer:
            for path, start_byte, end_byte in chunks:
                for line_pos, row in _iter_jsonl_chunk(path, start_byte, end_byte):
                    stats["read"] += 1
                    _bump_progress(1)
                    try:
                        tok = tokenize_sample(
                            _TOKENIZER,
                            row,
                            template_kwargs=_TEMPLATE_KWARGS,
                            max_length=_MAX_DOC_LENGTH,
                            chars_per_token=_CHARS_PER_TOKEN,
                        )
                    except OversizeError:
                        stats["oversize"] += 1
                        continue
                    except ValueError as e:
                        msg = str(e)
                        if "assistant" in msg.lower():
                            stats["no_assistant"] += 1
                        elif "too short" in msg.lower():
                            stats["too_short"] += 1
                        else:
                            stats["tokenize_error"] += 1
                        if stats["tokenize_error"] <= 5:
                            logger.warning("worker %d %s@%d tokenize_error: %s", worker_id, path, line_pos, e)
                        continue
                    except Exception as e:  # noqa: BLE001
                        stats["tokenize_error"] += 1
                        if stats["tokenize_error"] <= 5:
                            logger.warning("worker %d %s@%d unexpected: %s", worker_id, path, line_pos, e)
                        continue

                    writer.write(
                        {
                            "input_ids": np.asarray(tok["input_ids"], dtype=np.int32),
                            "labels": np.asarray(tok["labels"], dtype=np.int32),
                        }
                    )
                    stats["written"] += 1
    finally:
        _flush_progress()

    logger.info("worker %d done: %s", worker_id, stats)
    return stats


def _count_jsonl_lines(files: List[str]) -> int:
    """Fast newline count across files (no JSON parsing)."""
    total = 0
    for f in files:
        with open(f, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                total += chunk.count(b"\n")
    return total


def _count_lines_in_chunks(chunks: List[ByteChunk]) -> int:
    """Newline count within this node's assigned byte ranges (no JSON parsing).

    Used as the tqdm denominator in multi-node mode so the progress bar matches
    *this node's* workload (~1/num_nodes of the data) rather than the global
    total — otherwise the bar would top out at 1/num_nodes and the ETA would be
    wrong. It's an estimate, not exact: blank/malformed lines (skipped at read
    time) are still counted, and chunk-boundary lines may be off by ~1. Each
    node only scans its own slice.
    """
    total = 0
    block = 1 << 20
    for path, start, end in chunks:
        remaining = end - start
        if remaining <= 0:
            continue
        with open(path, "rb") as fh:
            fh.seek(start)
            while remaining > 0:
                buf = fh.read(min(block, remaining))
                if not buf:
                    break
                total += buf.count(b"\n")
                remaining -= len(buf)
    return total


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="JSONL SFT -> tokenized MDS")
    p.add_argument("--input-path", help="JSONL file, glob pattern, or directory (recursive *.jsonl scan). Not required with --merge-only.")
    p.add_argument("--out-root", required=True, help="Output root directory for MDS shards")
    p.add_argument("--tokenizer", help="HF tokenizer name or local path. Not required with --merge-only.")
    p.add_argument("--max-doc-length", type=int, default=2048, help="Drop samples whose tokenized length exceeds this")
    p.add_argument("--num-process", type=int, default=8, help="Number of parallel worker processes on THIS node")
    p.add_argument(
        "--num-nodes",
        type=int,
        default=1,
        help=(
            "Total number of cooperating nodes (machines) writing into --out-root. "
            "Global worker count = num_nodes * num_process. Default 1 (single-node, behavior unchanged)."
        ),
    )
    p.add_argument(
        "--node-rank",
        type=int,
        default=0,
        help="0-indexed rank of THIS node, must be in [0, num_nodes). Default 0.",
    )
    p.add_argument(
        "--merge-only",
        action="store_true",
        help=(
            "Skip tokenization; just scan --out-root/shard_*/index.json and merge them into a single "
            "top-level index.json. Use this once after all nodes finish in multi-node mode."
        ),
    )
    p.add_argument("--trust-remote-code", action="store_true", help="Pass trust_remote_code=True to from_pretrained")
    p.add_argument(
        "--template-kwargs",
        default="{}",
        help=(
            "JSON dict of extra kwargs forwarded to apply_chat_template at every call "
            "(both full tokenize and prefix mask reconstruction). Example for Nemotron: "
            '\'{"truncate_history_thinking": false}\' to preserve <think> blocks in earlier turns.'
        ),
    )
    p.add_argument(
        "--no-count-lines",
        action="store_true",
        help="Skip the pre-scan that counts input lines. Progress bar will still show throughput but no ETA.",
    )
    p.add_argument(
        "--chars-per-token",
        type=float,
        default=5.0,
        help=(
            "Char-based oversize pre-filter. Reject samples whose total content chars > "
            "max_doc_length * chars_per_token BEFORE any tokenize call. Should be an UPPER "
            "bound on chars/token in your data (Nemotron-Cascade-2 SFT is ~3.7-4.5; 5.0 leaves "
            "a safety margin). 0 disables the pre-filter."
        ),
    )
    return p.parse_args()


def _merge_only(out_root: str) -> int:
    """Scan ``out_root/shard_*/index.json`` and merge into a single index.json.

    Used after all nodes finish writing in multi-node mode. Idempotent: safe to
    re-run (overwrites the top-level index.json).
    """
    shard_indexes = sorted(glob(os.path.join(out_root, "shard_*", "index.json")))
    if not shard_indexes:
        raise SystemExit(f"--merge-only: no shard_*/index.json found under {out_root}")
    logger.info("merging %d shard indexes under %s", len(shard_indexes), out_root)
    merge_index(shard_indexes, out=out_root, keep_local=True)
    logger.info("merged index written to %s/index.json", out_root)
    return 0


def main() -> int:
    args = parse_args()

    if args.merge_only:
        return _merge_only(args.out_root)

    if not args.input_path:
        raise SystemExit("--input-path is required (only optional with --merge-only)")
    if not args.tokenizer:
        raise SystemExit("--tokenizer is required (only optional with --merge-only)")
    if not (0 <= args.node_rank < args.num_nodes):
        raise SystemExit(f"--node-rank {args.node_rank} must be in [0, --num-nodes={args.num_nodes})")

    try:
        template_kwargs = json.loads(args.template_kwargs)
        if not isinstance(template_kwargs, dict):
            raise ValueError("must be a JSON object")
    except (json.JSONDecodeError, ValueError) as e:
        raise SystemExit(f"--template-kwargs is not a valid JSON object: {e}")

    files = _resolve_files(args.input_path)
    total_bytes = sum(os.path.getsize(f) for f in files)
    logger.info(
        "resolved %d input file(s), %.2f GB total (node %d/%d)",
        len(files),
        total_bytes / 2**30,
        args.node_rank,
        args.num_nodes,
    )

    os.makedirs(args.out_root, exist_ok=True)
    shards = _build_shards(files, args.num_process, args.out_root, args.node_rank, args.num_nodes)
    worker_bytes = [sum(e - s for _, s, e in chunks) for _, chunks, _ in shards]
    global_worker_ids = [wid for wid, _, _ in shards]
    logger.info(
        "byte-range sharding: node %d/%d, global workers %d..%d (of %d total), "
        "per-worker bytes min=%.2fG max=%.2fG avg=%.2fG",
        args.node_rank,
        args.num_nodes,
        global_worker_ids[0],
        global_worker_ids[-1],
        args.num_nodes * args.num_process,
        min(worker_bytes) / 2**30,
        max(worker_bytes) / 2**30,
        sum(worker_bytes) / len(worker_bytes) / 2**30,
    )
    logger.info("template_kwargs=%s", template_kwargs)

    total_lines: Optional[int] = None
    if not args.no_count_lines:
        t0 = time.time()
        if args.num_nodes > 1:
            # Count only the lines in this node's assigned byte ranges so the
            # progress bar denominator matches this node's workload and the ETA
            # is correct. Each node scans only its own ~1/num_nodes slice.
            my_chunks = [c for _, chunks, _ in shards for c in chunks]
            total_lines = _count_lines_in_chunks(my_chunks)
            logger.info(
                "counted %d lines in this node's byte ranges in %.1fs (node %d/%d)",
                total_lines,
                time.time() - t0,
                args.node_rank,
                args.num_nodes,
            )
        else:
            total_lines = _count_jsonl_lines(files)
            logger.info("counted %d total lines across %d file(s) in %.1fs", total_lines, len(files), time.time() - t0)

    # Shared atomic counter; workers bump it in batches, main thread polls into tqdm.
    progress = mp.Value("Q", 0, lock=True)

    pool = Pool(
        processes=args.num_process,
        initializer=_worker_init,
        initargs=(
            args.tokenizer,
            args.max_doc_length,
            args.trust_remote_code,
            template_kwargs,
            args.chars_per_token,
            progress,
        ),
    )
    async_result = pool.map_async(_process_shard, shards)

    pbar = tqdm(total=total_lines, desc="tokenize", unit="docs", unit_scale=False, smoothing=0.05, dynamic_ncols=True)
    last = 0
    try:
        while not async_result.ready():
            cur = progress.value
            if cur > last:
                pbar.update(cur - last)
                last = cur
            time.sleep(0.5)
    finally:
        pool.close()
        pool.join()
        cur = progress.value
        if cur > last:
            pbar.update(cur - last)
        pbar.close()

    results = async_result.get()
    totals = {k: sum(r[k] for r in results) for k in results[0].keys()}
    logger.info("totals (this node): %s", totals)

    if args.num_nodes == 1:
        # Single-node: merge this node's shards directly.
        index_files = [os.path.join(out_dir, "index.json") for _, _, out_dir in shards]
        merge_index(index_files, out=args.out_root, keep_local=True)
        logger.info("merged index written to %s/index.json", args.out_root)
    else:
        logger.info(
            "node %d/%d done. After ALL nodes finish, run once on any node:\n"
            "    bash sft2mds.sh --out-root %s --num-nodes %d --merge-only",
            args.node_rank,
            args.num_nodes,
            args.out_root,
            args.num_nodes,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
