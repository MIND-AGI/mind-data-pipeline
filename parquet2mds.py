import argparse
import os
from glob import glob
from multiprocessing import Pool
from typing import Iterator, List, Tuple

from datasets import load_dataset
from streaming import MDSWriter
from streaming.base.util import merge_index

Task = Tuple[List[str], str, int, int, str]


def resolve_parquet_files(input_path: str) -> List[str]:
    # Support either a single parquet path or a glob pattern.
    if os.path.isfile(input_path):
        return [input_path]

    matches = sorted(glob(input_path))
    parquet_files = [p for p in matches if os.path.isfile(p) and p.endswith(".parquet")]
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found for input path/pattern: {input_path}")
    return parquet_files


def each_task(
    parquet_files: List[str],
    out_root: str,
    groups: int,
    text_column: str,
) -> Iterator[Task]:
    if groups <= 0:
        raise ValueError("groups must be a positive integer")

    start_percent = 0
    for data_group in range(groups):
        sub_out_root = os.path.join(out_root, f"shard_{data_group}")
        end_percent = (data_group + 1) * 100 // groups
        if data_group == groups - 1:
            end_percent = 100
        yield parquet_files, sub_out_root, start_percent, end_percent, text_column
        start_percent = end_percent


def convert_to_mds(args: Task) -> None:
    parquet_files, sub_out_root, start_percent, end_percent, text_column = args
    print(
        f"Processing parquet files from {start_percent}% to {end_percent}%",
        flush=True,
    )

    data_shard = load_dataset(
        "parquet",
        data_files=parquet_files,
        split=f"train[{start_percent}%:{end_percent}%]",
    )
    columns = {text_column: "str"}

    with MDSWriter(out=sub_out_root, columns=columns) as out:
        length = len(data_shard)
        for i in range(length):
            sample = {text_column: data_shard[i][text_column]}
            out.write(sample)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert one parquet file or a parquet glob pattern to MDS shards.",
    )
    parser.add_argument(
        "--input-path",
        default="./data/*.parquet",
        help="Single parquet path or glob pattern, e.g. ./data/*.parquet",
    )
    parser.add_argument(
        "--out-root",
        default="./data/parquet_mds",
        help="Output root directory for MDS shards",
    )
    parser.add_argument(
        "--num-groups",
        type=int,
        default=10,
        help="How many shard groups to split",
    )
    parser.add_argument(
        "--num-process",
        type=int,
        default=10,
        help="Worker process count",
    )
    parser.add_argument(
        "--text-column",
        default="text",
        help="Parquet column name to write into MDS",
    )

    args = parser.parse_args()
    if args.num_groups <= 0:
        parser.error("--num-groups must be a positive integer")
    if args.num_process <= 0:
        parser.error("--num-process must be a positive integer")
    return args


def init_worker() -> None:
    pid = os.getpid()
    print(f"\nInitialize Worker PID: {pid}", flush=True, end="")
    hf_cache = os.getenv("HF_DATASETS_CACHE")
    hf_hubcache = os.getenv("HUGGINGFACE_HUB_CACHE")
    print(f"the caches are {hf_cache} and {hf_hubcache}", flush=True, end="")


if __name__ == "__main__":
    args = parse_args()
    parquet_files = resolve_parquet_files(args.input_path)

    arg_tuples = each_task(
        parquet_files=parquet_files,
        out_root=args.out_root,
        groups=args.num_groups,
        text_column=args.text_column,
    )

    with Pool(initializer=init_worker, processes=args.num_process) as pool:
        for _ in pool.imap(convert_to_mds, arg_tuples):
            pass

    print("=========the sharding conversion is finished, now we do merge=========")
    merge_index(args.out_root)
