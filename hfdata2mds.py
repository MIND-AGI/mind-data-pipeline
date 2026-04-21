
import argparse
import os
from typing import Iterator, Optional, Tuple
from multiprocessing import Pool
from streaming import MDSWriter
from streaming.base.util import merge_index
from datasets import load_dataset

Task = Tuple[str, str, int, int, Optional[str], str]

def each_task(
    data_repo: str,
    out_root: str,
    groups: int,
    dataset_name: Optional[str],
    text_column: str,
) -> Iterator[Task]:

    if groups <= 0:
        raise ValueError("groups 必须是正整数")
    start_percent = 0
    for data_group in range(groups):
        sub_out_root = os.path.join(out_root, f"shard_{data_group}")
        # 计算当前分片的结束百分比
        # 使用 (data_group + 1) * 100 // groups 这种整数除法来计算终点
        # 这样可以避免浮点数误差，并且结果是整数
        end_percent = (data_group + 1) * 100 // groups
        # 特殊处理：确保最后一个分片的结束点是 100%
        if data_group == groups - 1:
            end_percent = 100
        yield data_repo, sub_out_root, start_percent, end_percent, dataset_name, text_column
        # 更新下一个分片的起点
        start_percent = end_percent

def convert_to_mds(args: Task) -> None:

    data_repo, sub_out_root, start_percent, end_percent, dataset_name, text_column = args
    print(f"Processing {data_repo} from {start_percent}% to {end_percent}%", flush=True)

    # here can be changed to related shard
    load_kwargs = {
        "path": data_repo,
        "split": f"train[{start_percent}%:{end_percent}%]",
    }
    if dataset_name:
        load_kwargs["name"] = dataset_name
    data_shard = load_dataset(**load_kwargs)
    columns = {text_column: 'str'}

    with MDSWriter(out=sub_out_root, columns=columns) as out:
        # change to for loop sequantially write back
        length = len(data_shard)
        for i in range(length):
            sample = {text_column: data_shard[i][text_column]}
            out.write(sample)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert Hugging Face dataset shards to MDS format.")
    parser.add_argument(
        "--data-repo",
        default="HuggingFaceFW/fineweb-edu",
        help="Hugging Face dataset repo, e.g. HuggingFaceFW/fineweb-edu",
    )
    parser.add_argument(
        "--out-root",
        default="./data/fineweb-edu-sample-10BT_mds",
        help="Output root directory for MDS shards",
    )
    parser.add_argument(
        "--num-groups",
        type=int,
        default=10,
        help="How many dataset shard groups to split",
    )
    parser.add_argument(
        "--num-process",
        type=int,
        default=10,
        help="Worker process count",
    )
    parser.add_argument(
        "--dataset-name",
        default="sample-10BT",
        help="Dataset config name for load_dataset; set empty to disable",
    )
    parser.add_argument(
        "--text-column",
        default="text",
        help="Dataset column name to write into MDS",
    )
    args = parser.parse_args()
    if args.num_groups <= 0:
        parser.error("--num-groups must be a positive integer")
    if args.num_process <= 0:
        parser.error("--num-process must be a positive integer")
    return args

def init_worker():
    # Get the pid for the current worker process
    pid = os.getpid()
    print(f'\nInitialize Worker PID: {pid}', flush=True, end='')
    hf_cache = os.getenv("HF_DATASETS_CACHE")
    hf_hubcache = os.getenv("HUGGINGFACE_HUB_CACHE")
    print(f"the caches are {hf_cache} and {hf_hubcache}", flush=True, end='')

if __name__ == "__main__":
    args = parse_args()
    dataset_name = args.dataset_name.strip() if args.dataset_name else None
    if dataset_name == "":
        dataset_name = None

    arg_tuples = each_task(
        args.data_repo,
        args.out_root,
        args.num_groups,
        dataset_name,
        args.text_column,
    )

    # Process group of data in parallel into directories of shards.
    with Pool(initializer=init_worker, processes=args.num_process) as pool:
        for _ in pool.imap(convert_to_mds, arg_tuples):
            pass

    print("=========the sharding convertion is finished, now we do merge=========")

    merge_index(args.out_root)
    
