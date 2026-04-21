import argparse
import os
import json
from tqdm import tqdm
from torch import distributed as dist
import datetime
infinite_timeout = datetime.timedelta(days=365)
from streaming import StreamingDataset, StreamingDataLoader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert MDS dataset to JSONL using torch distributed.")
    parser.add_argument(
        "--local-root",
        default="./data/fineweb-edu-sample-10BT_mds",
        help="Input local MDS directory",
    )
    parser.add_argument(
        "--out-root",
        default="./data/fineweb-edu-10BT_jsonl",
        help="Output directory for merged JSONL chunks",
    )
    return parser.parse_args()

def setup_distributed():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    
    backend = 'gloo'
    dist.init_process_group(
        backend=backend,
        rank=rank,
        world_size=world_size,
        timeout=infinite_timeout,
    )
    
    return rank, world_size, local_rank

if __name__ == "__main__":
    args = parse_args()
    rank, world_size, local_rank = setup_distributed()
    if rank == 0:
        print(f"the initialization is finished")
    
    dataset = StreamingDataset(
        local=args.local_root,
        batch_size=1,
        shuffle=True,
        shuffle_seed=42,
    )
    dataloader = StreamingDataLoader(
        dataset, 
        num_workers=4, 
        batch_size=1,
        prefetch_factor=16,
    )

    length = len(dataset)
    out_root = args.out_root
    sub_out_root = os.path.join(out_root, f"chunk.{rank}.jsonl")
    if out_root: # 确保目录不是空字符串（比如当文件就在当前目录时）
        os.makedirs(out_root, exist_ok=True)

    if rank == 0:
        print(f"the dataset is loaded and the length is {length}")
    
    count = 0
    with open(sub_out_root, 'w', encoding='utf-8') as out_file:
        for data in tqdm(dataloader, desc="Writing to JSONL"):
            if count >= length:
                break
            sample = {"text": data["text"][0]}
            json_string = json.dumps(sample, ensure_ascii=False)
            out_file.write(json_string + '\n')
            count += 1
    
    dist.barrier()
    dist.destroy_process_group()
