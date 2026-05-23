import argparse
import datetime
import os
from typing import Any, Dict, Iterable, List

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from torch import distributed as dist
from tqdm import tqdm
from streaming import StreamingDataset, StreamingDataLoader

infinite_timeout = datetime.timedelta(days=365)


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Convert shuffled MDS dataset to Parquet using torch distributed.",
	)
	parser.add_argument(
		"--local-root",
		default="./data/sft_mock_mds",
		help="Input local MDS directory",
	)
	parser.add_argument(
		"--out-root",
		default="./data/sft_mock_parquet",
		help="Output directory for parquet files",
	)
	parser.add_argument("--batch-size", type=int, default=1, help="Dataloader batch size")
	parser.add_argument("--num-workers", type=int, default=4, help="Dataloader worker count")
	parser.add_argument("--prefetch-factor", type=int, default=16, help="Dataloader prefetch factor")
	parser.add_argument("--shuffle-seed", type=int, default=42, help="Shuffle seed")
	parser.add_argument("--no-shuffle", action="store_true", help="Disable shuffle")
	parser.add_argument("--flush-rows", type=int, default=1024, help="Rows per parquet write")
	return parser.parse_args()


def setup_distributed() -> tuple[int, int, int]:
	rank = int(os.environ["RANK"])
	world_size = int(os.environ["WORLD_SIZE"])
	local_rank = int(os.environ["LOCAL_RANK"])

	backend = "gloo"
	dist.init_process_group(
		backend=backend,
		rank=rank,
		world_size=world_size,
		timeout=infinite_timeout,
	)

	return rank, world_size, local_rank


def normalize_value(value: Any) -> Any:
	if hasattr(value, "tolist"):
		value = value.tolist()
	if isinstance(value, tuple):
		return list(value)
	return value


def infer_batch_size(value: Any) -> int:
	if hasattr(value, "shape"):
		shape = getattr(value, "shape")
		if len(shape) <= 1:
			return 1
		return int(shape[0])
	if isinstance(value, list):
		if not value:
			return 1
		first = value[0]
		if isinstance(first, (list, tuple, np.ndarray)):
			return len(value)
		return 1
	return 1


def iter_samples_from_batch(batch: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
	first_value = next(iter(batch.values()))
	batch_size = infer_batch_size(first_value)
	for i in range(batch_size):
		sample: Dict[str, Any] = {}
		for key, value in batch.items():
			if batch_size == 1:
				item = value
				if isinstance(item, (list, tuple)) and item and not isinstance(item[0], (int, np.integer)):
					item = item[0]
				if hasattr(item, "shape") and getattr(item, "shape").__len__() >= 2:
					item = item[0]
			else:
				item = value[i]
			item = normalize_value(item)
			sample[key] = item
		yield sample


def infer_schema(sample: Dict[str, Any]) -> pa.schema:
	fields: List[pa.Field] = []
	for key, value in sample.items():
		if isinstance(value, list):
			fields.append(pa.field(key, pa.list_(pa.int32())))
		elif isinstance(value, (int, np.integer)):
			fields.append(pa.field(key, pa.int64()))
		else:
			raise TypeError(f"Unsupported field type for {key}: {type(value)}")
	return pa.schema(fields)


def flush_rows(writer: pq.ParquetWriter, buffer: Dict[str, List[Any]], schema: pa.Schema) -> None:
	if not buffer:
		return
	for key in buffer:
		if not buffer[key]:
			return
	batch = pa.Table.from_pydict(buffer, schema=schema)
	writer.write_table(batch)
	for key in buffer:
		buffer[key].clear()


def main() -> None:
	args = parse_args()
	rank, world_size, local_rank = setup_distributed()
	if rank == 0:
		print("the initialization is finished")

	dataset = StreamingDataset(
		local=args.local_root,
		batch_size=args.batch_size,
		shuffle=not args.no_shuffle,
		shuffle_seed=args.shuffle_seed,
	)
	loader = StreamingDataLoader(
		dataset,
		num_workers=args.num_workers,
		batch_size=args.batch_size,
		prefetch_factor=args.prefetch_factor,
	)

	length = len(dataset)
	if rank == 0:
		print(f"the dataset is loaded and the length is {length}")

	os.makedirs(args.out_root, exist_ok=True)
	output_path = os.path.join(args.out_root, f"part-{rank:05}.parquet")

	writer: pq.ParquetWriter | None = None
	buffer: Dict[str, List[Any]] = {}
	row_count = 0

	for batch in tqdm(loader, desc=f"Writing parquet rank {rank}"):
		for sample in iter_samples_from_batch(batch):
			if writer is None:
				schema = infer_schema(sample)
				writer = pq.ParquetWriter(output_path, schema)
				buffer = {name: [] for name in schema.names}
			for key, value in sample.items():
				buffer[key].append(value)
			row_count += 1
			if row_count % args.flush_rows == 0:
				flush_rows(writer, buffer, writer.schema)

	if writer is not None:
		flush_rows(writer, buffer, writer.schema)
		writer.close()

	dist.barrier()
	dist.destroy_process_group()
	if rank == 0:
		print(f"parquet shards written to {args.out_root} with world size {world_size}")


if __name__ == "__main__":
	main()
