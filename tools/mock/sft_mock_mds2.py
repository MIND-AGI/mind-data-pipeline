import argparse
import os
import random
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from streaming import MDSWriter
from streaming.base.util import merge_index


@dataclass
class SampleConfig:
	seq_len: int
	vocab_size: int
	pad_id: int
	min_docs: int
	max_docs: int
	min_doc_len: int
	max_doc_len: int
	mask_span_max: int
	mask_span_min: int
	mask_spans_max: int


def validate_config(cfg: SampleConfig) -> None:
	if cfg.seq_len <= 0:
		raise ValueError("seq_len must be positive")
	if cfg.vocab_size <= 1:
		raise ValueError("vocab_size must be > 1")
	if cfg.min_docs <= 0 or cfg.max_docs <= 0:
		raise ValueError("min_docs and max_docs must be positive")
	if cfg.min_docs > cfg.max_docs:
		raise ValueError("min_docs cannot exceed max_docs")
	if cfg.min_doc_len <= 0 or cfg.max_doc_len <= 0:
		raise ValueError("min_doc_len and max_doc_len must be positive")
	if cfg.min_doc_len > cfg.max_doc_len:
		raise ValueError("min_doc_len cannot exceed max_doc_len")
	if cfg.seq_len < cfg.min_doc_len * cfg.min_docs:
		raise ValueError("seq_len too small for min_docs * min_doc_len")
	if cfg.mask_span_min <= 0 or cfg.mask_span_max <= 0:
		raise ValueError("mask_span_min and mask_span_max must be positive")
	if cfg.mask_span_min > cfg.mask_span_max:
		raise ValueError("mask_span_min cannot exceed mask_span_max")
	if cfg.mask_spans_max <= 0:
		raise ValueError("mask_spans_max must be positive")


def choose_num_docs(cfg: SampleConfig, rng: random.Random) -> int:
	max_feasible = min(cfg.max_docs, cfg.seq_len // cfg.min_doc_len)
	if max_feasible < cfg.min_docs:
		raise ValueError("seq_len too small for requested doc count")
	return rng.randint(cfg.min_docs, max_feasible)


def build_doc_lengths(cfg: SampleConfig, rng: random.Random) -> List[int]:
	num_docs = choose_num_docs(cfg, rng)
	lengths: List[int] = []
	remaining = cfg.seq_len

	for i in range(num_docs):
		remaining_docs = num_docs - i
		max_len_for_current = min(
			cfg.max_doc_len,
			remaining - cfg.min_doc_len * (remaining_docs - 1),
		)
		min_len_for_current = cfg.min_doc_len
		if max_len_for_current < min_len_for_current:
			max_len_for_current = min_len_for_current
		length = rng.randint(min_len_for_current, max_len_for_current)
		lengths.append(length)
		remaining -= length

	return lengths


def mask_doc_spans(
	labels: List[int],
	doc_start: int,
	doc_len: int,
	cfg: SampleConfig,
	rng: random.Random,
) -> None:
	max_span = min(cfg.mask_span_max, doc_len)
	if max_span < cfg.mask_span_min:
		return
	spans = rng.randint(1, cfg.mask_spans_max)
	for _ in range(spans):
		span_len = rng.randint(cfg.mask_span_min, max_span)
		span_start = rng.randint(doc_start, doc_start + doc_len - span_len)
		labels[span_start:span_start + span_len] = [-100] * span_len


def build_sample(cfg: SampleConfig, rng: random.Random) -> Tuple[List[int], List[int], List[int], List[int]]:
	doc_lengths = build_doc_lengths(cfg, rng)
	total_tokens = sum(doc_lengths)
	if total_tokens > cfg.seq_len:
		raise RuntimeError("doc lengths exceed seq_len")

	input_ids = [rng.randint(1, cfg.vocab_size - 1) for _ in range(total_tokens)]
	padding = cfg.seq_len - total_tokens
	if padding:
		input_ids.extend([cfg.pad_id] * padding)

	labels = input_ids.copy()
	running = 0
	for doc_len in doc_lengths:
		mask_doc_spans(labels, running, doc_len, cfg, rng)
		running += doc_len
	if padding:
		labels[-padding:] = [-100] * padding

	attn_mask = [1] * total_tokens + [0] * padding

	cum_seq_len = [0]
	running = 0
	for length in doc_lengths:
		running += length
		cum_seq_len.append(running)
	if cum_seq_len[-1] != cfg.seq_len:
		cum_seq_len.append(cfg.seq_len)

	return input_ids, labels, cum_seq_len, attn_mask


def write_shard(out_dir: str, samples: int, cfg: SampleConfig, seed: int) -> None:
	os.makedirs(out_dir, exist_ok=True)
	columns = {
		"input_ids": f"ndarray:int32:{cfg.seq_len}",
		"labels": f"ndarray:int32:{cfg.seq_len}",
		"cum_seq_len": "ndarray:int32",
		"attn_mask": f"ndarray:int32:{cfg.seq_len}",
	}

	rng = random.Random(seed)
	with MDSWriter(out=out_dir, columns=columns) as out:
		for _ in range(samples):
			input_ids, labels, cum_seq_len, attn_mask = build_sample(cfg, rng)
			input_ids_arr = np.asarray(input_ids, dtype=np.int32)
			labels_arr = np.asarray(labels, dtype=np.int32)
			cum_seq_len_arr = np.asarray(cum_seq_len, dtype=np.int32)
			attn_mask_arr = np.asarray(attn_mask, dtype=np.int32)
			out.write(
				{
					"input_ids": input_ids_arr,
					"labels": labels_arr,
					"cum_seq_len": cum_seq_len_arr,
					"attn_mask": attn_mask_arr,
				}
			)


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Generate mock MDS dataset for SFT.")
	parser.add_argument("--out-root", default="./data/sft_mock_mds", help="Output root directory")
	parser.add_argument("--num-groups", type=int, default=4, help="Number of shard groups")
	parser.add_argument("--samples-per-group", type=int, default=100, help="Samples per shard group")
	parser.add_argument("--seq-len", type=int, default=4096, help="Fixed sequence length")
	parser.add_argument("--vocab-size", type=int, default=32000, help="Token vocab size")
	parser.add_argument("--pad-id", type=int, default=0, help="Padding token id")
	parser.add_argument("--min-docs", type=int, default=2, help="Minimum docs per sample")
	parser.add_argument("--max-docs", type=int, default=5, help="Maximum docs per sample")
	parser.add_argument("--min-doc-len", type=int, default=256, help="Minimum doc length")
	parser.add_argument("--max-doc-len", type=int, default=2048, help="Maximum doc length")
	parser.add_argument("--mask-span-min", type=int, default=8, help="Minimum masked span length")
	parser.add_argument("--mask-span-max", type=int, default=128, help="Maximum masked span length")
	parser.add_argument("--mask-spans-max", type=int, default=3, help="Max masked spans per doc")
	parser.add_argument("--seed", type=int, default=13, help="Random seed")
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	if args.num_groups <= 0:
		raise ValueError("num_groups must be positive")
	if args.samples_per_group <= 0:
		raise ValueError("samples_per_group must be positive")

	cfg = SampleConfig(
		seq_len=args.seq_len,
		vocab_size=args.vocab_size,
		pad_id=args.pad_id,
		min_docs=args.min_docs,
		max_docs=args.max_docs,
		min_doc_len=args.min_doc_len,
		max_doc_len=args.max_doc_len,
		mask_span_min=args.mask_span_min,
		mask_span_max=args.mask_span_max,
		mask_spans_max=args.mask_spans_max,
	)
	validate_config(cfg)

	for group_id in range(args.num_groups):
		sub_out = os.path.join(args.out_root, f"shard_{group_id}")
		seed = args.seed + group_id
		write_shard(sub_out, args.samples_per_group, cfg, seed)

	merge_index(args.out_root)


if __name__ == "__main__":
	main()
