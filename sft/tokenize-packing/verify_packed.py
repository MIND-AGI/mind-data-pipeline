"""Verify a Stage-2 packed MDS dataset and print detokenized samples.

Two jobs:
1. Walk every pack and assert the packing invariants (lengths, ``cu_seqlens``
   convention, tail-only padding, label/pad consistency). Aggregate fill-rate
   stats.
2. For the first ``--show`` packs, decode each real segment back to text and
   show both the full segment and the assistant-only training target
   (positions where ``labels != -100``).

Run directly with plain python (no torchrun) — it reads the merged top-level
``index.json`` produced by Stage 2.
"""

from __future__ import annotations

import argparse
from typing import List, Optional, Tuple

import numpy as np
from streaming import StreamingDataset


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Verify packed MDS + show detokenized samples")
    p.add_argument("--data-root", required=True, help="Packed MDS dir (Stage 2 --out-root)")
    p.add_argument(
        "--tokenizer",
        default="/mnt/nfs/liuzehao/modelbase/Nemo/Nemotron-Cascade-2-30B-A3B",
        help="HF tokenizer path/name for decoding the samples",
    )
    p.add_argument("--pad-token-id", type=int, default=11, help="Pad token id used at Stage 2")
    p.add_argument(
        "--num-check",
        type=int,
        default=0,
        help="How many packs to validate. 0 = all (validation is cheap, numpy-only).",
    )
    p.add_argument("--show", type=int, default=2, help="How many packs to print detokenized.")
    p.add_argument(
        "--segments-per-pack",
        type=int,
        default=3,
        help="How many real segments to print per shown pack.",
    )
    p.add_argument(
        "--max-chars",
        type=int,
        default=1200,
        help="Truncate each decoded text block to this many chars (head+tail kept).",
    )
    p.add_argument("--trust-remote-code", action="store_true")
    return p.parse_args()


def _truncate(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    head = max_chars * 2 // 3
    tail = max_chars - head
    omitted = len(text) - head - tail
    return f"{text[:head]}\n  ... [{omitted} chars omitted] ...\n{text[-tail:]}"


def check_pack(sample: dict, idx: int, max_len: int, pad_id: int) -> Tuple[List[str], dict]:
    """Validate one pack. Returns (errors, stats)."""
    errors: List[str] = []
    input_ids = np.asarray(sample["input_ids"])
    labels = np.asarray(sample["labels"])
    cu = np.asarray(sample["cu_seqlens"]).tolist()
    attn = np.asarray(sample["attn_mask"])

    # 1. lengths
    for name, arr in (("input_ids", input_ids), ("labels", labels), ("attn_mask", attn)):
        if len(arr) != max_len:
            errors.append(f"len({name})={len(arr)} != max_pack_length {max_len}")

    # 2. cu_seqlens convention
    if not cu or cu[0] != 0:
        errors.append(f"cu_seqlens[0]={cu[0] if cu else None} != 0")
    if not cu or cu[-1] != max_len:
        errors.append(f"cu_seqlens[-1]={cu[-1] if cu else None} != max_pack_length {max_len}")
    if any(cu[i + 1] <= cu[i] for i in range(len(cu) - 1)):
        errors.append(f"cu_seqlens not strictly increasing: {cu}")

    # 3. tail-only padding via attn_mask: must be 1...1 0...0
    real = int(attn.sum())
    if not np.array_equal(attn[:real], np.ones(real, dtype=attn.dtype)):
        errors.append("attn_mask has a 0 before the padding tail (mid-pack padding?)")
    if real < max_len and not np.array_equal(attn[real:], np.zeros(max_len - real, dtype=attn.dtype)):
        errors.append("attn_mask not contiguous 1s-then-0s")

    pad = max_len - real
    # 4. pad region consistency
    if pad > 0:
        if not np.all(input_ids[real:] == pad_id):
            errors.append(f"input_ids padding region not all pad_id={pad_id}")
        if not np.all(labels[real:] == -100):
            errors.append("labels padding region not all -100")
        # pad must be encoded as the final cu segment
        if len(cu) < 2 or cu[-2] != real:
            errors.append(f"padded pack but cu_seqlens[-2]={cu[-2] if len(cu) >= 2 else None} != real {real}")

    # 5. every real segment must carry at least one trained token
    n_pad_seg = 1 if pad > 0 else 0
    real_segments = len(cu) - 1 - n_pad_seg
    for s in range(real_segments):
        seg = labels[cu[s] : cu[s + 1]]
        if not np.any(seg != -100):
            errors.append(f"real segment {s} [{cu[s]}:{cu[s + 1]}) has no trained token (all -100)")

    stats = {
        "real": real,
        "pad": pad,
        "real_segments": real_segments,
        "trained": int(np.sum(labels != -100)),
    }
    return errors, stats


def show_pack(sample: dict, idx: int, tokenizer, max_len: int, n_segments: int, max_chars: int) -> None:
    input_ids = np.asarray(sample["input_ids"])
    labels = np.asarray(sample["labels"])
    cu = np.asarray(sample["cu_seqlens"]).tolist()
    attn = np.asarray(sample["attn_mask"])

    real = int(attn.sum())
    pad = max_len - real
    n_pad_seg = 1 if pad > 0 else 0
    real_segments = len(cu) - 1 - n_pad_seg
    fill = 100.0 * real / max_len

    print("=" * 78)
    print(f"PACK {idx}")
    print(
        f"  length={max_len}  real_tokens={real} ({fill:.1f}%)  pad={pad}  "
        f"real_segments={real_segments}  trained_tokens={int(np.sum(labels != -100))}"
    )
    head_cu = cu[:9] + (["..."] if len(cu) > 9 else [])
    print(f"  cu_seqlens (head): {head_cu}")
    print(f"  cu_seqlens (tail): {cu[-3:]}")

    for s in range(min(n_segments, real_segments)):
        lo, hi = cu[s], cu[s + 1]
        seg_ids = input_ids[lo:hi].tolist()
        seg_labels = labels[lo:hi]
        target_ids = [int(t) for t in seg_labels[seg_labels != -100].tolist()]

        seg_text = tokenizer.decode(seg_ids, skip_special_tokens=False)
        target_text = tokenizer.decode(target_ids, skip_special_tokens=False) if target_ids else "(none)"

        print(f"\n  -- segment {s}: tokens [{lo}:{hi}) len={hi - lo}, trained={len(target_ids)} --")
        print("  [INPUT decoded]:")
        print("  " + _truncate(seg_text, max_chars).replace("\n", "\n  "))
        print("  [TRAIN TARGET decoded (assistant-only)]:")
        print("  " + _truncate(target_text, max_chars).replace("\n", "\n  "))
    print("=" * 78)
    print()


def main() -> int:
    args = parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=args.trust_remote_code)

    ds = StreamingDataset(local=args.data_root, batch_size=1, shuffle=False)
    total = len(ds)
    if total == 0:
        raise SystemExit(f"no packs found under {args.data_root}")

    max_len = int(len(np.asarray(ds[0]["input_ids"])))
    print(f"opened packed MDS: {args.data_root}")
    print(f"  packs={total}  inferred max_pack_length={max_len}  pad_token_id={args.pad_token_id}\n")

    n_check = total if args.num_check <= 0 else min(args.num_check, total)

    n_bad = 0
    sum_real = 0
    sum_pad = 0
    sum_segments = 0
    sum_trained = 0
    first_errors: List[str] = []

    for i in range(n_check):
        sample = ds[i]
        errors, stats = check_pack(sample, i, max_len, args.pad_token_id)
        sum_real += stats["real"]
        sum_pad += stats["pad"]
        sum_segments += stats["real_segments"]
        sum_trained += stats["trained"]
        if errors:
            n_bad += 1
            if len(first_errors) < 10:
                first_errors.append(f"pack {i}: " + "; ".join(errors))

    total_tokens = n_check * max_len
    print("---- validation ----")
    print(f"  checked {n_check} packs")
    print(f"  packs with errors: {n_bad}")
    if first_errors:
        print("  first errors:")
        for e in first_errors:
            print(f"    - {e}")
    print(f"  fill rate: {100.0 * sum_real / total_tokens:.2f}%  "
          f"(real={sum_real}, pad={sum_pad}, total={total_tokens})")
    print(f"  trained-token rate: {100.0 * sum_trained / total_tokens:.2f}%  (trained={sum_trained})")
    print(f"  avg real segments per pack: {sum_segments / n_check:.1f}")
    print(f"  result: {'PASS' if n_bad == 0 else 'FAIL'}\n")

    n_show = min(args.show, total)
    if n_show > 0:
        print(f"---- detokenized samples (first {n_show} packs) ----\n")
        for i in range(n_show):
            show_pack(ds[i], i, tokenizer, max_len, args.segments_per_pack, args.max_chars)

    return 0 if n_bad == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
