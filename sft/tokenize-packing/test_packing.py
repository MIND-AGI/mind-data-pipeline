"""Unit tests for GreedyPacker. Pure-python, no torch/transformers needed.

Run with: python test_packing.py
"""

from __future__ import annotations

from packing_utils import GreedyPacker


def _assert_pack_invariants(pack: dict, max_pack_length: int, pad_token_id: int) -> None:
    assert len(pack["input_ids"]) == max_pack_length
    assert len(pack["labels"]) == max_pack_length
    assert len(pack["attn_mask"]) == max_pack_length
    assert pack["cu_seqlens"][0] == 0
    assert pack["cu_seqlens"][-1] == max_pack_length
    # cu_seqlens monotonically non-decreasing
    for a, b in zip(pack["cu_seqlens"], pack["cu_seqlens"][1:]):
        assert a < b, f"cu_seqlens not strictly increasing: {pack['cu_seqlens']}"

    # attn_mask: 1s then 0s, no mid-pack zeros
    found_zero = False
    for m in pack["attn_mask"]:
        if found_zero:
            assert m == 0, "padding only allowed at the tail"
        elif m == 0:
            found_zero = True

    real_len = sum(pack["attn_mask"])
    # Padding region in input_ids/labels must be exactly pad_token_id / -100.
    for i in range(real_len, max_pack_length):
        assert pack["input_ids"][i] == pad_token_id
        assert pack["labels"][i] == -100


def test_single_sample_exact_fit():
    packer = GreedyPacker(max_pack_length=8, pad_token_id=0, buffer_size=1)
    packer.append([1, 2, 3, 4, 5, 6, 7, 8], [10, 20, 30, 40, 50, 60, 70, 80])
    packs = list(packer.flush())
    assert len(packs) == 1
    p = packs[0]
    _assert_pack_invariants(p, 8, 0)
    assert p["cu_seqlens"] == [0, 8]
    assert sum(p["attn_mask"]) == 8
    print("ok: single_sample_exact_fit")


def test_single_sample_underfilled():
    packer = GreedyPacker(max_pack_length=8, pad_token_id=99, buffer_size=1)
    packer.append([1, 2, 3], [10, 20, 30])
    packs = list(packer.flush())
    assert len(packs) == 1
    p = packs[0]
    _assert_pack_invariants(p, 8, 99)
    # one real seg + one padding seg
    assert p["cu_seqlens"] == [0, 3, 8]
    assert p["input_ids"] == [1, 2, 3, 99, 99, 99, 99, 99]
    assert p["labels"] == [10, 20, 30, -100, -100, -100, -100, -100]
    assert p["attn_mask"] == [1, 1, 1, 0, 0, 0, 0, 0]
    print("ok: single_sample_underfilled")


def test_two_samples_fit_exactly():
    packer = GreedyPacker(max_pack_length=8, pad_token_id=0, buffer_size=1)
    packer.append([1, 2, 3], [11, 22, 33])
    packer.append([4, 5, 6, 7, 8], [44, 55, 66, 77, 88])
    packs = list(packer.flush())
    assert len(packs) == 1
    p = packs[0]
    _assert_pack_invariants(p, 8, 0)
    assert p["cu_seqlens"] == [0, 3, 8]
    assert p["input_ids"] == [1, 2, 3, 4, 5, 6, 7, 8]
    assert p["labels"] == [11, 22, 33, 44, 55, 66, 77, 88]
    print("ok: two_samples_fit_exactly")


def test_first_fit_skips_oversize_keeps_for_next():
    packer = GreedyPacker(max_pack_length=8, pad_token_id=0, buffer_size=1)
    # First sample of length 5; second of length 6 (won't fit after first);
    # third of length 3 (fits after first).
    packer.append([1, 2, 3, 4, 5], [1, 2, 3, 4, 5])  # len 5
    packer.append([6, 7, 8, 9, 10, 11], [6, 7, 8, 9, 10, 11])  # len 6, won't fit after 5
    packer.append([12, 13, 14], [12, 13, 14])  # len 3, fits after 5
    packs = list(packer.flush())
    assert len(packs) == 2
    p0, p1 = packs
    _assert_pack_invariants(p0, 8, 0)
    _assert_pack_invariants(p1, 8, 0)
    # First pack: sample 0 (len 5) + sample 2 (len 3) = 8, no padding
    assert p0["cu_seqlens"] == [0, 5, 8]
    assert p0["input_ids"] == [1, 2, 3, 4, 5, 12, 13, 14]
    # Second pack: sample 1 (len 6) + 2 padding
    assert p1["cu_seqlens"] == [0, 6, 8]
    assert p1["input_ids"] == [6, 7, 8, 9, 10, 11, 0, 0]
    assert p1["attn_mask"] == [1, 1, 1, 1, 1, 1, 0, 0]
    print("ok: first_fit_skips_oversize_keeps_for_next")


def test_oversize_raises():
    packer = GreedyPacker(max_pack_length=4, pad_token_id=0, buffer_size=1)
    try:
        packer.append([1, 2, 3, 4, 5], [1, 2, 3, 4, 5])
    except ValueError:
        print("ok: oversize_raises")
        return
    raise AssertionError("expected ValueError for oversize sample")


def test_streaming_emit_buffer_threshold():
    packer = GreedyPacker(max_pack_length=4, pad_token_id=0, buffer_size=3)
    # Before threshold: no emit.
    packer.append([1, 2], [1, 2])
    packer.append([3, 4], [3, 4])
    assert list(packer.emit_ready()) == []
    # At threshold: emit_ready should produce packs.
    packer.append([5, 6], [5, 6])
    out = list(packer.emit_ready())
    # All three are length 2; two fit per pack of length 4.
    # First emit: pack0 takes items 0,1 (len 4). Active drops to 1 (<3) -> stop.
    assert len(out) == 1, f"unexpected: {out}"
    _assert_pack_invariants(out[0], 4, 0)
    # Flush remaining
    rest = list(packer.flush())
    assert len(rest) == 1
    _assert_pack_invariants(rest[0], 4, 0)
    print("ok: streaming_emit_buffer_threshold")


def test_no_mid_pack_padding():
    """Padding must only appear at the tail, never between two real sequences."""
    packer = GreedyPacker(max_pack_length=10, pad_token_id=0, buffer_size=1)
    packer.append([1, 1, 1], [1, 1, 1])
    packer.append([2, 2], [2, 2])
    packer.append([3, 3, 3, 3], [3, 3, 3, 3])
    # Total real tokens = 9, one pack of 10 -> 1 padding at the end.
    packs = list(packer.flush())
    assert len(packs) == 1
    p = packs[0]
    _assert_pack_invariants(p, 10, 0)
    assert p["cu_seqlens"] == [0, 3, 5, 9, 10]
    assert p["attn_mask"] == [1, 1, 1, 1, 1, 1, 1, 1, 1, 0]
    print("ok: no_mid_pack_padding")


def main() -> int:
    test_single_sample_exact_fit()
    test_single_sample_underfilled()
    test_two_samples_fit_exactly()
    test_first_fit_skips_oversize_keeps_for_next()
    test_oversize_raises()
    test_streaming_emit_buffer_threshold()
    test_no_mid_pack_padding()
    print("all tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
